"""
Dual Q-Network: RegularQNetwork + AssemblyQNetwork
===================================================
Regular → Assembly: C_p 컴포넌트 진행 상태 벡터 (hidden_dim + 3)
Assembly → Regular: U_p 조립 긴급도 신호 벡터 (hidden_dim + 3)

학습: Window Best-Trajectory DQN (동시 업데이트)
  - Regular + Assembly: ep 5, 10, 15 … 동시 업데이트 (Regular 먼저)
  - Target update: 각 네트워크 2 update cycle마다 독립적으로
  - Gradient detach: 상대 네트워크 출력은 항상 detach() 후 입력
  - Target 계산: 다음 상태의 Regular / Assembly Q값 모두 반영 (글로벌 max)
"""

import copy
import random
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import GATConv

from ffsa_instance import OpFeat

RegularAction = Tuple[int, int]
AssemblyAction = Tuple[Tuple[int, ...], int]
Action = Union[RegularAction, AssemblyAction]


def _is_assembly(action: Action) -> bool:
    return isinstance(action[0], tuple)


class ReplayBuffer:
    """Regular / Assembly 각각 별도 인스턴스로 사용하는 경험 리플레이 버퍼."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action_idx: int, reward: float, next_obs, done: bool):
        self.buffer.append((obs, action_idx, reward, next_obs, done))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ──────────────────────────────────────────────────────────
# 공통 GNN 인코더 (두 네트워크가 각자 독립적으로 보유)
# ──────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    op / machine 노드를 GAT + theta MLP로 인코딩.
    RegularQNetwork 와 AssemblyQNetwork 각자 독립적인 가중치를 가짐.
    """

    def __init__(self, op_feat_dim, machine_feat_dim, edge_feat_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.edge_feat_dim = edge_feat_dim

        self.op_encoder      = nn.Linear(op_feat_dim, hidden_dim)
        self.machine_encoder = nn.Linear(machine_feat_dim, hidden_dim)

        self.gat_op2m = nn.ModuleList([
            GATConv((hidden_dim, hidden_dim), hidden_dim,
                    edge_dim=edge_feat_dim, add_self_loops=False)
            for _ in range(num_layers)
        ])
        self.gat_m2o = nn.ModuleList([
            GATConv((hidden_dim, hidden_dim), hidden_dim,
                    edge_dim=edge_feat_dim, add_self_loops=False)
            for _ in range(num_layers)
        ])

        self.theta1 = nn.Linear(hidden_dim, hidden_dim)
        self.theta2 = nn.Linear(hidden_dim, hidden_dim)
        self.theta3 = nn.Linear(hidden_dim, hidden_dim)
        self.theta4 = nn.Linear(hidden_dim, hidden_dim)
        self.theta0 = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.ELU())

    def forward(self, graph_data, precedence_info, op_id_to_idx, device):
        op_x      = graph_data['op'].x.to(device)
        machine_x = graph_data['machine'].x.to(device)
        machine_h = self.machine_encoder(machine_x)

        if op_x.size(0) == 0:
            return torch.zeros((0, self.hidden_dim), device=device), machine_h

        op_h = self.op_encoder(op_x)

        cand_edge = graph_data['op', 'candidate', 'machine'].edge_index.to(device)
        cand_attr = graph_data['op', 'candidate', 'machine'].edge_attr.to(device)
        rev_edge  = graph_data['machine', 'candidate_rev', 'op'].edge_index.to(device)
        rev_attr  = graph_data['machine', 'candidate_rev', 'op'].edge_attr.to(device)

        for i in range(self.num_layers):
            if cand_edge.size(1) > 0:
                machine_h = F.elu(self.gat_op2m[i]((op_h, machine_h), cand_edge, edge_attr=cand_attr) + machine_h)
            if rev_edge.size(1) > 0:
                op_h = F.elu(self.gat_m2o[i]((machine_h, op_h), rev_edge, edge_attr=rev_attr) + op_h)

        prev_map           = precedence_info['prev_map']
        next_map           = precedence_info['next_map']
        candidate_machines = precedence_info['candidate_machines']
        num_ops            = op_h.size(0)
        zero_pad           = torch.zeros((1, self.hidden_dim), device=device)
        op_h_padded        = torch.cat([op_h, zero_pad], dim=0)

        sorted_op_ids = sorted(op_id_to_idx, key=op_id_to_idx.get)
        prev_idx = torch.tensor(
            [op_id_to_idx.get(prev_map.get(oid), num_ops)
             if prev_map.get(oid) is not None else num_ops
             for oid in sorted_op_ids], dtype=torch.long, device=device)
        next_idx = torch.tensor(
            [op_id_to_idx.get(next_map.get(oid), num_ops)
             if next_map.get(oid) is not None else num_ops
             for oid in sorted_op_ids], dtype=torch.long, device=device)

        prev_h = op_h_padded[prev_idx]
        next_h = op_h_padded[next_idx]

        machine_mean = torch.zeros((num_ops, self.hidden_dim), device=device)
        for op_id, idx in op_id_to_idx.items():
            valid_m = [m for m in candidate_machines.get(op_id, []) if m < machine_h.size(0)]
            if valid_m:
                machine_mean[idx] = machine_h[valid_m].mean(dim=0)

        op_h = self.theta0(torch.cat([
            F.elu(self.theta1(prev_h)),
            F.elu(self.theta2(next_h)),
            F.elu(self.theta3(machine_mean)),
            F.elu(self.theta4(op_h)),
        ], dim=-1))

        return op_h, machine_h

    def forward_batch(self, graphs_and_info, device):
        """
        여러 그래프를 한 번에 인코딩 (PyG mega-graph).
        graphs_and_info: list of (graph_data, precedence_info, op_id_to_idx)
        Returns: list of (op_h, machine_h) — 순서 보존
        """
        n       = len(graphs_and_info)
        results = [None] * n

        non_empty_indices = []
        for i, (g, _, _) in enumerate(graphs_and_info):
            if g['op'].x.size(0) == 0:
                machine_h  = self.machine_encoder(g['machine'].x.to(device))
                results[i] = (torch.zeros((0, self.hidden_dim), device=device), machine_h)
            else:
                non_empty_indices.append(i)

        if not non_empty_indices:
            return results

        # ── 1. mega-graph 생성 및 선형 인코딩 ──
        graph_list = [graphs_and_info[i][0] for i in non_empty_indices]
        batched    = PyGBatch.from_data_list(graph_list).to(device)

        op_h_b      = self.op_encoder(batched['op'].x)
        machine_h_b = self.machine_encoder(batched['machine'].x)

        cand_edge = batched['op', 'candidate', 'machine'].edge_index
        cand_attr = batched['op', 'candidate', 'machine'].edge_attr
        rev_edge  = batched['machine', 'candidate_rev', 'op'].edge_index
        rev_attr  = batched['machine', 'candidate_rev', 'op'].edge_attr

        # ── 2. GAT (한 번의 메가-그래프 순전파) ──
        for i in range(self.num_layers):
            if cand_edge.size(1) > 0:
                machine_h_b = F.elu(self.gat_op2m[i]((op_h_b, machine_h_b), cand_edge, edge_attr=cand_attr) + machine_h_b)
            if rev_edge.size(1) > 0:
                op_h_b = F.elu(self.gat_m2o[i]((machine_h_b, op_h_b), rev_edge, edge_attr=rev_attr) + op_h_b)

        op_batch_vec      = batched['op'].batch
        machine_batch_vec = batched['machine'].batch

        # ── 3. 그래프별 분할 + theta 집계 ──
        for local_i, orig_i in enumerate(non_empty_indices):
            _, prec, op_id_to_idx = graphs_and_info[orig_i]

            op_h      = op_h_b[op_batch_vec == local_i]
            machine_h = machine_h_b[machine_batch_vec == local_i]

            num_ops     = op_h.size(0)
            zero_pad    = torch.zeros((1, self.hidden_dim), device=device)
            op_h_padded = torch.cat([op_h, zero_pad], dim=0)

            prev_map           = prec['prev_map']
            next_map           = prec['next_map']
            candidate_machines = prec['candidate_machines']

            sorted_op_ids = sorted(op_id_to_idx, key=op_id_to_idx.get)
            prev_idx = torch.tensor(
                [op_id_to_idx.get(prev_map.get(oid), num_ops)
                 if prev_map.get(oid) is not None else num_ops
                 for oid in sorted_op_ids], dtype=torch.long, device=device)
            next_idx = torch.tensor(
                [op_id_to_idx.get(next_map.get(oid), num_ops)
                 if next_map.get(oid) is not None else num_ops
                 for oid in sorted_op_ids], dtype=torch.long, device=device)

            prev_h = op_h_padded[prev_idx]
            next_h = op_h_padded[next_idx]

            machine_mean = torch.zeros((num_ops, self.hidden_dim), device=device)
            for op_id, idx in op_id_to_idx.items():
                valid_m = [m for m in candidate_machines.get(op_id, []) if m < machine_h.size(0)]
                if valid_m:
                    machine_mean[idx] = machine_h[valid_m].mean(dim=0)

            op_h = self.theta0(torch.cat([
                F.elu(self.theta1(prev_h)),
                F.elu(self.theta2(next_h)),
                F.elu(self.theta3(machine_mean)),
                F.elu(self.theta4(op_h)),
            ], dim=-1))

            results[orig_i] = (op_h, machine_h)

        return results


# ──────────────────────────────────────────────────────────
# RegularQNetwork
# ──────────────────────────────────────────────────────────

class RegularQNetwork(nn.Module):
    """
    일반 공정 Q-network.

    추가 입력 U_p (Assembly → Regular 긴급도 신호, shape: hidden_dim+3):
      [0:hd]  GNN 임베딩 평균 (Assembly 네트워크 관점의 조립 상태)
      [hd]    urgency_score  (조립 op weight/due 기반 긴급도)
      [hd+1]  shortage_rate  (pool 미완성 제품 비율)
      [hd+2]  inactive_pressure (조립 대기 final job 비율)

    q_mlp 입력 차원:
      op_emb(hd) | machine_emb(hd) | graph_emb(2hd) | edge_feat(2) | U_p(hd+3)
      = 5*hd + 5
    """

    def __init__(self, op_feat_dim=OpFeat.DIM, machine_feat_dim=6, edge_feat_dim=2,
                 hidden_dim=16, num_layers=2, mlp_hidden=128):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.cross_dim     = hidden_dim + 3

        self.encoder = GNNEncoder(
            op_feat_dim, machine_feat_dim, edge_feat_dim, hidden_dim, num_layers
        )
        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + edge_feat_dim + self.cross_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def _edge_feat_map(self, graph_data, device):
        emap = {}
        ei   = graph_data['op', 'candidate', 'machine'].edge_index
        ea   = graph_data['op', 'candidate', 'machine'].edge_attr
        for i in range(ei.size(1)):
            emap[(ei[0, i].item(), ei[1, i].item())] = ea[i].to(device)
        return emap

    def forward(self, graph_data, actions, precedence_info, op_id_to_idx, u_p=None):
        """
        Returns Q-values for regular actions only, shape [n_regular].
        u_p: 긴급도 신호 (Assembly 출력 detach), None이면 영벡터 사용.
        """
        device   = next(self.parameters()).device
        reg_acts = [a for a in actions if not _is_assembly(a)]

        if not reg_acts or graph_data['op'].x.size(0) == 0:
            return torch.zeros(max(len(reg_acts), 1), device=device)

        op_h, machine_h = self.encoder(graph_data, precedence_info, op_id_to_idx, device)
        graph_emb       = torch.cat([op_h.mean(dim=0), machine_h.mean(dim=0)])
        emap            = self._edge_feat_map(graph_data, device)

        if u_p is None:
            u_p = torch.zeros(self.cross_dim, device=device)

        q_vals = []
        for op_id, mid in reg_acts:
            op_idx = op_id_to_idx.get(op_id)
            if op_idx is None or op_idx >= op_h.size(0) or mid >= machine_h.size(0):
                q_vals.append(torch.tensor(-1e9, device=device))
                continue
            ef  = emap.get((op_idx, mid), torch.zeros(self.edge_feat_dim, device=device))
            inp = torch.cat([op_h[op_idx], machine_h[mid], graph_emb, ef, u_p])
            q_vals.append(self.q_mlp(inp.unsqueeze(0)).squeeze())

        return torch.stack(q_vals) if q_vals else torch.zeros(1, device=device)

    def compute_C_p(self, graph_data, precedence_info, op_id_to_idx, obs):
        """
        C_p: Regular → Assembly 전달 벡터 (hidden_dim + 3).
          [0:hd]  GNN op 임베딩 평균 (컴포넌트 처리 상태 표현)
          [hd]    pool_count_norm  : assembly pool 내 총 컴포넌트 수 (정규화)
          [hd+1]  pool_ready_rate  : 2종 이상 갖춘 제품 비율
          [hd+2]  inactive_norm    : 미활성 final job 평균 수 (정규화)
        """
        device        = next(self.parameters()).device
        assembly_pool = obs.get("assembly_pool", {})
        inactive      = obs.get("inactive_final_jobs", {})

        pool_total = sum(
            len(jobs)
            for type_pool in assembly_pool.values()
            for jobs in type_pool.values()
        )
        pool_count_norm = min(pool_total / max(len(assembly_pool) * 5, 1), 1.0)

        if assembly_pool:
            ready           = sum(1 for tp in assembly_pool.values() if len(tp) >= 2)
            pool_ready_rate = ready / len(assembly_pool)
        else:
            pool_ready_rate = 0.0

        inactive_counts = [len(v) for v in inactive.values()]
        inactive_norm   = min(float(np.mean(inactive_counts)) / 5.0, 1.0) if inactive_counts else 0.0

        with torch.no_grad():
            if graph_data['op'].x.size(0) > 0:
                op_h, _ = self.encoder(graph_data, precedence_info, op_id_to_idx, device)
                gnn_emb  = op_h.mean(dim=0)
            else:
                gnn_emb = torch.zeros(self.hidden_dim, device=device)

        scalar = torch.tensor(
            [pool_count_norm, pool_ready_rate, inactive_norm],
            dtype=torch.float32, device=device,
        )
        return torch.cat([gnn_emb.detach(), scalar])

    def compute_C_p_from_embeddings(self, op_h: torch.Tensor, obs: dict) -> torch.Tensor:
        """사전 계산된 op_h로 C_p 산출 — 인코더 재호출 없음."""
        device        = op_h.device
        assembly_pool = obs.get("assembly_pool", {})
        inactive      = obs.get("inactive_final_jobs", {})

        pool_total = sum(
            len(jobs)
            for type_pool in assembly_pool.values()
            for jobs in type_pool.values()
        )
        pool_count_norm = min(pool_total / max(len(assembly_pool) * 5, 1), 1.0)

        if assembly_pool:
            ready           = sum(1 for tp in assembly_pool.values() if len(tp) >= 2)
            pool_ready_rate = ready / len(assembly_pool)
        else:
            pool_ready_rate = 0.0

        inactive_counts = [len(v) for v in inactive.values()]
        inactive_norm   = min(float(np.mean(inactive_counts)) / 5.0, 1.0) if inactive_counts else 0.0

        gnn_emb = op_h.mean(dim=0) if op_h.size(0) > 0 else torch.zeros(self.hidden_dim, device=device)
        scalar  = torch.tensor([pool_count_norm, pool_ready_rate, inactive_norm],
                               dtype=torch.float32, device=device)
        return torch.cat([gnn_emb.detach(), scalar])

    def forward_from_embeddings(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        actions: list,
        graph_data,
        op_id_to_idx: dict,
        u_p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """사전 계산된 임베딩으로 Regular Q-값 산출 — MLP 한 번 호출."""
        device   = op_h.device
        reg_acts = [a for a in actions if not _is_assembly(a)]
        if not reg_acts:
            return torch.zeros(1, device=device)

        graph_emb = torch.cat([op_h.mean(dim=0), machine_h.mean(dim=0)])
        if u_p is None:
            u_p = torch.zeros(self.cross_dim, device=device)

        ei = graph_data['op', 'candidate', 'machine'].edge_index.to(device)
        ea = graph_data['op', 'candidate', 'machine'].edge_attr.to(device)
        ef_matrix = torch.zeros(
            max(op_h.size(0), 1), max(machine_h.size(0), 1), self.edge_feat_dim, device=device
        )
        if ei.size(1) > 0:
            ef_matrix[ei[0], ei[1]] = ea

        valid_inps, valid_locals = [], []
        for local_i, (op_id, mid) in enumerate(reg_acts):
            op_idx = op_id_to_idx.get(op_id)
            if op_idx is None or op_idx >= op_h.size(0) or mid >= machine_h.size(0):
                continue
            inp = torch.cat([op_h[op_idx], machine_h[mid], graph_emb, ef_matrix[op_idx, mid], u_p])
            valid_inps.append(inp)
            valid_locals.append(local_i)

        if not valid_inps:
            return torch.full((len(reg_acts),), -1e9, device=device)

        q_computed = self.q_mlp(torch.stack(valid_inps)).squeeze(-1)

        if len(valid_locals) == len(reg_acts):
            return q_computed

        valid_set = set(valid_locals)
        q_full, ptr = [], 0
        for local_i in range(len(reg_acts)):
            if local_i in valid_set:
                q_full.append(q_computed[ptr]); ptr += 1
            else:
                q_full.append(torch.tensor(-1e9, device=device))
        return torch.stack(q_full)


# ──────────────────────────────────────────────────────────
# AssemblyQNetwork
# ──────────────────────────────────────────────────────────

class AssemblyQNetwork(nn.Module):
    """
    조립 공정 Q-network.

    추가 입력 C_p (Regular → Assembly 컴포넌트 진행 상태, shape: hidden_dim+3):
      [0:hd]  GNN 임베딩 평균 (Regular 네트워크 관점의 컴포넌트 상태)
      [hd]    pool_count_norm  (pool 내 컴포넌트 수 정규화)
      [hd+1]  pool_ready_rate  (모든 타입 갖춘 제품 비율)
      [hd+2]  inactive_norm    (미활성 final job 수 정규화)

    q_asm_mlp 입력 차원:
      comp_pool(hd) | machine_emb(hd) | graph_emb(2hd) | C_p(hd+3)
      = 5*hd + 3
    """

    def __init__(self, op_feat_dim=OpFeat.DIM, machine_feat_dim=6, edge_feat_dim=2,
                 hidden_dim=16, num_layers=2, mlp_hidden=128):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.cross_dim     = hidden_dim + 3

        self.encoder = GNNEncoder(
            op_feat_dim, machine_feat_dim, edge_feat_dim, hidden_dim, num_layers
        )
        self.q_asm_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + self.cross_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def _get_job_last_op_idx(self, job_id, op_id_to_idx, job_op_map):
        for op_id in reversed(job_op_map.get(job_id, [])):
            if op_id in op_id_to_idx:
                return op_id_to_idx[op_id]
        return None

    def forward(self, graph_data, actions, precedence_info, op_id_to_idx, job_op_map, c_p=None):
        """
        Returns Q-values for assembly actions only, shape [n_assembly].
        c_p: 컴포넌트 진행 신호 (Regular 출력 detach), None이면 영벡터 사용.
        """
        device   = next(self.parameters()).device
        asm_acts = [a for a in actions if _is_assembly(a)]

        if not asm_acts or graph_data['op'].x.size(0) == 0:
            return torch.zeros(max(len(asm_acts), 1), device=device)

        op_h, machine_h = self.encoder(graph_data, precedence_info, op_id_to_idx, device)
        graph_emb        = torch.cat([op_h.mean(dim=0), machine_h.mean(dim=0)])
        zero_h           = torch.zeros(self.hidden_dim, device=device)

        if c_p is None:
            c_p = torch.zeros(self.cross_dim, device=device)

        q_vals = []
        for comp_job_ids, mid in asm_acts:
            if mid >= machine_h.size(0):
                q_vals.append(torch.tensor(-1e9, device=device))
                continue
            comp_embs, valid = [], True
            for job_id in comp_job_ids:
                idx = self._get_job_last_op_idx(job_id, op_id_to_idx, job_op_map)
                if idx is None:
                    valid = False
                    break
                comp_embs.append(op_h[idx] if idx < op_h.size(0) else zero_h)
            if not valid:
                q_vals.append(torch.tensor(-1e9, device=device))
                continue
            comp_pool = torch.stack(comp_embs).mean(dim=0)
            inp = torch.cat([comp_pool, machine_h[mid], graph_emb, c_p])
            q_vals.append(self.q_asm_mlp(inp.unsqueeze(0)).squeeze())

        return torch.stack(q_vals) if q_vals else torch.zeros(1, device=device)

    def compute_U_p(self, graph_data, precedence_info, op_id_to_idx, obs):
        """
        U_p: Assembly → Regular 전달 벡터 (hidden_dim + 3).
          [0:hd]  GNN op 임베딩 평균 (조립 네트워크 관점의 상태)
          [hd]    urgency_score    (조립 op weight/due 기반 긴급도)
          [hd+1]  shortage_rate   (pool 미완성 제품 비율)
          [hd+2]  inactive_pressure (조립 대기 final job 비율)
        """
        device        = next(self.parameters()).device
        op_x          = graph_data['op'].x.to(device)
        assembly_pool = obs.get("assembly_pool", {})
        inactive      = obs.get("inactive_final_jobs", {})

        urgency_score = 0.0
        if op_x.size(0) > 0:
            is_asm   = op_x[:, OpFeat.IS_ASSEMBLY]
            due_norm = op_x[:, OpFeat.DUE_DATE_NORM]
            wt_norm  = op_x[:, OpFeat.WEIGHT_NORM]
            mask     = is_asm > 0.5
            if mask.any():
                urgency_score = float(
                    (wt_norm[mask] * (1.0 - due_norm[mask])).mean().item()
                )

        if assembly_pool:
            shortage      = sum(1 for tp in assembly_pool.values() if len(tp) < 2)
            shortage_rate = shortage / len(assembly_pool)
        else:
            shortage_rate = 1.0

        if inactive:
            pressure = sum(1 for v in inactive.values() if v) / len(inactive)
        else:
            pressure = 0.0

        with torch.no_grad():
            if op_x.size(0) > 0:
                op_h, _ = self.encoder(graph_data, precedence_info, op_id_to_idx, device)
                gnn_emb  = op_h.mean(dim=0)
            else:
                gnn_emb = torch.zeros(self.hidden_dim, device=device)

        scalar = torch.tensor(
            [urgency_score, shortage_rate, pressure],
            dtype=torch.float32, device=device,
        )
        return torch.cat([gnn_emb.detach(), scalar])

    def compute_U_p_from_embeddings(
        self, op_h: torch.Tensor, graph_data, obs: dict
    ) -> torch.Tensor:
        """사전 계산된 op_h로 U_p 산출 — 인코더 재호출 없음."""
        device        = op_h.device
        op_x          = graph_data['op'].x.to(device)
        assembly_pool = obs.get("assembly_pool", {})
        inactive      = obs.get("inactive_final_jobs", {})

        urgency_score = 0.0
        if op_x.size(0) > 0:
            is_asm   = op_x[:, OpFeat.IS_ASSEMBLY]
            due_norm = op_x[:, OpFeat.DUE_DATE_NORM]
            wt_norm  = op_x[:, OpFeat.WEIGHT_NORM]
            mask     = is_asm > 0.5
            if mask.any():
                urgency_score = float(
                    (wt_norm[mask] * (1.0 - due_norm[mask])).mean().item()
                )

        shortage_rate = (
            sum(1 for tp in assembly_pool.values() if len(tp) < 2) / len(assembly_pool)
            if assembly_pool else 1.0
        )
        pressure = (
            sum(1 for v in inactive.values() if v) / len(inactive)
            if inactive else 0.0
        )

        gnn_emb = op_h.mean(dim=0) if op_h.size(0) > 0 else torch.zeros(self.hidden_dim, device=device)
        scalar  = torch.tensor([urgency_score, shortage_rate, pressure],
                               dtype=torch.float32, device=device)
        return torch.cat([gnn_emb.detach(), scalar])

    def forward_from_embeddings(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        actions: list,
        op_id_to_idx: dict,
        job_op_map: dict,
        c_p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """사전 계산된 임베딩으로 Assembly Q-값 산출 — MLP 한 번 호출."""
        device   = op_h.device
        asm_acts = [a for a in actions if _is_assembly(a)]
        if not asm_acts:
            return torch.zeros(1, device=device)

        graph_emb = torch.cat([op_h.mean(dim=0), machine_h.mean(dim=0)])
        zero_h    = torch.zeros(self.hidden_dim, device=device)
        if c_p is None:
            c_p = torch.zeros(self.cross_dim, device=device)

        valid_inps, valid_locals = [], []
        for local_i, (comp_job_ids, mid) in enumerate(asm_acts):
            if mid >= machine_h.size(0):
                continue
            comp_embs, valid = [], True
            for job_id in comp_job_ids:
                idx = self._get_job_last_op_idx(job_id, op_id_to_idx, job_op_map)
                if idx is None:
                    valid = False; break
                comp_embs.append(op_h[idx] if idx < op_h.size(0) else zero_h)
            if not valid:
                continue
            comp_pool = torch.stack(comp_embs).mean(dim=0)
            inp = torch.cat([comp_pool, machine_h[mid], graph_emb, c_p])
            valid_inps.append(inp)
            valid_locals.append(local_i)

        if not valid_inps:
            return torch.full((len(asm_acts),), -1e9, device=device)

        q_computed = self.q_asm_mlp(torch.stack(valid_inps)).squeeze(-1)

        if len(valid_locals) == len(asm_acts):
            return q_computed

        valid_set = set(valid_locals)
        q_full, ptr = [], 0
        for local_i in range(len(asm_acts)):
            if local_i in valid_set:
                q_full.append(q_computed[ptr]); ptr += 1
            else:
                q_full.append(torch.tensor(-1e9, device=device))
        return torch.stack(q_full)


# ──────────────────────────────────────────────────────────
# DualDQNAgent
# ──────────────────────────────────────────────────────────

class DualDQNAgent:
    """
    Regular Q-network + Assembly Q-network 듀얼 에이전트.

    액션 선택:
      두 네트워크 Q-value를 통합해 전체 action space에서 argmax.
      · Regular 네트워크 입력 = obs + U_p (Assembly 출력 detach)
      · Assembly 네트워크 입력 = obs + C_p (Regular 출력 detach)

    업데이트 (엇갈린 스케줄):
      · update_regular(): trajectory 내 regular action 스텝만 학습
      · update_assembly(): trajectory 내 assembly action 스텝만 학습
      · 업데이트 시 상대 네트워크 출력은 항상 detach() 처리
      · 4개 네트워크: reg_online, reg_target, asm_online, asm_target
    """

    def __init__(
        self,
        reg_net: RegularQNetwork,
        asm_net: AssemblyQNetwork,
        lr: float = 2e-4,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.device        = device
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_grad_norm = max_grad_norm

        self.reg_online = reg_net.to(device)
        self.reg_target = copy.deepcopy(reg_net).to(device)
        for p in self.reg_target.parameters():
            p.requires_grad = False

        self.asm_online = asm_net.to(device)
        self.asm_target = copy.deepcopy(asm_net).to(device)
        for p in self.asm_target.parameters():
            p.requires_grad = False

        self.reg_optimizer = torch.optim.Adam(self.reg_online.parameters(), lr=lr)
        self.asm_optimizer = torch.optim.Adam(self.asm_online.parameters(), lr=lr)

    def _parse_obs(self, obs: dict):
        prec         = obs["precedence_info"]
        op_id_to_idx = {oid: i for i, oid in enumerate(sorted(prec["prev_map"].keys()))}
        return (obs["graph"], obs["actions"], prec,
                op_id_to_idx, obs.get("job_op_map", {}))

    def _compute_c_p(self, obs: dict, use_target: bool = False) -> torch.Tensor:
        """C_p 만 계산 (Regular → Assembly). use_target=True 이면 reg_target 사용."""
        graph, _, prec, op_id_to_idx, _ = self._parse_obs(obs)
        reg_net = self.reg_target if use_target else self.reg_online
        with torch.no_grad():
            c_p = reg_net.compute_C_p(graph, prec, op_id_to_idx, obs)
        return c_p.detach()

    def _compute_u_p(self, obs: dict, use_target: bool = False) -> torch.Tensor:
        """U_p 만 계산 (Assembly → Regular). use_target=True 이면 asm_target 사용."""
        graph, _, prec, op_id_to_idx, _ = self._parse_obs(obs)
        asm_net = self.asm_target if use_target else self.asm_online
        with torch.no_grad():
            u_p = asm_net.compute_U_p(graph, prec, op_id_to_idx, obs)
        return u_p.detach()

    def select_action(self, obs: dict) -> int:
        """ε-greedy: ε 확률 랜덤, 나머지는 두 네트워크 Q값 통합 argmax.

        각 인코더를 1회만 호출해 교차 신호(C_p/U_p)와 Q-값을 모두 계산.
        """
        actions = obs["actions"]
        if not actions:
            return 0
        if random.random() < self.epsilon:
            return random.randint(0, len(actions) - 1)

        graph, _, prec, op_id_to_idx, job_op_map = self._parse_obs(obs)

        with torch.no_grad():
            # reg_online 인코더 1회 → C_p + Q_reg 공용
            op_h_reg, machine_h_reg = self.reg_online.encoder(graph, prec, op_id_to_idx, self.device)
            # asm_online 인코더 1회 → U_p + Q_asm 공용
            op_h_asm, machine_h_asm = self.asm_online.encoder(graph, prec, op_id_to_idx, self.device)

            c_p = self.reg_online.compute_C_p_from_embeddings(op_h_reg, obs)
            u_p = self.asm_online.compute_U_p_from_embeddings(op_h_asm, graph, obs)

            reg_idx = [i for i, a in enumerate(actions) if not _is_assembly(a)]
            asm_idx = [i for i, a in enumerate(actions) if     _is_assembly(a)]
            q_all   = torch.full((len(actions),), -1e9, device=self.device)

            if reg_idx:
                q_reg = self.reg_online.forward_from_embeddings(
                    op_h_reg, machine_h_reg, actions, graph, op_id_to_idx, u_p=u_p
                )
                for local_i, global_i in enumerate(reg_idx):
                    if local_i < q_reg.size(0):
                        q_all[global_i] = q_reg[local_i]

            if asm_idx:
                q_asm = self.asm_online.forward_from_embeddings(
                    op_h_asm, machine_h_asm, actions, op_id_to_idx, job_op_map, c_p=c_p
                )
                for local_i, global_i in enumerate(asm_idx):
                    if local_i < q_asm.size(0):
                        q_all[global_i] = q_asm[local_i]

        return int(q_all.argmax().item())

    def update_regular(self, trajectory: list) -> dict:
        """
        Regular action 스텝만 TD loss 계산 → reg_online 업데이트.
        U_p 는 asm_online/asm_target 에서 detach 해 입력.
        """
        reg_steps = [
            (obs, act, r, nobs, done)
            for obs, act, r, nobs, done in trajectory
            if act < len(obs["actions"]) and not _is_assembly(obs["actions"][act])
        ]
        if not reg_steps:
            return {"loss_reg": 0.0}

        self.reg_optimizer.zero_grad()
        loss_sum = 0.0

        for obs, action_idx, reward, next_obs, done in reg_steps:
            graph, actions, prec, op_id_to_idx, _ = self._parse_obs(obs)
            u_p = self._compute_u_p(obs, use_target=False)

            q_vals = self.reg_online(graph, actions, prec, op_id_to_idx, u_p=u_p)

            reg_global = [i for i, a in enumerate(actions) if not _is_assembly(a)]
            if action_idx not in reg_global:
                continue
            local_idx = min(reg_global.index(action_idx), q_vals.size(0) - 1)
            q_val = q_vals[local_idx]

            if done or not next_obs["actions"]:
                target = torch.tensor(float(reward), device=self.device)
            else:
                ng, na, np_, noi, njm = self._parse_obs(next_obs)
                nu_p = self._compute_u_p(next_obs, use_target=True)
                nc_p = self._compute_c_p(next_obs, use_target=True)
                next_reg = [a for a in na if not _is_assembly(a)]
                next_asm = [a for a in na if     _is_assembly(a)]
                with torch.no_grad():
                    nq_reg = (self.reg_target(ng, na, np_, noi, u_p=nu_p).max()
                              if next_reg else torch.tensor(-1e9, device=self.device))
                    nq_asm = (self.asm_target(ng, na, np_, noi, njm, c_p=nc_p).max()
                              if next_asm else torch.tensor(-1e9, device=self.device))
                next_val = torch.max(nq_reg, nq_asm)
                target = torch.tensor(float(reward), device=self.device) + self.gamma * next_val

            loss_sum += F.mse_loss(q_val, target.detach())

        if loss_sum == 0.0:
            return {"loss_reg": 0.0}

        loss = loss_sum / len(reg_steps)
        loss.backward()
        nn.utils.clip_grad_norm_(self.reg_online.parameters(), self.max_grad_norm)
        self.reg_optimizer.step()
        return {"loss_reg": loss.item()}

    def update_regular_batch(self, batch: list) -> dict:
        """reg_buffer 샘플 배치로 Regular 네트워크 업데이트.

        Stage 1: 4회 batched GNN 인코딩 (개별 ~192회 → 4회)
        Stage 2: forward_from_embeddings로 MLP 벡터화
        """
        reg_steps = [
            (obs, act, r, nobs, done)
            for obs, act, r, nobs, done in batch
            if act < len(obs["actions"]) and not _is_assembly(obs["actions"][act])
        ]
        if not reg_steps:
            return {"loss_reg": 0.0}

        parsed_obs  = [self._parse_obs(obs)  for obs, _, _, _, _    in reg_steps]
        parsed_nobs = [self._parse_obs(nobs) for _, _, _, nobs, _ in reg_steps]
        obs_info    = [(g, p, o) for g, _, p, o, _ in parsed_obs]
        nobs_info   = [(g, p, o) for g, _, p, o, _ in parsed_nobs]

        # 4회 batched 인코더 호출
        self.reg_optimizer.zero_grad()
        with torch.no_grad():
            asm_online_obs  = self.asm_online.encoder.forward_batch(obs_info,  self.device)
            reg_target_nobs = self.reg_target.encoder.forward_batch(nobs_info, self.device)
            asm_target_nobs = self.asm_target.encoder.forward_batch(nobs_info, self.device)
        reg_online_obs = self.reg_online.encoder.forward_batch(obs_info, self.device)

        losses = []
        for i, (obs, action_idx, reward, next_obs, done) in enumerate(reg_steps):
            graph, actions, prec, op_id_to_idx, _ = parsed_obs[i]
            op_h_asm, _             = asm_online_obs[i]
            op_h_reg, machine_h_reg = reg_online_obs[i]

            with torch.no_grad():
                u_p = self.asm_online.compute_U_p_from_embeddings(op_h_asm, graph, obs)

            q_vals = self.reg_online.forward_from_embeddings(
                op_h_reg, machine_h_reg, actions, graph, op_id_to_idx, u_p=u_p
            )

            reg_global = [j for j, a in enumerate(actions) if not _is_assembly(a)]
            if action_idx not in reg_global:
                continue
            local_idx = min(reg_global.index(action_idx), q_vals.size(0) - 1)
            q_val = q_vals[local_idx]

            if done or not next_obs["actions"]:
                target = torch.tensor(float(reward), device=self.device)
            else:
                ng, na, np_, noi, njm = parsed_nobs[i]
                op_h_reg_t, machine_h_reg_t = reg_target_nobs[i]
                op_h_asm_t, machine_h_asm_t = asm_target_nobs[i]

                next_reg = [a for a in na if not _is_assembly(a)]
                next_asm = [a for a in na if     _is_assembly(a)]

                with torch.no_grad():
                    nu_p = self.asm_target.compute_U_p_from_embeddings(op_h_asm_t, ng, next_obs)
                    nc_p = self.reg_target.compute_C_p_from_embeddings(op_h_reg_t, next_obs)
                    nq_reg = (self.reg_target.forward_from_embeddings(
                                  op_h_reg_t, machine_h_reg_t, na, ng, noi, u_p=nu_p
                              ).max()
                              if next_reg else torch.tensor(-1e9, device=self.device))
                    nq_asm = (self.asm_target.forward_from_embeddings(
                                  op_h_asm_t, machine_h_asm_t, na, noi, njm, c_p=nc_p
                              ).max()
                              if next_asm else torch.tensor(-1e9, device=self.device))
                    next_val = torch.max(nq_reg, nq_asm)
                target = torch.tensor(float(reward), device=self.device) + self.gamma * next_val

            losses.append(F.smooth_l1_loss(q_val, target.detach()))

        if not losses:
            return {"loss_reg": 0.0}

        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.reg_online.parameters(), self.max_grad_norm)
        self.reg_optimizer.step()
        return {"loss_reg": loss.item()}

    def update_assembly_batch(self, batch: list) -> dict:
        """asm_buffer 샘플 배치로 Assembly 네트워크 업데이트.

        Stage 1: 4회 batched GNN 인코딩
        Stage 2: forward_from_embeddings로 MLP 벡터화
        """
        asm_steps = [
            (obs, act, r, nobs, done)
            for obs, act, r, nobs, done in batch
            if act < len(obs["actions"]) and _is_assembly(obs["actions"][act])
        ]
        if not asm_steps:
            return {"loss_asm": 0.0}

        parsed_obs  = [self._parse_obs(obs)  for obs, _, _, _, _    in asm_steps]
        parsed_nobs = [self._parse_obs(nobs) for _, _, _, nobs, _ in asm_steps]
        obs_info    = [(g, p, o) for g, _, p, o, _ in parsed_obs]
        nobs_info   = [(g, p, o) for g, _, p, o, _ in parsed_nobs]

        # 4회 batched 인코더 호출
        self.asm_optimizer.zero_grad()
        with torch.no_grad():
            reg_online_obs  = self.reg_online.encoder.forward_batch(obs_info,  self.device)
            reg_target_nobs = self.reg_target.encoder.forward_batch(nobs_info, self.device)
            asm_target_nobs = self.asm_target.encoder.forward_batch(nobs_info, self.device)
        asm_online_obs = self.asm_online.encoder.forward_batch(obs_info, self.device)

        losses = []
        for i, (obs, action_idx, reward, next_obs, done) in enumerate(asm_steps):
            graph, actions, prec, op_id_to_idx, job_op_map = parsed_obs[i]
            op_h_reg, _             = reg_online_obs[i]
            op_h_asm, machine_h_asm = asm_online_obs[i]

            with torch.no_grad():
                c_p = self.reg_online.compute_C_p_from_embeddings(op_h_reg, obs)

            q_vals = self.asm_online.forward_from_embeddings(
                op_h_asm, machine_h_asm, actions, op_id_to_idx, job_op_map, c_p=c_p
            )

            asm_global = [j for j, a in enumerate(actions) if _is_assembly(a)]
            if action_idx not in asm_global:
                continue
            local_idx = min(asm_global.index(action_idx), q_vals.size(0) - 1)
            q_val = q_vals[local_idx]

            if done or not next_obs["actions"]:
                target = torch.tensor(float(reward), device=self.device)
            else:
                ng, na, np_, noi, njm = parsed_nobs[i]
                op_h_reg_t, machine_h_reg_t = reg_target_nobs[i]
                op_h_asm_t, machine_h_asm_t = asm_target_nobs[i]

                next_reg = [a for a in na if not _is_assembly(a)]
                next_asm = [a for a in na if     _is_assembly(a)]

                with torch.no_grad():
                    nc_p = self.reg_target.compute_C_p_from_embeddings(op_h_reg_t, next_obs)
                    nu_p = self.asm_target.compute_U_p_from_embeddings(op_h_asm_t, ng, next_obs)
                    nq_reg = (self.reg_target.forward_from_embeddings(
                                  op_h_reg_t, machine_h_reg_t, na, ng, noi, u_p=nu_p
                              ).max()
                              if next_reg else torch.tensor(-1e9, device=self.device))
                    nq_asm = (self.asm_target.forward_from_embeddings(
                                  op_h_asm_t, machine_h_asm_t, na, noi, njm, c_p=nc_p
                              ).max()
                              if next_asm else torch.tensor(-1e9, device=self.device))
                    next_val = torch.max(nq_reg, nq_asm)
                target = torch.tensor(float(reward), device=self.device) + self.gamma * next_val

            losses.append(F.smooth_l1_loss(q_val, target.detach()))

        if not losses:
            return {"loss_asm": 0.0}

        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.asm_online.parameters(), self.max_grad_norm)
        self.asm_optimizer.step()
        return {"loss_asm": loss.item()}

    def update_assembly(self, trajectory: list) -> dict:
        """
        Assembly action 스텝만 TD loss 계산 → asm_online 업데이트.
        C_p 는 reg_online/reg_target 에서 detach 해 입력.
        """
        asm_steps = [
            (obs, act, r, nobs, done)
            for obs, act, r, nobs, done in trajectory
            if act < len(obs["actions"]) and _is_assembly(obs["actions"][act])
        ]
        if not asm_steps:
            return {"loss_asm": 0.0}

        self.asm_optimizer.zero_grad()
        loss_sum = 0.0

        for obs, action_idx, reward, next_obs, done in asm_steps:
            graph, actions, prec, op_id_to_idx, job_op_map = self._parse_obs(obs)
            c_p = self._compute_c_p(obs, use_target=False)

            q_vals = self.asm_online(graph, actions, prec, op_id_to_idx, job_op_map, c_p=c_p)

            asm_global = [i for i, a in enumerate(actions) if _is_assembly(a)]
            if action_idx not in asm_global:
                continue
            local_idx = min(asm_global.index(action_idx), q_vals.size(0) - 1)
            q_val = q_vals[local_idx]

            if done or not next_obs["actions"]:
                target = torch.tensor(float(reward), device=self.device)
            else:
                ng, na, np_, noi, njm = self._parse_obs(next_obs)
                nc_p = self._compute_c_p(next_obs, use_target=True)
                nu_p = self._compute_u_p(next_obs, use_target=True)
                next_reg = [a for a in na if not _is_assembly(a)]
                next_asm = [a for a in na if     _is_assembly(a)]
                with torch.no_grad():
                    nq_reg = (self.reg_target(ng, na, np_, noi, u_p=nu_p).max()
                              if next_reg else torch.tensor(-1e9, device=self.device))
                    nq_asm = (self.asm_target(ng, na, np_, noi, njm, c_p=nc_p).max()
                              if next_asm else torch.tensor(-1e9, device=self.device))
                next_val = torch.max(nq_reg, nq_asm)
                target = torch.tensor(float(reward), device=self.device) + self.gamma * next_val

            loss_sum += F.mse_loss(q_val, target.detach())

        if loss_sum == 0.0:
            return {"loss_asm": 0.0}

        loss = loss_sum / len(asm_steps)
        loss.backward()
        nn.utils.clip_grad_norm_(self.asm_online.parameters(), self.max_grad_norm)
        self.asm_optimizer.step()
        return {"loss_asm": loss.item()}

    def update_regular_target(self, tau: float = 0.005):
        for p_t, p_o in zip(self.reg_target.parameters(), self.reg_online.parameters()):
            p_t.data.copy_(tau * p_o.data + (1 - tau) * p_t.data)

    def update_assembly_target(self, tau: float = 0.005):
        for p_t, p_o in zip(self.asm_target.parameters(), self.asm_online.parameters()):
            p_t.data.copy_(tau * p_o.data + (1 - tau) * p_t.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
