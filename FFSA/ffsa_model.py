"""
HGNN Q-Network + DQN Agent
===========================
3-Stage Embedding (PPT Slide 12 구조 유지)
  Stage 1: Machine Embedding (GAT 기반)
  Stage 2: Operation Embedding (MLP)
  Q-Head:  MLP_Q(op_emb ‖ machine_emb ‖ graph_emb ‖ edge_feat) → Q(s,a)
           조립 action: MLP_Q_asm(mean(comp_embs) ‖ machine_emb ‖ graph_emb) → Q(s,a)

학습: Window 기반 Best-Trajectory DQN
  - window_size 에피소드 수집 → WT 최소 에피소드 trajectory 선택
  - 선택된 trajectory로 TD loss → online Q-network 업데이트
  - target_update_cycles 업데이트마다 target ← online 가중치 복사
  탐색: ε-greedy (ε 지수 감소)
"""

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

RegularAction = Tuple[int, int]
AssemblyAction = Tuple[Tuple[int, ...], int]
Action = Union[RegularAction, AssemblyAction]


# ──────────────────────────────────────────────────────────
# HGNN Q-Network
# ──────────────────────────────────────────────────────────

class HGNNQNetwork(nn.Module):
    """
    GNN 인코더 + Q-value MLP 헤드.
    각 후보 action에 대해 Q(s, a) 스칼라를 출력.
    """

    def __init__(
        self,
        op_feat_dim: int = 10,
        machine_feat_dim: int = 6,
        edge_feat_dim: int = 2,
        hidden_dim: int = 16,
        num_layers: int = 2,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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

        # Q-value Head — 일반 action
        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + edge_feat_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        # Q-value Head — 조립 action
        self.q_asm_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        graph_data,
        actions: List[Action],
        precedence_info: Dict,
        op_id_to_idx: Dict[int, int],
        job_op_map: Dict[int, List[int]],
    ) -> torch.Tensor:
        """Returns: q_values shape [num_actions]"""
        device = next(self.parameters()).device

        op_x      = graph_data['op'].x.to(device)
        machine_x = graph_data['machine'].x.to(device)

        if op_x.size(0) == 0 or not actions:
            return torch.zeros(max(len(actions), 1), device=device)

        op_h      = self.op_encoder(op_x)
        machine_h = self.machine_encoder(machine_x)

        cand_edge = graph_data['op', 'candidate', 'machine'].edge_index.to(device)
        cand_attr = graph_data['op', 'candidate', 'machine'].edge_attr.to(device)
        rev_edge  = graph_data['machine', 'candidate_rev', 'op'].edge_index.to(device)
        rev_attr  = graph_data['machine', 'candidate_rev', 'op'].edge_attr.to(device)

        for i in range(self.num_layers):
            if cand_edge.size(1) > 0:
                machine_h = F.elu(self.gat_op2m[i]((op_h, machine_h), cand_edge, edge_attr=cand_attr) + machine_h)
            if rev_edge.size(1) > 0:
                op_h = F.elu(self.gat_m2o[i]((machine_h, op_h), rev_edge, edge_attr=rev_attr) + op_h)

        prev_map         = precedence_info['prev_map']
        next_map         = precedence_info['next_map']
        candidate_machines = precedence_info['candidate_machines']
        num_ops          = op_h.size(0)
        zero_pad         = torch.zeros((1, self.hidden_dim), device=device)
        op_h_padded      = torch.cat([op_h, zero_pad], dim=0)

        sorted_op_ids = sorted(op_id_to_idx, key=op_id_to_idx.get)
        prev_idx = torch.tensor(
            [op_id_to_idx.get(prev_map.get(oid), num_ops) if prev_map.get(oid) is not None else num_ops
             for oid in sorted_op_ids], dtype=torch.long, device=device)
        next_idx = torch.tensor(
            [op_id_to_idx.get(next_map.get(oid), num_ops) if next_map.get(oid) is not None else num_ops
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

        graph_emb    = torch.cat([op_h.mean(dim=0), machine_h.mean(dim=0)])
        edge_feat_map = self._build_edge_feat_map(graph_data, device)

        q_vals = []
        for action in actions:
            if not isinstance(action[0], tuple):
                op_id, mid = action
                op_idx = op_id_to_idx.get(op_id)
                if op_idx is None or op_idx >= op_h.size(0) or mid >= machine_h.size(0):
                    q_vals.append(torch.tensor(-1e9, device=device))
                    continue
                ef  = edge_feat_map.get((op_idx, mid), torch.zeros(self.edge_feat_dim, device=device))
                inp = torch.cat([op_h[op_idx], machine_h[mid], graph_emb, ef])
                q_vals.append(self.q_mlp(inp.unsqueeze(0)).squeeze())
            else:
                comp_job_ids, mid = action
                if mid >= machine_h.size(0):
                    q_vals.append(torch.tensor(-1e9, device=device))
                    continue
                comp_embs, valid = [], True
                for job_id in comp_job_ids:
                    idx = self._get_job_last_op_idx(job_id, op_id_to_idx, job_op_map)
                    if idx is None:
                        valid = False
                        break
                    comp_embs.append(op_h[idx] if idx < op_h.size(0) else zero_pad.squeeze(0))
                if not valid:
                    q_vals.append(torch.tensor(-1e9, device=device))
                    continue
                comp_pool = torch.stack(comp_embs).mean(dim=0)
                inp = torch.cat([comp_pool, machine_h[mid], graph_emb])
                q_vals.append(self.q_asm_mlp(inp.unsqueeze(0)).squeeze())

        return torch.stack(q_vals)

    def _get_job_last_op_idx(self, job_id, op_id_to_idx, job_op_map):
        for op_id in reversed(job_op_map.get(job_id, [])):
            if op_id in op_id_to_idx:
                return op_id_to_idx[op_id]
        return None

    def _build_edge_feat_map(self, graph_data, device):
        edge_map   = {}
        edge_index = graph_data['op', 'candidate', 'machine'].edge_index
        edge_attr  = graph_data['op', 'candidate', 'machine'].edge_attr
        for i in range(edge_index.size(1)):
            edge_map[(edge_index[0, i].item(), edge_index[1, i].item())] = edge_attr[i].to(device)
        return edge_map


# ──────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────

class DQNAgent:
    """
    online Q-network + target Q-network.

    업데이트 방식:
      - window 내 WT 최소 에피소드의 trajectory로 update_from_trajectory() 호출
      - target_update_cycles 번 업데이트마다 target ← online 가중치 복사
    탐색: ε-greedy (매 에피소드 지수 감소)
    """

    def __init__(
        self,
        q_net: HGNNQNetwork,
        lr: float = 2e-4,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.online_net = q_net.to(device)
        self.target_net = copy.deepcopy(q_net).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer     = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_grad_norm = max_grad_norm
        self.device        = device

    def _obs_to_args(self, obs: dict):
        graph           = obs["graph"]
        actions         = obs["actions"]
        precedence_info = obs["precedence_info"]
        op_id_to_idx    = {op_id: idx for idx, op_id in
                           enumerate(sorted(precedence_info["prev_map"].keys()))}
        job_op_map      = obs.get("job_op_map", {})
        return graph, actions, precedence_info, op_id_to_idx, job_op_map

    def select_action(self, obs: dict) -> int:
        """ε-greedy: ε 확률로 랜덤, 나머지는 Q값 최대 액션"""
        import random
        actions = obs["actions"]
        if not actions:
            return 0
        if random.random() < self.epsilon:
            return random.randint(0, len(actions) - 1)
        with torch.no_grad():
            q_vals = self.online_net(*self._obs_to_args(obs))
        return int(q_vals.argmax().item())

    def update_from_trajectory(self, trajectory: list) -> dict:
        """window 내 최고 에피소드 trajectory로 TD loss → online network 업데이트"""
        if not trajectory:
            return {}

        self.optimizer.zero_grad()
        loss_sum = 0.0

        for obs, action_idx, reward, next_obs, done in trajectory:
            q_vals     = self.online_net(*self._obs_to_args(obs))
            action_idx = min(action_idx, q_vals.size(0) - 1)
            q_val      = q_vals[action_idx]

            if done or not next_obs["actions"]:
                target = torch.tensor(float(reward), device=self.device)
            else:
                with torch.no_grad():
                    next_q = self.target_net(*self._obs_to_args(next_obs))
                target = torch.tensor(float(reward), device=self.device) + self.gamma * next_q.max()

            loss_sum += F.mse_loss(q_val, target)

        loss = loss_sum / len(trajectory)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_target(self):
        """online 가중치를 target에 복사"""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
