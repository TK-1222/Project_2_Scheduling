"""
HGNN 3-Stage Policy + PPO Agent
================================
PPT Slide 12: 3-Stage Embedding
  Stage 1: Machine Embedding (GAT 기반)
  Stage 2: Operation Embedding (MLP)
  Policy:  MLPπ(μij ‖ νk ‖ ht ‖ λijk) → softmax with mask
           조립 action: MLPπ_asm(μA ‖ μB ‖ νk ‖ ht) → softmax
  Value:   MLPv(ht) → scalar

PPT Slide 13: PPO 하이퍼파라미터
  L=2, d=16, hidden=128, γ=1.0, λ_GAE=0.95, ε=0.2,
  entropy=0.01, vf=0.5, lr=2e-4, epochs=3~5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATConv
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

RegularAction = Tuple[int, int]
AssemblyAction = Tuple[int, int, int]
Action = Union[RegularAction, AssemblyAction]


# ──────────────────────────────────────────────────────────
# 3-Stage HGNN Policy Network
# ──────────────────────────────────────────────────────────

class HGNNPolicy(nn.Module):
    """
    PPT Slide 12: 3-Stage Embedding

    Stage 1: GAT — (op ↔ machine) candidate edges로 메시지 전달
    Stage 2: Operation MLP — prev/next/machine_mean/self 결합
    Policy Head (일반): MLPπ(op_emb ‖ machine_emb ‖ graph_emb ‖ edge_feat)
    Policy Head (조립): MLPπ_asm(comp_A_emb ‖ comp_B_emb ‖ machine_emb ‖ graph_emb)
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

        # Input projections
        self.op_encoder = nn.Linear(op_feat_dim, hidden_dim)
        self.machine_encoder = nn.Linear(machine_feat_dim, hidden_dim)

        # Stage 1: GAT (op ↔ machine)
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

        # Stage 2: Operation Embedding MLP
        self.theta1 = nn.Linear(hidden_dim, hidden_dim)
        self.theta2 = nn.Linear(hidden_dim, hidden_dim)
        self.theta3 = nn.Linear(hidden_dim, hidden_dim)
        self.theta4 = nn.Linear(hidden_dim, hidden_dim)
        self.theta0 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ELU(),
        )

        # Policy Head — 일반 action: (op ‖ machine ‖ graph ‖ edge) = 4d + edge_dim
        self.policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + edge_feat_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        # Policy Head — 조립 action: (comp_A ‖ comp_B ‖ machine ‖ graph) = 4d
        self.policy_asm_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        # Value Head: MLPv(graph_emb) = MLPv(2d)
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        graph_data,
        actions: List[Action],
        action_mask: torch.Tensor,
        precedence_info: Dict,
        op_id_to_idx: Dict[int, int],
    ):
        """
        Args:
            op_id_to_idx: 활성 op_id → graph 내 노드 인덱스 매핑
        Returns:
            probs: 각 action의 확률 (masked softmax)
            value: state value scalar
        """
        device = next(self.parameters()).device

        op_x = graph_data['op'].x.to(device)
        machine_x = graph_data['machine'].x.to(device)

        if op_x.size(0) == 0:
            dummy_prob = torch.ones(max(len(actions), 1), device=device) / max(len(actions), 1)
            dummy_val = torch.tensor(0.0, device=device)
            return dummy_prob, dummy_val

        op_h = self.op_encoder(op_x)
        machine_h = self.machine_encoder(machine_x)

        cand_edge = graph_data['op', 'candidate', 'machine'].edge_index.to(device)
        cand_attr = graph_data['op', 'candidate', 'machine'].edge_attr.to(device)
        rev_edge = graph_data['machine', 'candidate_rev', 'op'].edge_index.to(device)
        rev_attr = graph_data['machine', 'candidate_rev', 'op'].edge_attr.to(device)

        # Stage 1: GAT
        for layer_idx in range(self.num_layers):
            if cand_edge.size(1) > 0:
                m_new = self.gat_op2m[layer_idx](
                    (op_h, machine_h), cand_edge, edge_attr=cand_attr
                )
                machine_h = F.elu(m_new + machine_h)
            if rev_edge.size(1) > 0:
                o_new = self.gat_m2o[layer_idx](
                    (machine_h, op_h), rev_edge, edge_attr=rev_attr
                )
                op_h = F.elu(o_new + op_h)

        # Stage 2: Operation Embedding
        prev_map = precedence_info['prev_map']
        next_map = precedence_info['next_map']
        candidate_machines = precedence_info['candidate_machines']
        num_ops = op_h.size(0)

        zero_pad = torch.zeros((1, self.hidden_dim), device=device)
        op_h_padded = torch.cat([op_h, zero_pad], dim=0)

        prev_idx = torch.tensor(
            [op_id_to_idx.get(prev_map.get(op_id), num_ops) if prev_map.get(op_id) is not None else num_ops
             for op_id in sorted(op_id_to_idx, key=op_id_to_idx.get)],
            dtype=torch.long, device=device
        )
        next_idx = torch.tensor(
            [op_id_to_idx.get(next_map.get(op_id), num_ops) if next_map.get(op_id) is not None else num_ops
             for op_id in sorted(op_id_to_idx, key=op_id_to_idx.get)],
            dtype=torch.long, device=device
        )

        prev_h = op_h_padded[prev_idx]
        next_h = op_h_padded[next_idx]

        machine_mean = torch.zeros((num_ops, self.hidden_dim), device=device)
        for op_id, idx in op_id_to_idx.items():
            cand_m = candidate_machines.get(op_id, [])
            if cand_m:
                valid_m = [m for m in cand_m if m < machine_h.size(0)]
                if valid_m:
                    machine_mean[idx] = machine_h[valid_m].mean(dim=0)

        op_h = self.theta0(torch.cat([
            F.elu(self.theta1(prev_h)),
            F.elu(self.theta2(next_h)),
            F.elu(self.theta3(machine_mean)),
            F.elu(self.theta4(op_h)),
        ], dim=-1))

        # Graph pooling
        graph_emb = torch.cat([op_h.mean(dim=0), machine_h.mean(dim=0)])

        if not actions:
            return torch.tensor([1.0], device=device), self.value_mlp(graph_emb).squeeze()

        edge_feat_map = self._build_edge_feat_map(graph_data, device)

        # 각 action에 대해 logit 계산
        logits = []
        for action in actions:
            if len(action) == 2:
                # 일반 action: (op_id, machine_id)
                op_id, mid = action
                op_idx = op_id_to_idx.get(op_id)
                if op_idx is None or op_idx >= op_h.size(0) or mid >= machine_h.size(0):
                    logits.append(torch.tensor(-1e9, device=device))
                    continue
                sel_op = op_h[op_idx]
                sel_m = machine_h[mid]
                ef = edge_feat_map.get((op_idx, mid),
                                       torch.zeros(self.edge_feat_dim, device=device))
                inp = torch.cat([sel_op, sel_m, graph_emb, ef])
                logits.append(self.policy_mlp(inp.unsqueeze(0)).squeeze())
            else:
                # 조립 action: (comp_A_job_id, comp_B_job_id, machine_id)
                comp_a_job, comp_b_job, mid = action
                # component job의 마지막 active op 임베딩 사용
                a_idx = self._get_job_last_op_idx(comp_a_job, op_id_to_idx)
                b_idx = self._get_job_last_op_idx(comp_b_job, op_id_to_idx)
                if a_idx is None or b_idx is None or mid >= machine_h.size(0):
                    logits.append(torch.tensor(-1e9, device=device))
                    continue
                sel_a = op_h[a_idx] if a_idx < op_h.size(0) else zero_pad.squeeze(0)
                sel_b = op_h[b_idx] if b_idx < op_h.size(0) else zero_pad.squeeze(0)
                sel_m = machine_h[mid]
                inp = torch.cat([sel_a, sel_b, sel_m, graph_emb])
                logits.append(self.policy_asm_mlp(inp.unsqueeze(0)).squeeze())

        logits = torch.stack(logits)
        mask = action_mask.to(device).bool()
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits, dim=0)
        value = self.value_mlp(graph_emb).squeeze()

        return probs, value

    def _get_job_last_op_idx(
        self, job_id: int, op_id_to_idx: Dict[int, int]
    ) -> Optional[int]:
        """job의 op 중 op_id_to_idx에 있는 마지막 op 인덱스 반환"""
        # op_id_to_idx 키 중 해당 job의 op를 찾는 건 환경 정보 없이 불가하므로
        # job_id와 매핑을 외부에서 주입받는 방식 대신
        # 가장 큰 op_id를 사용 (같은 job의 op들 중 마지막)
        candidates = [idx for oid, idx in op_id_to_idx.items()]
        # 단순히 첫 번째 매핑 반환 (호출부에서 job_op_map을 전달하는 방식으로 개선 가능)
        return candidates[0] if candidates else None

    def _build_edge_feat_map(self, graph_data, device) -> Dict[Tuple[int, int], torch.Tensor]:
        edge_map = {}
        edge_index = graph_data['op', 'candidate', 'machine'].edge_index
        edge_attr = graph_data['op', 'candidate', 'machine'].edge_attr
        if edge_index.size(1) > 0:
            for idx in range(edge_index.size(1)):
                oid = edge_index[0, idx].item()
                mid = edge_index[1, idx].item()
                edge_map[(oid, mid)] = edge_attr[idx].to(device)
        return edge_map


# ──────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, obs, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 1.0, gae_lambda: float = 0.95
    ):
        advantages = []
        gae = 0.0
        values = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(self.values, dtype=torch.float32)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ──────────────────────────────────────────────────────────
# PPO Agent
# ──────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(
        self,
        policy: HGNNPolicy,
        lr: float = 2e-4,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        update_epochs: int = 4,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.buffer = RolloutBuffer()

    def _obs_to_forward_args(self, obs: dict):
        graph = obs["graph"]
        actions = obs["actions"]
        mask = torch.tensor(obs["action_mask"], dtype=torch.float32)
        precedence_info = obs["precedence_info"]
        # op_id → graph 노드 인덱스 매핑 구성
        op_id_to_idx = self._build_op_id_to_idx(obs)
        return graph, actions, mask, precedence_info, op_id_to_idx

    def _build_op_id_to_idx(self, obs: dict) -> Dict[int, int]:
        """precedence_info의 prev_map 키(활성 op_id)를 순서대로 인덱싱"""
        op_ids = sorted(obs["precedence_info"]["prev_map"].keys())
        return {op_id: idx for idx, op_id in enumerate(op_ids)}

    def select_action(self, obs: dict) -> Tuple[int, float, float]:
        actions = obs["actions"]
        if not actions:
            return 0, 0.0, 0.0

        args = self._obs_to_forward_args(obs)
        with torch.no_grad():
            probs, value = self.policy(*args)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def store(self, obs, action, log_prob, reward, value, done):
        self.buffer.store(obs, action, log_prob, reward, value, done)

    def update(self) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {}

        last_obs = self.buffer.observations[-1]
        with torch.no_grad():
            _, last_val = self.policy(*self._obs_to_forward_args(last_obs))
            last_value = last_val.item() if not self.buffer.dones[-1] else 0.0

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32)

        total_loss_sum = policy_loss_sum = value_loss_sum = entropy_sum = 0.0

        for _ in range(self.update_epochs):
            self.optimizer.zero_grad()
            batch_size = max(len(self.buffer), 1)

            for t in range(len(self.buffer)):
                obs = self.buffer.observations[t]
                action = self.buffer.actions[t]

                if not obs["actions"]:
                    continue

                args = self._obs_to_forward_args(obs)
                probs, value = self.policy(*args)

                dist = Categorical(probs)
                new_log_prob = dist.log_prob(torch.tensor(action))
                entropy = dist.entropy()

                ratio = torch.exp(new_log_prob - old_log_probs[t])
                surr1 = ratio * advantages[t]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[t]
                policy_loss = -torch.min(surr1, surr2)
                value_loss = F.mse_loss(value, returns[t])
                loss = (policy_loss + self.value_coeff * value_loss
                        - self.entropy_coeff * entropy) / batch_size
                loss.backward()

                total_loss_sum += loss.item() * batch_size
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.item()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        n = max(len(self.buffer) * self.update_epochs, 1)
        self.buffer.clear()
        return {
            "loss": total_loss_sum / n,
            "policy_loss": policy_loss_sum / n,
            "value_loss": value_loss_sum / n,
            "entropy": entropy_sum / n,
        }
