"""
HGNN 3-Stage Policy + PPO Agent
================================
PPT Slide 12: 3-Stage Embedding
  Stage 1: Machine Embedding (GAT 기반) — fjsp-drl 동일
  Stage 2: Assembly Dependency 전파   — 본 연구 추가
  Stage 3: Operation Embedding (MLP)  — fjsp-drl 동일
  Policy:  MLPπ(μij ‖ νk ‖ ht ‖ λijk) → softmax with mask
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
from typing import Dict, List, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────
# 3-Stage HGNN Policy Network
# ──────────────────────────────────────────────────────────

class HGNNPolicy(nn.Module):
    """
    PPT Slide 12: 3-Stage Embedding

    Stage 1: GAT — (op ↔ machine) candidate edges로 메시지 전달
    Stage 2: Assembly MLP — component embeddings → 조립 node에 집약
    Stage 3: Operation MLP — prev/next/machine_mean/self 결합
    """

    def __init__(
        self,
        op_feat_dim: int = 14,
        machine_feat_dim: int = 7,
        edge_feat_dim: int = 2,
        hidden_dim: int = 16,        # d=16 (PPT)
        num_layers: int = 2,         # L=2 (PPT)
        mlp_hidden: int = 128,       # MLP hidden=128 (PPT)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_feat_dim = edge_feat_dim

        # Input projections
        self.op_encoder = nn.Linear(op_feat_dim, hidden_dim)
        self.machine_encoder = nn.Linear(machine_feat_dim, hidden_dim)

        # ── Stage 1: Machine Embedding (GAT) ──
        # op → machine, machine → op 양방향 GAT
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

        # ── Stage 2: Assembly Dependency 전파 ──
        # μ''_asm = MLPasm(μ'_asm ‖ AGG({μ'_comp}))
        self.mlp_asm = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Stage 3: Operation Embedding (MLP) ──
        # μ''' = MLP_θ0(ELU[θ1(prev) ‖ θ2(next) ‖ θ3(machine_mean) ‖ θ4(self)])
        self.theta1 = nn.Linear(hidden_dim, hidden_dim)
        self.theta2 = nn.Linear(hidden_dim, hidden_dim)
        self.theta3 = nn.Linear(hidden_dim, hidden_dim)
        self.theta4 = nn.Linear(hidden_dim, hidden_dim)
        self.theta0 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ELU(),
        )

        # ── Policy Head ──
        # P(at, st) = MLPπ(μij ‖ νk ‖ ht ‖ λijk)
        # 입력: sel_op(d) + sel_m(d) + graph_emb(2d) + edge_feat = 4d + edge_feat
        self.policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + edge_feat_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        # ── Value Head ──
        # V(st) = MLPv(ht)
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
        action_pairs: List[Tuple[int, int]],
        action_mask: torch.Tensor,
        assembly_map: Dict[int, List[int]],
        precedence_info: Dict,
    ):
        """
        Returns:
            probs: 각 action pair의 확률 (masked softmax)
            value: state value (scalar)
        """
        device = next(self.parameters()).device

        # ── Input Encoding ──
        op_x = graph_data['op'].x.to(device)
        machine_x = graph_data['machine'].x.to(device)
        op_h = self.op_encoder(op_x)           # (num_ops, d)
        machine_h = self.machine_encoder(machine_x)  # (num_machines, d)

        # Edge indices & attrs
        cand_edge = graph_data['op', 'candidate', 'machine'].edge_index.to(device)
        cand_attr = graph_data['op', 'candidate', 'machine'].edge_attr.to(device)
        rev_edge = graph_data['machine', 'candidate_rev', 'op'].edge_index.to(device)
        rev_attr = graph_data['machine', 'candidate_rev', 'op'].edge_attr.to(device)

        # ── Stage 1: GAT Message Passing ──
        for layer_idx in range(self.num_layers):
            # op → machine
            if cand_edge.size(1) > 0:
                m_new = self.gat_op2m[layer_idx](
                    (op_h, machine_h), cand_edge, edge_attr=cand_attr
                )
                machine_h = F.elu(m_new + machine_h)

            # machine → op
            if rev_edge.size(1) > 0:
                o_new = self.gat_m2o[layer_idx](
                    (machine_h, op_h), rev_edge, edge_attr=rev_attr
                )
                op_h = F.elu(o_new + op_h)

        # ── Stage 2: Assembly Dependency ──
        new_op_h = op_h.clone()
        for asm_idx, comp_indices in assembly_map.items():
            if comp_indices and asm_idx < op_h.size(0):
                valid_comps = [c for c in comp_indices if c < op_h.size(0)]
                if valid_comps:
                    comp_agg = op_h[valid_comps].mean(dim=0)
                    new_op_h[asm_idx] = self.mlp_asm(
                        torch.cat([op_h[asm_idx], comp_agg])
                    )
        op_h = new_op_h

        # ── Stage 3: Operation Embedding ──
        prev_map = precedence_info['prev_map']
        next_map = precedence_info['next_map']
        candidate_machines = precedence_info['candidate_machines']
        num_ops = op_h.size(0)

        # Vectorization: Padding for index mapping (index num_ops points to zero vector)
        zero_pad = torch.zeros((1, self.hidden_dim), device=device)
        op_h_padded = torch.cat([op_h, zero_pad], dim=0)

        prev_idx = torch.tensor([prev_map.get(i) if prev_map.get(i) is not None else num_ops for i in range(num_ops)], dtype=torch.long, device=device)
        next_idx = torch.tensor([next_map.get(i) if next_map.get(i) is not None else num_ops for i in range(num_ops)], dtype=torch.long, device=device)

        prev_h = op_h_padded[prev_idx]
        next_h = op_h_padded[next_idx]

        machine_mean = torch.zeros((num_ops, self.hidden_dim), device=device)
        for i in range(num_ops):
            cand_m = candidate_machines.get(i, [])
            if cand_m:
                valid_m = [m for m in cand_m if m < machine_h.size(0)]
                if valid_m:
                    machine_mean[i] = machine_h[valid_m].mean(dim=0)

        new_op_h = self.theta0(torch.cat([
            F.elu(self.theta1(prev_h)),
            F.elu(self.theta2(next_h)),
            F.elu(self.theta3(machine_mean)),
            F.elu(self.theta4(op_h)),
        ], dim=-1))

        op_h = new_op_h

        # ── Graph Pooling ──
        # ht = [mean(ops) ‖ mean(machines)]
        graph_emb = torch.cat([op_h.mean(dim=0), machine_h.mean(dim=0)])

        # ── Policy: score for each action pair ──
        if not action_pairs:
            return torch.tensor([1.0], device=device), self.value_mlp(graph_emb).squeeze()

        # Edge feature lookup table
        edge_feat_map = self._build_edge_feat_map(graph_data, device)

        # Batch scoring
        op_indices = torch.tensor([p[0] for p in action_pairs], dtype=torch.long, device=device)
        m_indices = torch.tensor([p[1] for p in action_pairs], dtype=torch.long, device=device)

        sel_op = op_h[op_indices]                # (N, d)
        sel_m = machine_h[m_indices]             # (N, d)
        graph_exp = graph_emb.unsqueeze(0).expand(len(action_pairs), -1)  # (N, 2d)

        # Edge features for each pair
        edge_feats = []
        for op_id, mid in action_pairs:
            ef = edge_feat_map.get((op_id, mid), torch.zeros(self.edge_feat_dim, device=device))
            edge_feats.append(ef)
        edge_feats = torch.stack(edge_feats)     # (N, edge_dim)

        policy_input = torch.cat([sel_op, sel_m, graph_exp, edge_feats], dim=1)
        logits = self.policy_mlp(policy_input).squeeze(-1)  # (N,)

        # Masking
        mask = action_mask.to(device).bool()
        logits = logits.masked_fill(~mask, -1e9)

        probs = F.softmax(logits, dim=0)
        value = self.value_mlp(graph_emb).squeeze()

        return probs, value

    def _build_edge_feat_map(self, graph_data, device) -> Dict[Tuple[int, int], torch.Tensor]:
        """(op_id, machine_id) → edge_attr 매핑 구축"""
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
# PPO Agent
# ──────────────────────────────────────────────────────────

class RolloutBuffer:
    """에피소드 경험 저장"""

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
        """GAE (Generalized Advantage Estimation)"""
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

        # 정규화
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    """PPO Agent (PPT Slide 13 하이퍼파라미터)"""

    def __init__(
        self,
        policy: HGNNPolicy,
        lr: float = 2e-4,           # PPT
        gamma: float = 1.0,         # PPT
        gae_lambda: float = 0.95,   # PPT
        clip_ratio: float = 0.2,    # PPT
        entropy_coeff: float = 0.01, # PPT
        value_coeff: float = 0.5,   # PPT
        update_epochs: int = 4,     # PPT: 3~5
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

    def select_action(self, obs: dict) -> Tuple[int, float, float]:
        """
        관측에서 action 선택

        Returns:
            action_idx, log_prob, value
        """
        graph = obs["graph"]
        action_pairs = obs["action_pairs"]
        mask = torch.tensor(obs["action_mask"], dtype=torch.float32)
        assembly_map = obs["assembly_map"]
        precedence_info = obs["precedence_info"]

        if not action_pairs:
            return 0, 0.0, 0.0

        with torch.no_grad():
            probs, value = self.policy(
                graph, action_pairs, mask, assembly_map, precedence_info
            )

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store(self, obs, action, log_prob, reward, value, done):
        self.buffer.store(obs, action, log_prob, reward, value, done)

    def update(self) -> Dict[str, float]:
        """PPO 업데이트"""
        if len(self.buffer) == 0:
            return {}

        # 마지막 value 추정
        last_obs = self.buffer.observations[-1]
        with torch.no_grad():
            _, last_val = self.policy(
                last_obs["graph"],
                last_obs["action_pairs"],
                torch.tensor(last_obs["action_mask"], dtype=torch.float32),
                last_obs["assembly_map"],
                last_obs["precedence_info"],
            )
            last_value = last_val.item() if not self.buffer.dones[-1] else 0.0

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32)

        # PPO 업데이트 (여러 epoch)
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0

        for epoch in range(self.update_epochs):
            self.optimizer.zero_grad()
            batch_size = max(len(self.buffer), 1)

            for t in range(len(self.buffer)):
                obs = self.buffer.observations[t]
                action = self.buffer.actions[t]

                graph = obs["graph"]
                action_pairs = obs["action_pairs"]
                mask = torch.tensor(obs["action_mask"], dtype=torch.float32)
                assembly_map = obs["assembly_map"]
                precedence_info = obs["precedence_info"]

                if not action_pairs:
                    continue

                probs, value = self.policy(
                    graph, action_pairs, mask, assembly_map, precedence_info
                )

                dist = Categorical(probs)
                new_log_prob = dist.log_prob(torch.tensor(action))
                entropy = dist.entropy()

                # Clipped surrogate
                ratio = torch.exp(new_log_prob - old_log_probs[t])
                surr1 = ratio * advantages[t]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[t]
                policy_loss = -torch.min(surr1, surr2)

                # Value loss
                value_loss = F.mse_loss(value, returns[t])

                # Total loss (scaled for gradient accumulation over rollout)
                loss = (policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy) / batch_size

                loss.backward()

                total_loss_sum += loss.item() * batch_size
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.item()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        n = max(len(self.buffer) * self.update_epochs, 1)
        metrics = {
            "loss": total_loss_sum / n,
            "policy_loss": policy_loss_sum / n,
            "value_loss": value_loss_sum / n,
            "entropy": entropy_sum / n,
        }

        self.buffer.clear()
        return metrics
