"""
FFSA 학습 루프 — Dual Q-Network (GNN 방식)
==========================================
RegularQNetwork + AssemblyQNetwork 듀얼 에이전트 학습.

업데이트 스케줄 (엇갈린 Window Best-Trajectory):
  - Regular : ep 5, 10, 15 … window 내 WT 최소 trajectory로 업데이트
  - Assembly: ep 6, 11, 16 … Regular 보다 1 에피소드 늦게 업데이트
  - Target  : 각 네트워크 독립적으로 2 update cycle마다 업데이트

네트워크 간 정보 교환:
  - C_p (Regular → Assembly): GNN 임베딩 평균 + pool 상태 3 스칼라
  - U_p (Assembly → Regular): GNN 임베딩 평균 + 긴급도 3 스칼라

모니터링: TensorBoard
  tensorboard --logdir runs/
"""

import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ffsa_instance import InstanceConfig, simple_config, assembly_config, full_config
from ffsa_env import FFSASchedulingEnv
from ffsa_model import RegularQNetwork, AssemblyQNetwork, DualDQNAgent
from ffsa_viz import log_hetero_graph_to_tensorboard


# ──────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────

class Logger:
    def __init__(self, exp_name: str):
        self.writer = SummaryWriter(log_dir=f"runs/{exp_name}")

    def log_episode(self, ep: int, wt: float, makespan: float, reward: float,
                    deadlock: bool, epsilon: float):
        self.writer.add_scalar("episode/weighted_tardiness", wt, ep)
        self.writer.add_scalar("episode/makespan", makespan, ep)
        self.writer.add_scalar("episode/reward", reward, ep)
        self.writer.add_scalar("episode/deadlock", int(deadlock), ep)
        self.writer.add_scalar("train/epsilon", epsilon, ep)

    def log_window_reg(self, ep: int, metrics: dict, best_wt: float,
                       wt_list: list, target_updated: bool):
        self.writer.add_scalar("train/reg_loss",           metrics.get("loss_reg", 0), ep)
        self.writer.add_scalar("train/reg_best_wt",        best_wt,                    ep)
        self.writer.add_scalar("train/reg_mean_wt",        float(np.mean(wt_list)),    ep)
        self.writer.add_scalar("train/reg_target_updated", int(target_updated),         ep)

    def log_window_asm(self, ep: int, metrics: dict, best_wt: float,
                       wt_list: list, target_updated: bool):
        self.writer.add_scalar("train/asm_loss",           metrics.get("loss_asm", 0), ep)
        self.writer.add_scalar("train/asm_best_wt",        best_wt,                    ep)
        self.writer.add_scalar("train/asm_mean_wt",        float(np.mean(wt_list)),    ep)
        self.writer.add_scalar("train/asm_target_updated", int(target_updated),         ep)

    def log_weights(self, ep: int, reg_net: torch.nn.Module, asm_net: torch.nn.Module):
        for name, param in reg_net.named_parameters():
            self.writer.add_histogram(f"weights/reg/{name}", param.data, ep)
            if param.grad is not None:
                self.writer.add_histogram(f"grads/reg/{name}", param.grad, ep)
        for name, param in asm_net.named_parameters():
            self.writer.add_histogram(f"weights/asm/{name}", param.data, ep)
            if param.grad is not None:
                self.writer.add_histogram(f"grads/asm/{name}", param.grad, ep)

    def finish(self):
        self.writer.close()


# ──────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────

def train(
    config: InstanceConfig,
    num_episodes: int = 500,
    window_size: int = 5,
    target_update_cycles: int = 2,
    lr: float = 2e-4,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    hidden_dim: int = 16,
    device: str = "cpu",
    log_interval: int = 10,
    hist_interval: int = 100,
    exp_name: str = "ffsa_dual_dqn",
):
    print(f"{'='*60}")
    print(f"FFSA Dual DQN 학습 시작  [{exp_name}]")
    print(f"  제품 수: {config.num_products}")
    print(f"  Stage 수: {config.num_stages}")
    print(f"  정규주문: {config.num_regular_orders}건")
    print(f"  긴급주문: {config.num_urgent_orders}건")
    print(f"  Assembly: {config.use_assembly}")
    print(f"  Setup: {config.use_setup}")
    print(f"  유한 버퍼: {config.use_finite_buffer}")
    print(f"  Episodes: {num_episodes}  |  Window: {window_size}")
    print(f"  업데이트: ep {window_size}, {window_size*2}, ... (Regular → Assembly 순서, 동시)")

    print(f"  Target 업데이트: {target_update_cycles} 업데이트 사이클마다")
    print(f"  ε: {epsilon_start} → {epsilon_min} (decay={epsilon_decay})")
    print(f"  TensorBoard: runs/{exp_name}")
    print(f"{'='*60}")

    logger = Logger(exp_name)
    env    = FFSASchedulingEnv(config)

    reg_net = RegularQNetwork(
        op_feat_dim=10, machine_feat_dim=6, edge_feat_dim=2,
        hidden_dim=hidden_dim, num_layers=2, mlp_hidden=128,
    )
    asm_net = AssemblyQNetwork(
        op_feat_dim=10, machine_feat_dim=6, edge_feat_dim=2,
        hidden_dim=hidden_dim, num_layers=2, mlp_hidden=128,
    )
    agent = DualDQNAgent(
        reg_net=reg_net, asm_net=asm_net,
        lr=lr, gamma=gamma,
        epsilon_start=epsilon_start, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
        device=device,
    )

    episode_rewards   = []
    episode_tardiness = []
    episode_makespans = []
    episode_deadlocks = []

    window_buffer: list    = []
    reg_update_cycle: int  = 0
    asm_update_cycle: int  = 0
    metrics_reg: dict      = {}
    metrics_asm: dict      = {}

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done         = False
        total_reward = 0.0
        ep_deadlock  = False
        trajectory   = []

        while not done:
            if not obs["actions"]:
                break

            action_idx = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action_idx)

            trajectory.append((obs, action_idx, reward, next_obs, done or truncated))
            total_reward += reward
            obs = next_obs

            if info.get("deadlock"):
                ep_deadlock = True
                trajectory.append((obs, 0, -1000.0, obs, True))
                total_reward += -1000.0
                break

        wt = env.get_actual_weighted_tardiness()
        ms = env.get_makespan()

        episode_rewards.append(total_reward)
        episode_tardiness.append(wt)
        episode_makespans.append(ms)
        episode_deadlocks.append(int(ep_deadlock))

        window_buffer.append((wt, trajectory))

        agent.decay_epsilon()
        logger.log_episode(ep, wt, ms, total_reward, ep_deadlock, agent.epsilon)

        # ── 매 window마다 Regular → Assembly 순서로 동시 업데이트
        reg_updated        = False
        reg_target_updated = False
        asm_updated        = False
        asm_target_updated = False
        if ep % window_size == 0:
            wt_list                = [x[0] for x in window_buffer]
            best_wt, best_traj     = min(window_buffer, key=lambda x: x[0])
            window_buffer          = []

            # ① Regular 먼저 업데이트
            metrics_reg            = agent.update_regular(best_traj)
            reg_update_cycle      += 1
            reg_updated            = True
            if reg_update_cycle % target_update_cycles == 0:
                agent.update_regular_target()
                reg_target_updated = True

            # ② Regular 업데이트 직후 최신 C_p로 Assembly 업데이트
            metrics_asm            = agent.update_assembly(best_traj)
            asm_update_cycle      += 1
            asm_updated            = True
            if asm_update_cycle % target_update_cycles == 0:
                agent.update_assembly_target()
                asm_target_updated = True

            logger.log_window_reg(ep, metrics_reg, best_wt, wt_list, reg_target_updated)
            logger.log_window_asm(ep, metrics_asm, best_wt, wt_list, asm_target_updated)

        if ep % hist_interval == 0:
            logger.log_weights(ep, agent.reg_online, agent.asm_online)

        if ep % log_interval == 0 or ep == 1:
            avg_r  = np.mean(episode_rewards[-log_interval:])
            avg_wt = np.mean(episode_tardiness[-log_interval:])
            avg_ms = np.mean(episode_makespans[-log_interval:])
            avg_dl = np.mean(episode_deadlocks[-log_interval:])
            reg_loss = f"reg={metrics_reg.get('loss_reg', 0):.4f}" if metrics_reg else "reg=--"
            asm_loss = f"asm={metrics_asm.get('loss_asm', 0):.4f}" if metrics_asm else "asm=--"
            dl_str   = f" | DL={avg_dl:.1f}" if avg_dl > 0 else ""
            upd_str  = " | [UPDATE]" if reg_updated else ""
            if reg_target_updated or asm_target_updated: upd_str += " [TGT]"
            print(
                f"[Ep {ep:4d}] "
                f"reward={total_reward:8.2f} (avg={avg_r:8.2f}) | "
                f"WT={wt:8.2f} (avg={avg_wt:8.2f}) | "
                f"MS={ms:8.1f} (avg={avg_ms:8.1f}) | "
                f"ε={agent.epsilon:.3f} | {reg_loss} {asm_loss}{dl_str}{upd_str}"
            )

    print(f"\n{'='*60}")
    print(f"학습 완료!")
    print(f"  최종 평균 Weighted Tardiness: {np.mean(episode_tardiness[-50:]):.2f}")
    print(f"  최종 평균 Makespan: {np.mean(episode_makespans[-50:]):.2f}")
    dl_rate = np.mean(episode_deadlocks[-50:]) * 100
    if dl_rate > 0:
        print(f"  데드락 발생 에피소드 비율 (최근 50): {dl_rate:.1f}%")
    print(f"{'='*60}")

    log_hetero_graph_to_tensorboard(logger.writer, env, num_episodes)
    print(f"  최종 그래프 TensorBoard 저장 완료 (graph/hetero_state)")

    logger.finish()
    return agent, episode_rewards, episode_tardiness, episode_makespans


# ──────────────────────────────────────────────────────────
# Random Agent Test
# ──────────────────────────────────────────────────────────

def test_random_agent(config: InstanceConfig, num_episodes: int = 5):
    print(f"\n{'='*60}")
    print("랜덤 에이전트 테스트")
    print(f"{'='*60}")

    env = FFSASchedulingEnv(config)
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done         = False
        total_reward = 0.0

        while not done:
            actions = obs["actions"]
            if not actions:
                break
            obs, reward, done, truncated, info = env.step(int(np.random.randint(len(actions))))
            total_reward += reward

        wt        = env.get_actual_weighted_tardiness()
        ms        = env.get_makespan()
        completed = info.get("completed_ops", 0)
        total_ops = info.get("total_ops", 0)
        print(
            f"[Ep {ep}] reward={total_reward:.2f} | "
            f"WT={wt:.2f} | MS={ms:.1f} | "
            f"completed={completed}/{total_ops} | done={done}"
        )


# ──────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFSA Dual DQN 학습")
    parser.add_argument("--step",                  type=int,   default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes",              type=int,   default=300)
    parser.add_argument("--window",                type=int,   default=5)
    parser.add_argument("--target-update-cycles",  type=int,   default=2)
    parser.add_argument("--lr",                    type=float, default=2e-4)
    parser.add_argument("--gamma",                 type=float, default=1.0)
    parser.add_argument("--epsilon-start",         type=float, default=1.0)
    parser.add_argument("--epsilon-min",           type=float, default=0.05)
    parser.add_argument("--epsilon-decay",         type=float, default=0.995)
    parser.add_argument("--hidden-dim",            type=int,   default=16)
    parser.add_argument("--products",              type=int,   default=4)
    parser.add_argument("--device",                type=str,   default="cpu")
    parser.add_argument("--exp-name",              type=str,   default=None)
    parser.add_argument("--test-only",             action="store_true")
    args = parser.parse_args()

    step_name = {1: "simple", 2: "assembly", 3: "full"}[args.step]
    exp_name  = args.exp_name or f"dual_dqn_{step_name}_w{args.window}_lr{args.lr}"

    if args.step == 1:
        config = simple_config(num_products=args.products)
    elif args.step == 2:
        config = assembly_config(num_products=args.products)
    else:
        config = full_config(num_products=args.products)

    if args.test_only:
        test_random_agent(config)
    else:
        test_random_agent(config, num_episodes=3)
        train(
            config,
            num_episodes=args.episodes,
            window_size=args.window,
            target_update_cycles=args.target_update_cycles,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            hidden_dim=args.hidden_dim,
            device=args.device,
            exp_name=exp_name,
        )
