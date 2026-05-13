"""
FFSA 학습 루프 — DQN
=====================
PPT Slide 13: 단계적 실험 전략
  Step 1: 단순 FFSA (assembly 없음)
  Step 2: Assembly 포함
  Step 3: Setup + Buffer

학습 방식: DQN (Deep Q-Network)
  - Replay Buffer: 매 스텝 (s, a, r, s', done) 저장
  - Online network: 매 업데이트 주기마다 TD loss로 업데이트
  - Target network: target_update_interval 에피소드마다 online 가중치 복사
  - 탐색: ε-greedy (ε 지수 감소)

모니터링: TensorBoard
  tensorboard --logdir runs/
"""

import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ffsa_instance import InstanceConfig, simple_config, assembly_config, full_config
from ffsa_env import FFSASchedulingEnv
from ffsa_model import HGNNQNetwork, DQNAgent
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

    def log_update(self, ep: int, metrics: dict):
        if metrics:
            self.writer.add_scalar("train/loss", metrics.get("loss", 0), ep)

    def log_weights(self, ep: int, net: torch.nn.Module):
        for name, param in net.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param.data, ep)
            if param.grad is not None:
                self.writer.add_histogram(f"grads/{name}", param.grad, ep)

    def finish(self):
        self.writer.close()


# ──────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────

def train(
    config: InstanceConfig,
    num_episodes: int = 500,
    lr: float = 2e-4,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    buffer_size: int = 10000,
    batch_size: int = 32,
    update_interval: int = 4,
    target_update_interval: int = 10,
    hidden_dim: int = 16,
    device: str = "cpu",
    log_interval: int = 10,
    hist_interval: int = 100,
    exp_name: str = "ffsa_dqn",
):
    print(f"{'='*60}")
    print(f"FFSA 스케줄링 DQN 학습 시작  [{exp_name}]")
    print(f"  제품 수: {config.num_products}")
    print(f"  Stage 수: {config.num_stages}")
    print(f"  정규주문: {config.num_regular_orders}건")
    print(f"  긴급주문: {config.num_urgent_orders}건")
    print(f"  Assembly: {config.use_assembly}")
    print(f"  Setup: {config.use_setup}")
    print(f"  유한 버퍼: {config.use_finite_buffer}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Buffer: {buffer_size}  |  Batch: {batch_size}")
    print(f"  ε: {epsilon_start} → {epsilon_min} (decay={epsilon_decay})")
    print(f"  Target update: 매 {target_update_interval} 에피소드")
    print(f"  TensorBoard: runs/{exp_name}")
    print(f"{'='*60}")

    logger = Logger(exp_name)
    env    = FFSASchedulingEnv(config)

    q_net = HGNNQNetwork(
        op_feat_dim=10,
        machine_feat_dim=6,
        edge_feat_dim=2,
        hidden_dim=hidden_dim,
        num_layers=2,
        mlp_hidden=128,
    )

    agent = DQNAgent(
        q_net=q_net,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_interval=target_update_interval,
        device=device,
    )

    episode_rewards   = []
    episode_tardiness = []
    episode_makespans = []
    episode_deadlocks = []
    metrics: dict = {}
    step_count = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done         = False
        total_reward = 0.0
        ep_deadlock  = False

        while not done:
            if not obs["actions"]:
                break

            action_idx = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action_idx)

            agent.store(obs, action_idx, reward, next_obs, done or truncated)
            total_reward += reward
            step_count   += 1
            obs           = next_obs

            if info.get("deadlock"):
                ep_deadlock = True
                agent.store(obs, 0, -1000.0, obs, True)
                total_reward += -1000.0
                break

            if step_count % update_interval == 0:
                metrics = agent.update()

        wt = env.get_actual_weighted_tardiness()
        ms = env.get_makespan()

        episode_rewards.append(total_reward)
        episode_tardiness.append(wt)
        episode_makespans.append(ms)
        episode_deadlocks.append(int(ep_deadlock))

        agent.decay_epsilon()

        if ep % target_update_interval == 0:
            agent.update_target()

        logger.log_episode(ep, wt, ms, total_reward, ep_deadlock, agent.epsilon)
        logger.log_update(ep, metrics)

        if ep % hist_interval == 0:
            logger.log_weights(ep, agent.online_net)

        if ep % log_interval == 0 or ep == 1:
            avg_r  = np.mean(episode_rewards[-log_interval:])
            avg_wt = np.mean(episode_tardiness[-log_interval:])
            avg_ms = np.mean(episode_makespans[-log_interval:])
            avg_dl = np.mean(episode_deadlocks[-log_interval:])
            loss_str = f"loss={metrics.get('loss', 0):.4f}" if metrics else "updating..."
            dl_str   = f" | DL={avg_dl:.1f}" if avg_dl > 0 else ""
            tgt_str  = " | [TARGET]" if ep % target_update_interval == 0 else ""
            print(
                f"[Ep {ep:4d}] "
                f"reward={total_reward:8.2f} (avg={avg_r:8.2f}) | "
                f"WT={wt:8.2f} (avg={avg_wt:8.2f}) | "
                f"MS={ms:8.1f} (avg={avg_ms:8.1f}) | "
                f"ε={agent.epsilon:.3f} | {loss_str}{dl_str}{tgt_str}"
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
    return agent.online_net, episode_rewards, episode_tardiness, episode_makespans


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
    parser = argparse.ArgumentParser(description="FFSA 스케줄링 DQN 학습")
    parser.add_argument("--step",                   type=int,   default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes",               type=int,   default=300)
    parser.add_argument("--lr",                     type=float, default=2e-4)
    parser.add_argument("--gamma",                  type=float, default=1.0)
    parser.add_argument("--epsilon-start",          type=float, default=1.0)
    parser.add_argument("--epsilon-min",            type=float, default=0.05)
    parser.add_argument("--epsilon-decay",          type=float, default=0.995)
    parser.add_argument("--buffer-size",            type=int,   default=10000)
    parser.add_argument("--batch-size",             type=int,   default=32)
    parser.add_argument("--update-interval",        type=int,   default=4)
    parser.add_argument("--target-update-interval", type=int,   default=10)
    parser.add_argument("--hidden-dim",             type=int,   default=16)
    parser.add_argument("--products",               type=int,   default=4)
    parser.add_argument("--device",                 type=str,   default="cpu")
    parser.add_argument("--exp-name",               type=str,   default=None)
    parser.add_argument("--test-only",              action="store_true")
    args = parser.parse_args()

    step_name = {1: "simple", 2: "assembly", 3: "full"}[args.step]
    exp_name  = args.exp_name or f"dqn_{step_name}_lr{args.lr}"

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
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            update_interval=args.update_interval,
            target_update_interval=args.target_update_interval,
            hidden_dim=args.hidden_dim,
            device=args.device,
            exp_name=exp_name,
        )
