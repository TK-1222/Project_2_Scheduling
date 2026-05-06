"""
FFSA 학습 루프
==============
PPT Slide 13: 단계적 실험 전략
  Step 1: 단순 FFSA (assembly 없음)
  Step 2: Assembly 포함
  Step 3: Setup + Buffer

학습 방식: Window 기반 Best-Trajectory 업데이트
  - window_size 에피소드 동안 동일 정책으로 경험 수집 (정책 고정)
  - window 종료 시 가장 낮은 WT를 기록한 에피소드의 trajectory로 1회 정책 업데이트
  - 업데이트된 정책으로 다음 window 시작

모니터링: TensorBoard
  tensorboard --logdir runs/
"""

import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ffsa_instance import InstanceConfig, simple_config, assembly_config, full_config
from ffsa_env import FFSASchedulingEnv
from ffsa_model import HGNNPolicy, PPOAgent
from ffsa_viz import extract_schedule, log_schedule_to_tensorboard, visualize_schedule


# ──────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────

class Logger:
    """TensorBoard 로깅"""

    def __init__(self, exp_name: str):
        self.writer = SummaryWriter(log_dir=f"runs/{exp_name}")

    def log_episode(self, ep: int, wt: float, makespan: float, reward: float, deadlock: bool):
        """에피소드별 지표"""
        self.writer.add_scalar("episode/weighted_tardiness", wt, ep)
        self.writer.add_scalar("episode/makespan", makespan, ep)
        self.writer.add_scalar("episode/reward", reward, ep)
        self.writer.add_scalar("episode/deadlock", int(deadlock), ep)

    def log_window(self, ep: int, metrics: dict, best_wt: float, window_wt_list: list):
        """window 업데이트 시점 지표"""
        self.writer.add_scalar("train/loss",              metrics.get("loss", 0),         ep)
        self.writer.add_scalar("train/policy_loss",       metrics.get("policy_loss", 0),  ep)
        self.writer.add_scalar("train/value_loss",        metrics.get("value_loss", 0),   ep)
        self.writer.add_scalar("train/entropy",           metrics.get("entropy", 0),      ep)
        self.writer.add_scalar("train/best_wt_in_window", best_wt,                        ep)
        self.writer.add_scalar("train/mean_wt_in_window", float(np.mean(window_wt_list)), ep)
        self.writer.add_scalar("train/worst_wt_in_window",float(np.max(window_wt_list)),  ep)

    def log_weights(self, ep: int, policy: torch.nn.Module):
        """신경망 가중치 및 그래디언트 분포 히스토그램"""
        for name, param in policy.named_parameters():
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
    window_size: int = 10,
    lr: float = 2e-4,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
    update_epochs: int = 4,
    hidden_dim: int = 16,
    device: str = "cpu",
    log_interval: int = 10,
    hist_interval: int = 100,
    exp_name: str = "ffsa_run",
):
    print(f"{'='*60}")
    print(f"FFSA 스케줄링 RL 학습 시작  [{exp_name}]")
    print(f"  제품 수: {config.num_products}")
    print(f"  Stage 수: {config.num_stages}")
    print(f"  정규주문: {config.num_regular_orders}건")
    print(f"  긴급주문: {config.num_urgent_orders}건")
    print(f"  Assembly: {config.use_assembly}")
    print(f"  Setup: {config.use_setup}")
    print(f"  유한 버퍼: {config.use_finite_buffer}")
    print(f"  Episodes: {num_episodes}  |  Window: {window_size}")
    print(f"  TensorBoard: runs/{exp_name}")
    print(f"{'='*60}")

    logger = Logger(exp_name)
    env = FFSASchedulingEnv(config)

    policy = HGNNPolicy(
        op_feat_dim=10,
        machine_feat_dim=6,
        edge_feat_dim=2,
        hidden_dim=hidden_dim,
        num_layers=2,
        mlp_hidden=128,
    )

    agent = PPOAgent(
        policy=policy,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        entropy_coeff=entropy_coeff,
        value_coeff=value_coeff,
        update_epochs=update_epochs,
        device=device,
    )

    episode_rewards = []
    episode_tardiness = []
    episode_makespans = []
    episode_deadlocks = []

    window_buffer: list = []
    metrics: dict = {}
    best_schedule: dict = {}

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_steps = env.num_operations * 10
        ep_deadlock = False
        trajectory = []

        with torch.no_grad():
            while not done and steps < max_steps:
                if not obs["actions"]:
                    break

                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, truncated, info = env.step(action)

                trajectory.append((obs, action, log_prob, reward, value, done or truncated))
                total_reward += reward
                obs = next_obs
                steps += 1

                if info.get("deadlock"):
                    ep_deadlock = True
                    break

        wt = env.get_actual_weighted_tardiness()
        ms = env.get_makespan()
        schedule = extract_schedule(env)   # env.reset() 전에 추출

        episode_rewards.append(total_reward)
        episode_tardiness.append(wt)
        episode_makespans.append(ms)
        episode_deadlocks.append(int(ep_deadlock))
        window_buffer.append((wt, trajectory, schedule))

        logger.log_episode(ep, wt, ms, total_reward, ep_deadlock)

        # window 종료: 최고 trajectory로 정책 업데이트
        policy_updated = False
        if ep % window_size == 0:
            window_wt_list = [x[0] for x in window_buffer]
            best_wt, best_traj, best_schedule = min(window_buffer, key=lambda x: x[0])

            agent.buffer.clear()
            for obs_t, act_t, lp_t, r_t, v_t, d_t in best_traj:
                agent.buffer.store(obs_t, act_t, lp_t, r_t, v_t, d_t)
            metrics = agent.update()
            window_buffer = []
            policy_updated = True

            logger.log_window(ep, metrics, best_wt, window_wt_list)
            log_schedule_to_tensorboard(logger.writer, best_schedule, ep)

        # 가중치 히스토그램
        if ep % hist_interval == 0:
            logger.log_weights(ep, policy)

        if ep % log_interval == 0 or ep == 1:
            avg_r  = np.mean(episode_rewards[-log_interval:])
            avg_wt = np.mean(episode_tardiness[-log_interval:])
            avg_ms = np.mean(episode_makespans[-log_interval:])
            avg_dl = np.mean(episode_deadlocks[-log_interval:])
            loss_str = f"loss={metrics.get('loss', 0):.4f}" if metrics else "no update"
            dl_str  = f" | DL={avg_dl:.1f}" if avg_dl > 0 else ""
            upd_str = " | [UPDATE]" if policy_updated else ""
            print(
                f"[Ep {ep:4d}] steps={steps:4d} | "
                f"reward={total_reward:8.2f} (avg={avg_r:8.2f}) | "
                f"WT={wt:8.2f} (avg={avg_wt:8.2f}) | "
                f"MS={ms:8.1f} (avg={avg_ms:8.1f}) | "
                f"{loss_str}{dl_str}{upd_str}"
            )

    print(f"\n{'='*60}")
    print(f"학습 완료!")
    print(f"  최종 평균 Weighted Tardiness: {np.mean(episode_tardiness[-50:]):.2f}")
    print(f"  최종 평균 Makespan: {np.mean(episode_makespans[-50:]):.2f}")
    dl_rate = np.mean(episode_deadlocks[-50:]) * 100
    if dl_rate > 0:
        print(f"  데드락 발생 에피소드 비율 (최근 50): {dl_rate:.1f}%")
    print(f"{'='*60}")

    # 최종 스케줄 그래프 PNG 저장 (마지막 window 최고 에피소드)
    final_schedule = best_schedule
    if window_buffer:
        _, _, final_schedule = min(window_buffer, key=lambda x: x[0])
    if final_schedule:
        import matplotlib.pyplot as plt
        save_path = f"runs/{exp_name}/final_schedule.png"
        fig = visualize_schedule(final_schedule, ep=num_episodes, save_path=save_path)
        plt.close(fig)
        print(f"  최종 스케줄 그래프 저장: {save_path}")

    logger.finish()
    return policy, episode_rewards, episode_tardiness, episode_makespans


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
        done = False
        total_reward = 0.0
        steps = 0
        max_steps = env.num_operations * 10

        while not done and steps < max_steps:
            actions = obs["actions"]
            if not actions:
                break
            action = int(np.random.randint(len(actions)))
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        wt = env.get_actual_weighted_tardiness()
        ms = env.get_makespan()
        completed = info.get("completed_ops", 0)
        total_ops = info.get("total_ops", 0)
        print(
            f"[Ep {ep}] steps={steps} | reward={total_reward:.2f} | "
            f"WT={wt:.2f} | MS={ms:.1f} | "
            f"completed={completed}/{total_ops} | done={done}"
        )


# ──────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFSA 스케줄링 RL 학습")
    parser.add_argument("--step",          type=int,   default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes",      type=int,   default=300)
    parser.add_argument("--window",        type=int,   default=10)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--gamma",         type=float, default=1.0)
    parser.add_argument("--gae-lambda",    type=float, default=0.95)
    parser.add_argument("--clip-ratio",    type=float, default=0.2)
    parser.add_argument("--entropy",       type=float, default=0.01)
    parser.add_argument("--value-coeff",   type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int,   default=4)
    parser.add_argument("--hidden-dim",    type=int,   default=16)
    parser.add_argument("--products",      type=int,   default=4)
    parser.add_argument("--device",        type=str,   default="cpu")
    parser.add_argument("--exp-name",      type=str,   default=None)
    parser.add_argument("--test-only",     action="store_true")
    args = parser.parse_args()

    step_name = {1: "simple", 2: "assembly", 3: "full"}[args.step]
    exp_name = args.exp_name or f"{step_name}_w{args.window}_lr{args.lr}"

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
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            entropy_coeff=args.entropy,
            value_coeff=args.value_coeff,
            update_epochs=args.update_epochs,
            hidden_dim=args.hidden_dim,
            device=args.device,
            exp_name=exp_name,
        )
