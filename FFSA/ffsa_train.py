"""
FFSA 학습 루프
==============
PPT Slide 13: 단계적 실험 전략
  Step 1: 단순 FFSA (assembly 없음)
  Step 2: Assembly 포함
  Step 3: Setup + Buffer
"""

import argparse
import numpy as np
import torch

from ffsa_instance import InstanceConfig, simple_config, assembly_config, full_config
from ffsa_env import FFSASchedulingEnv
from ffsa_model import HGNNPolicy, PPOAgent


def train(
    config: InstanceConfig,
    num_episodes: int = 500,
    device: str = "cpu",
    log_interval: int = 10,
):
    print(f"{'='*60}")
    print(f"FFSA 스케줄링 RL 학습 시작")
    print(f"  제품 수: {config.num_products}")
    print(f"  Stage 수: {config.num_stages}")
    print(f"  주문 수: {config.orders_per_product}")
    print(f"  Assembly: {config.use_assembly}")
    print(f"  Setup: {config.use_setup}")
    print(f"  유한 버퍼: {config.use_finite_buffer}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    env = FFSASchedulingEnv(config)

    policy = HGNNPolicy(
        op_feat_dim=10,
        machine_feat_dim=6,
        edge_feat_dim=2,
        hidden_dim=16,
        num_layers=2,
        mlp_hidden=128,
    )

    agent = PPOAgent(
        policy=policy,
        lr=2e-4,
        gamma=1.0,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=0.01,
        value_coeff=0.5,
        update_epochs=4,
        target_update_interval=30,
        device=device,
    )

    episode_rewards = []
    episode_tardiness = []
    episode_makespans = []
    episode_deadlocks = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_steps = env.num_operations * 10
        ep_deadlock = False

        while not done and steps < max_steps:
            if not obs["actions"]:
                break

            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            agent.store(obs, action, log_prob, reward, value, done or truncated)
            total_reward += reward
            obs = next_obs
            steps += 1

            if info.get("deadlock"):
                ep_deadlock = True
                break

        metrics = agent.update()

        wt = env.get_actual_weighted_tardiness()
        ms = env.get_makespan()
        agent.record_episode(wt)
        episode_rewards.append(total_reward)
        episode_tardiness.append(wt)
        episode_makespans.append(ms)
        episode_deadlocks.append(int(ep_deadlock))

        target_updated = (agent._episode_count % agent.target_update_interval == 0)

        if ep % log_interval == 0 or ep == 1:
            avg_r  = np.mean(episode_rewards[-log_interval:])
            avg_wt = np.mean(episode_tardiness[-log_interval:])
            avg_ms = np.mean(episode_makespans[-log_interval:])
            avg_dl = np.mean(episode_deadlocks[-log_interval:])
            loss_str = f"loss={metrics.get('loss', 0):.4f}" if metrics else "no update"
            dl_str = f" | DL={avg_dl:.1f}" if avg_dl > 0 else ""
            tgt_str = " | [TGT]" if target_updated else ""
            print(
                f"[Ep {ep:4d}] steps={steps:4d} | "
                f"reward={total_reward:8.2f} (avg={avg_r:8.2f}) | "
                f"WT={wt:8.2f} (avg={avg_wt:8.2f}) | "
                f"MS={ms:8.1f} (avg={avg_ms:8.1f}) | "
                f"{loss_str}{dl_str}{tgt_str}"
            )

    print(f"\n{'='*60}")
    print(f"학습 완료!")
    print(f"  최종 평균 Weighted Tardiness: {np.mean(episode_tardiness[-50:]):.2f}")
    print(f"  최종 평균 Makespan: {np.mean(episode_makespans[-50:]):.2f}")
    dl_rate = np.mean(episode_deadlocks[-50:]) * 100
    if dl_rate > 0:
        print(f"  데드락 발생 에피소드 비율 (최근 50): {dl_rate:.1f}%")
    print(f"{'='*60}")

    return policy, episode_rewards, episode_tardiness, episode_makespans


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFSA 스케줄링 RL 학습")
    parser.add_argument("--step", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--products", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

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
        train(config, num_episodes=args.episodes, device=args.device)
