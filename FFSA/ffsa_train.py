"""
FFSA 학습 루프
==============
PPT Slide 13: 단계적 실험 전략
  Step 1: 단순 FFSA (assembly 없음, α=1.0만)
  Step 2: Assembly 포함, CLB feature 검증
  Step 3: Setup + Buffer, 보조 reward 추가
  Step 4: 일반화 학습
"""

import argparse
import time
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
    use_completion_bonus: bool = False,
    use_idle_penalty: bool = False,
    use_buffer_penalty: bool = False,
):
    """학습 메인 루프"""
    print(f"{'='*60}")
    print(f"FFSA 스케줄링 RL 학습 시작")
    print(f"  제품 수: {config.num_products}")
    print(f"  Stage 수: {config.num_stages}")
    print(f"  Assembly: {config.use_assembly}")
    print(f"  Setup: {config.use_setup}")
    print(f"  유한 버퍼: {config.use_finite_buffer}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # 환경 생성
    env = FFSASchedulingEnv(config)
    env.use_completion_bonus = use_completion_bonus
    env.use_idle_penalty = use_idle_penalty
    env.use_buffer_penalty = use_buffer_penalty

    # 정책 네트워크 생성
    policy = HGNNPolicy(
        op_feat_dim=14,
        machine_feat_dim=7,
        edge_feat_dim=2,
        hidden_dim=16,     # PPT
        num_layers=2,      # PPT
        mlp_hidden=128,    # PPT
    )

    # PPO Agent
    agent = PPOAgent(
        policy=policy,
        lr=2e-4,           # PPT
        gamma=1.0,         # PPT
        gae_lambda=0.95,   # PPT
        clip_ratio=0.2,    # PPT
        entropy_coeff=0.01,# PPT
        value_coeff=0.5,   # PPT
        update_epochs=4,   # PPT: 3~5
        device=device,
    )

    # 학습 기록
    episode_rewards = []
    episode_tardiness = []
    episode_makespans = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_steps = 5000  # 안전장치

        while not done and steps < max_steps:
            if not obs["action_pairs"]:
                break

            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            agent.store(obs, action, log_prob, reward, value, done)
            total_reward += reward
            obs = next_obs
            steps += 1

        # 에피소드 종료 후 PPO 업데이트
        metrics = agent.update()

        # 성능 기록
        wt = env.get_actual_weighted_tardiness()
        ms = env.get_makespan()
        episode_rewards.append(total_reward)
        episode_tardiness.append(wt)
        episode_makespans.append(ms)

        if ep % log_interval == 0 or ep == 1:
            avg_r = np.mean(episode_rewards[-log_interval:])
            avg_wt = np.mean(episode_tardiness[-log_interval:])
            avg_ms = np.mean(episode_makespans[-log_interval:])
            loss_str = f"loss={metrics.get('loss', 0):.4f}" if metrics else "no update"
            print(
                f"[Ep {ep:4d}] "
                f"steps={steps:4d} | "
                f"reward={total_reward:8.2f} (avg={avg_r:8.2f}) | "
                f"WT={wt:8.2f} (avg={avg_wt:8.2f}) | "
                f"MS={ms:8.1f} (avg={avg_ms:8.1f}) | "
                f"{loss_str}"
            )

    print(f"\n{'='*60}")
    print(f"학습 완료!")
    print(f"  최종 평균 Weighted Tardiness: {np.mean(episode_tardiness[-50:]):.2f}")
    print(f"  최종 평균 Makespan: {np.mean(episode_makespans[-50:]):.2f}")
    print(f"{'='*60}")

    return policy, episode_rewards, episode_tardiness, episode_makespans


def test_random_agent(config: InstanceConfig, num_episodes: int = 5):
    """랜덤 에이전트로 환경 테스트"""
    print(f"\n{'='*60}")
    print("랜덤 에이전트 테스트")
    print(f"{'='*60}")

    env = FFSASchedulingEnv(config)

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_steps = 5000

        while not done and steps < max_steps:
            action_pairs = obs["action_pairs"]
            if not action_pairs:
                break

            action = int(np.random.randint(len(action_pairs)))
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
    parser.add_argument("--step", type=int, default=1, choices=[1, 2, 3],
                        help="실험 단계: 1=simple, 2=assembly, 3=full")
    parser.add_argument("--episodes", type=int, default=100,
                        help="학습 에피소드 수")
    parser.add_argument("--test-only", action="store_true",
                        help="랜덤 에이전트로 환경만 테스트")
    parser.add_argument("--products", type=int, default=4,
                        help="제품 수")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device")

    args = parser.parse_args()

    # 실험 단계에 따른 config 선택
    if args.step == 1:
        config = simple_config(num_products=args.products)
    elif args.step == 2:
        config = assembly_config(num_products=args.products)
    else:
        config = full_config(num_products=args.products)

    if args.test_only:
        test_random_agent(config)
    else:
        # 먼저 랜덤 에이전트로 환경 테스트
        test_random_agent(config, num_episodes=3)

        # 학습
        train(
            config,
            num_episodes=args.episodes,
            device=args.device,
            use_completion_bonus=(args.step >= 3),
            use_idle_penalty=(args.step >= 3),
            use_buffer_penalty=(args.step >= 3),
        )
