"""
FFSA 학습 루프 — Dual Q-Network (GNN 방식)
==========================================
RegularQNetwork + AssemblyQNetwork 듀얼 에이전트 학습.

업데이트 스케줄 (Experience Replay):
  - Regular + Assembly: 각 버퍼에서 독립적으로 배치 샘플링 후 업데이트
  - train_freq 환경 스텝마다 업데이트 시도
  - learn_start transitions 이상 쌓인 후 학습 시작
  - Target: 각 네트워크 target_update_freq 업데이트마다 독립적으로 갱신

인스턴스 전략:
  - 학습: 실행 시작 시 시드를 랜덤 추출 → 전체 에피소드 동일 인스턴스 사용
  - 평가: 고정 시드 인스턴스 (--eval-seeds) → 일반화 성능 확인용 (선택)

모니터링: TensorBoard
  tensorboard --logdir runs/
"""

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ffsa_instance import InstanceConfig, simple_config, assembly_config, full_config
from ffsa_env import FFSASchedulingEnv
from ffsa_model import RegularQNetwork, AssemblyQNetwork, DualDQNAgent, ReplayBuffer, _is_assembly
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

    def log_train_step(self, step: int, reg_loss: float, asm_loss: float):
        if reg_loss > 0:
            self.writer.add_scalar("train/reg_loss", reg_loss, step)
        if asm_loss > 0:
            self.writer.add_scalar("train/asm_loss", asm_loss, step)

    def log_train_target(self, step: int, reg_updated: bool, asm_updated: bool):
        if reg_updated:
            self.writer.add_scalar("train/reg_target_updated", 1, step)
        if asm_updated:
            self.writer.add_scalar("train/asm_target_updated", 1, step)

    def log_buffer(self, ep: int, reg_size: int, asm_size: int):
        self.writer.add_scalar("train/reg_buffer_size", reg_size, ep)
        self.writer.add_scalar("train/asm_buffer_size", asm_size, ep)

    def log_eval(self, ep: int, eval_wt: float):
        self.writer.add_scalar("eval/weighted_tardiness", eval_wt, ep)

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
# 고정 인스턴스 평가
# ──────────────────────────────────────────────────────────

def run_eval(agent: DualDQNAgent, eval_envs: list) -> float:
    """
    고정 인스턴스로 greedy 정책 평가 (epsilon=0, 파라미터 업데이트 없음).
    여러 eval_env의 WT 평균을 반환.
    """
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    wt_list = []
    for env in eval_envs:
        obs, _ = env.reset()
        done = False
        while not done:
            if not obs["actions"]:
                break
            action_idx = agent.select_action(obs)
            obs, _, done, truncated, _ = env.step(action_idx)
            if truncated:
                break
        wt_list.append(env.get_actual_weighted_tardiness())

    agent.epsilon = saved_eps
    return float(np.mean(wt_list))


# ──────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────

def train(
    config: InstanceConfig,
    num_episodes: int = 500,
    buffer_size: int = 50000,
    batch_size: int = 64,
    train_freq: int = 4,
    learn_start: int = 1000,
    target_update_freq: int = 100,
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
    eval_envs: list = None,
    eval_interval: int = 10,
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
    print(f"  학습 인스턴스: 고정 (seed={config.seed}, 실행마다 랜덤 추출)")
    if eval_envs:
        print(f"  평가 인스턴스: {len(eval_envs)}개 고정 (일반화 확인용)")
    else:
        print(f"  평가: 없음")
    print(f"  Episodes: {num_episodes}")
    print(f"  Replay Buffer: reg/asm 별도 | 크기={buffer_size} | 배치={batch_size}")
    print(f"  학습 주기: {train_freq} 스텝마다 | 워밍업: {learn_start} transitions")
    print(f"  Target 업데이트: {target_update_freq} 업데이트마다")
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

    reg_buffer = ReplayBuffer(buffer_size)
    asm_buffer = ReplayBuffer(buffer_size)

    episode_rewards   = []
    episode_tardiness = []
    episode_makespans = []
    episode_deadlocks = []

    global_step      = 0
    reg_update_count = 0
    asm_update_count = 0
    metrics_reg: dict = {}
    metrics_asm: dict = {}
    last_reg_loss    = 0.0
    last_asm_loss    = 0.0

    save_dir     = f"runs/{exp_name}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_ever_wt = float("inf")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done         = False
        total_reward = 0.0
        ep_deadlock  = False

        reg_target_updated = False
        asm_target_updated = False

        while not done:
            if not obs["actions"]:
                break

            action_idx = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action_idx)
            step_done = done or truncated

            # 액션 타입에 따라 버퍼 분리 push
            if _is_assembly(obs["actions"][action_idx]):
                asm_buffer.push(obs, action_idx, reward, next_obs, step_done)
            else:
                reg_buffer.push(obs, action_idx, reward, next_obs, step_done)

            total_reward += reward
            global_step  += 1
            obs = next_obs

            if info.get("deadlock"):
                ep_deadlock   = True
                total_reward += -1000.0
                # done=True이므로 target = -1000 (액션 인덱스는 학습에 영향 없음)
                dl_act = next(
                    (i for i, a in enumerate(obs["actions"]) if not _is_assembly(a)),
                    0
                )
                reg_buffer.push(obs, dl_act, -1000.0, obs, True)
                break

            # train_freq 스텝마다 배치 학습
            if global_step % train_freq == 0:
                if len(reg_buffer) >= learn_start:
                    batch         = reg_buffer.sample(batch_size)
                    metrics_reg   = agent.update_regular_batch(batch)
                    last_reg_loss = metrics_reg.get("loss_reg", 0.0)
                    reg_update_count += 1
                    if reg_update_count % target_update_freq == 0:
                        agent.update_regular_target()
                        reg_target_updated = True

                if len(asm_buffer) >= learn_start:
                    batch         = asm_buffer.sample(batch_size)
                    metrics_asm   = agent.update_assembly_batch(batch)
                    last_asm_loss = metrics_asm.get("loss_asm", 0.0)
                    asm_update_count += 1
                    if asm_update_count % target_update_freq == 0:
                        agent.update_assembly_target()
                        asm_target_updated = True

                logger.log_train_step(global_step, last_reg_loss, last_asm_loss)
                if reg_target_updated or asm_target_updated:
                    logger.log_train_target(global_step, reg_target_updated, asm_target_updated)
                    reg_target_updated = False
                    asm_target_updated = False

        wt = env.get_actual_weighted_tardiness()
        ms = env.get_makespan()

        episode_rewards.append(total_reward)
        episode_tardiness.append(wt)
        episode_makespans.append(ms)
        episode_deadlocks.append(int(ep_deadlock))

        agent.decay_epsilon()
        logger.log_episode(ep, wt, ms, total_reward, ep_deadlock, agent.epsilon)
        logger.log_buffer(ep, len(reg_buffer), len(asm_buffer))

        # 고정 인스턴스 평가
        if eval_envs and ep % eval_interval == 0:
            eval_wt = run_eval(agent, eval_envs)
            logger.log_eval(ep, eval_wt)
            print(f"  → [EVAL] ep={ep}  eval_wt={eval_wt:.2f}")

        # 워밍업 완료 후 최저 WT 갱신 시 저장
        warmup_done = len(reg_buffer) >= learn_start
        if warmup_done and wt < best_ever_wt:
            best_ever_wt = wt
            torch.save({
                "ep": ep,
                "best_wt": best_ever_wt,
                "reg_online": agent.reg_online.state_dict(),
                "asm_online": agent.asm_online.state_dict(),
            }, f"{save_dir}/best_model.pt")
            print(f"  → [SAVED] ep={ep}  best_wt={best_ever_wt:.2f}")

        if ep % hist_interval == 0:
            logger.log_weights(ep, agent.reg_online, agent.asm_online)

        if ep % log_interval == 0 or ep == 1:
            avg_r  = np.mean(episode_rewards[-log_interval:])
            avg_wt = np.mean(episode_tardiness[-log_interval:])
            avg_ms = np.mean(episode_makespans[-log_interval:])
            avg_dl = np.mean(episode_deadlocks[-log_interval:])
            reg_loss_str = f"reg={last_reg_loss:.4f}" if metrics_reg else "reg=--"
            asm_loss_str = f"asm={last_asm_loss:.4f}" if metrics_asm else "asm=--"
            buf_str = f" | buf=reg:{len(reg_buffer)}/asm:{len(asm_buffer)}"
            dl_str  = f" | DL={avg_dl:.1f}" if avg_dl > 0 else ""
            print(
                f"[Ep {ep:4d}] "
                f"reward={total_reward:8.2f} (avg={avg_r:8.2f}) | "
                f"WT={wt:8.2f} (avg={avg_wt:8.2f}) | "
                f"MS={ms:8.1f} (avg={avg_ms:8.1f}) | "
                f"ε={agent.epsilon:.3f} | {reg_loss_str} {asm_loss_str}{buf_str}{dl_str}"
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

def _make_config(step: int, num_products: int, seed) -> InstanceConfig:
    if step == 1:
        return simple_config(num_products=num_products, seed=seed)
    elif step == 2:
        return assembly_config(num_products=num_products, seed=seed)
    else:
        return full_config(num_products=num_products, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFSA Dual DQN 학습")
    parser.add_argument("--step",               type=int,   default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes",           type=int,   default=300)
    parser.add_argument("--buffer-size",        type=int,   default=50000)
    parser.add_argument("--batch-size",         type=int,   default=64)
    parser.add_argument("--train-freq",         type=int,   default=4)
    parser.add_argument("--learn-start",        type=int,   default=1000)
    parser.add_argument("--target-update-freq", type=int,   default=100)
    parser.add_argument("--lr",                 type=float, default=2e-4)
    parser.add_argument("--gamma",              type=float, default=1.0)
    parser.add_argument("--epsilon-start",      type=float, default=1.0)
    parser.add_argument("--epsilon-min",        type=float, default=0.05)
    parser.add_argument("--epsilon-decay",      type=float, default=0.995)
    parser.add_argument("--hidden-dim",         type=int,   default=16)
    parser.add_argument("--products",           type=int,   default=4)
    parser.add_argument("--device",             type=str,   default="cpu")
    parser.add_argument("--exp-name",           type=str,   default=None)
    parser.add_argument("--test-only",          action="store_true")
    parser.add_argument("--eval-seeds",         type=int,   nargs="+", default=[42, 123, 777])
    parser.add_argument("--eval-interval",      type=int,   default=10)
    args = parser.parse_args()

    step_name = {1: "simple", 2: "assembly", 3: "full"}[args.step]
    exp_name  = args.exp_name or f"dual_dqn_{step_name}_buf{args.buffer_size}_lr{args.lr}"

    # 학습: 실행마다 랜덤 시드 추출 → 전체 에피소드 동일 인스턴스 사용
    train_seed = random.randint(0, 99999)
    print(f"학습 인스턴스 시드: {train_seed}")
    config = _make_config(args.step, args.products, seed=train_seed)

    # 평가: 고정 시드 인스턴스 (일반화 확인용, 선택)
    eval_envs = [
        FFSASchedulingEnv(_make_config(args.step, args.products, seed=s))
        for s in args.eval_seeds
    ]

    if args.test_only:
        test_random_agent(config)
    else:
        test_random_agent(config, num_episodes=3)
        train(
            config,
            num_episodes=args.episodes,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            learn_start=args.learn_start,
            target_update_freq=args.target_update_freq,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            hidden_dim=args.hidden_dim,
            device=args.device,
            exp_name=exp_name,
            eval_envs=eval_envs,
            eval_interval=args.eval_interval,
        )
