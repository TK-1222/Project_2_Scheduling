"""
FFSA 학습 루프 — Dual Q-Network (디스패칭 룰 버전)
===================================================
ffsa_train.py 와 동일하나 ffsa_env_dispatch.FFSASchedulingEnv 를 사용.
액션 후보를 FIFO / EDD / MWKR / SPT / WINQ 디스패칭 룰로 최대 5개 생성 후
Dual Q-Network (RegularQNetwork + AssemblyQNetwork) Q-value 기반 선택.

업데이트 스케줄 (Experience Replay):
  - Regular + Assembly: 각 버퍼에서 독립적으로 배치 샘플링 후 업데이트
  - train_freq 환경 스텝마다 업데이트 시도
  - learn_start transitions 이상 쌓인 후 학습 시작
  - Target: Soft update (τ=0.005, 매 학습 스텝) — Polyak averaging

인스턴스 설정:
  - ffsa_instance.py InstanceConfig 기본값이 유일한 설정 소스
  - 인스턴스 변경 시 InstanceConfig 기본값만 수정

모니터링: TensorBoard
  tensorboard --logdir runs/
"""

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ffsa_instance import InstanceConfig, full_config, OpFeat
from ffsa_env_dispatch import FFSASchedulingEnv
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
    순수 Q-value argmax 정책으로 여러 eval_env의 WT 평균을 반환.
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
            action_idx = agent.select_greedy(obs)
            obs, _, done, truncated, _ = env.step(action_idx)
            if truncated:
                break
        wt_list.append(env.get_actual_weighted_tardiness())

    agent.epsilon = saved_eps
    return float(np.mean(wt_list))


# ──────────────────────────────────────────────────────────
# 불균형 패널티 / Q-value 가산점
# ──────────────────────────────────────────────────────────

def compute_balance_bonus(obs: dict, env, weight: float = 2.0) -> torch.Tensor:
    """각 후보 액션에 대해 해당 유닛(order_id, unit_idx)의 comp type 균형 기여도에 비례한 가산점을 반환.

    동일 유닛 내 이미 완료된 컴포넌트가 많을수록(조립 임박) -> 양수 보너스
    이미 완료된 컴포넌트 타입을 중복 생산하는 액션 -> 음수 보너스 (FFSA 구조상 불가능하지만 수식의 완비성을 위해 제공)
    조립/final 액션 -> 0
    """
    actions = obs["actions"]
    if not actions:
        return torch.zeros(0)

    # 각 활성 유닛(prod_id, order_id, unit_idx)별 완료/처리 중인 comp_type 세트 수집
    unit_status = {}

    # 1. 조립 pool 체크
    for prod_id, unit_groups in env.assembly_pool.items():
        for (order_id, unit_idx), comp_dict in unit_groups.items():
            key = (prod_id, order_id, unit_idx)
            unit_status[key] = set(comp_dict.keys())

    # 2. 처리 중인 컴포넌트 체크
    for op in env.operations.values():
        if op.active and op.is_processing:
            job = env.instance.jobs[op.job_id]
            if job.is_component:
                key = (job.product_id, job.order_id, job.order_unit_idx)
                if key not in unit_status:
                    unit_status[key] = set()
                unit_status[key].add(job.component_type_idx)

    bonuses = []
    for action in actions:
        if _is_assembly(action):
            bonuses.append(0.0)
            continue
        op_id = action[0]
        job = env.instance.jobs[env.operations[op_id].job_id]
        if not job.is_component:
            bonuses.append(0.0)
            continue

        prod_id = job.product_id
        order_id = job.order_id
        unit_idx = job.order_unit_idx
        ct = job.component_type_idx
        num_types = env.instance.products[prod_id].num_components

        key = (prod_id, order_id, unit_idx)
        ready_types = unit_status.get(key, set())

        # 해당 유닛의 완비율(0 ~ 1)
        mean_count = len(ready_types) / num_types
        current_val = 1.0 if ct in ready_types else 0.0

        # 완비율이 높고( mean_count가 크고) 현재 생산하려는 타입이 아직 준비 안 되었을 때(current_val = 0) 최대 보너스
        bonuses.append((mean_count - current_val) * weight)

    return torch.tensor(bonuses, dtype=torch.float32)


def compute_imbalance_penalty(env, weight: float = 5.0) -> float:
    """유닛(order_id, unit_idx) 단위의 컴포넌트 불균형(대기 중인 불완전 유닛 수)에 비례한 패널티."""
    unit_status = {}

    # 1. 조립 pool 체크
    for prod_id, unit_groups in env.assembly_pool.items():
        for (order_id, unit_idx), comp_dict in unit_groups.items():
            key = (prod_id, order_id, unit_idx)
            unit_status[key] = set(comp_dict.keys())

    # 2. 처리 중인 컴포넌트 체크
    for op in env.operations.values():
        if op.active and op.is_processing:
            job = env.instance.jobs[op.job_id]
            if job.is_component:
                key = (job.product_id, job.order_id, job.order_unit_idx)
                if key not in unit_status:
                    unit_status[key] = set()
                unit_status[key].add(job.component_type_idx)

    total_imbalance = 0.0
    for key, ready_types in unit_status.items():
        prod_id, _, _ = key
        num_types = env.instance.products[prod_id].num_components
        if num_types < 2:
            continue

        # 일부 컴포넌트만 준비되어 불균형 상태인 유닛
        # (WIP: 0 < ready_count < num_types)
        if 0 < len(ready_types) < num_types:
            total_imbalance += 1.0

    return -total_imbalance * weight


# ──────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────

def train(
    config: InstanceConfig,
    num_episodes: int = 500,
    buffer_size: int = 20000,
    batch_size: int = 64,
    train_freq: int = 4,
    learn_start: int = 1000,
    asm_learn_start: int = 200,
    tau: float = 0.005,
    lr: float = 2e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    imbalance_weight: float = 5.0,
    balance_bonus_weight: float = 2.0,
    hidden_dim: int = 16,
    device: str = "cpu",
    log_interval: int = 10,
    hist_interval: int = 100,
    exp_name: str = "ffsa_dual_dqn_dispatch",
    eval_envs: list = None,
    eval_interval: int = 10,
):
    print(f"{'='*60}")
    print(f"FFSA Dual DQN (디스패칭) 학습 시작  [{exp_name}]")
    print(f"  제품 수: {config.num_products}")
    print(f"  Stage 수: {config.num_stages}")
    print(f"  정규주문: {config.num_regular_orders}건")
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
    print(f"  Target 업데이트: Soft update (τ={tau}, 매 학습 스텝)")
    print(f"  ε: {epsilon_start} → {epsilon_min} (decay={epsilon_decay})")
    print(f"  불균형 패널티 weight: {imbalance_weight}")
    print(f"  TensorBoard: runs/{exp_name}")
    print(f"{'='*60}")

    logger = Logger(exp_name)
    env    = FFSASchedulingEnv(config)

    reg_net = RegularQNetwork(
        op_feat_dim=OpFeat.DIM, machine_feat_dim=6, edge_feat_dim=2,
        hidden_dim=hidden_dim, num_layers=2, mlp_hidden=128,
    )
    asm_net = AssemblyQNetwork(
        op_feat_dim=OpFeat.DIM, machine_feat_dim=6, edge_feat_dim=2,
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

    global_step   = 0
    metrics_reg: dict = {}
    metrics_asm: dict = {}
    last_reg_loss = 0.0
    last_asm_loss = 0.0

    save_dir     = f"runs/{exp_name}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_ever_eval_wt = float("inf")  # eval_envs 기반 저장 (데드락 WT=0 면역)

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done         = False
        total_reward = 0.0
        ep_deadlock  = False

        while not done:
            if not obs["actions"]:
                break

            bonus = compute_balance_bonus(obs, env, balance_bonus_weight)
            action_idx = agent.select_action(obs, explore_bonus=bonus)
            next_obs, reward, done, truncated, info = env.step(action_idx)
            step_done = done or truncated
            reward += compute_imbalance_penalty(env, imbalance_weight)
            # 데드락 패널티를 마지막 스텝 reward에 포함 — 단일 전이로 학습 (Sutton & Barto §6.1)
            if info.get("deadlock"):
                reward += -100.0

            # 액션 타입에 따라 버퍼 분리 push
            if _is_assembly(obs["actions"][action_idx]):
                asm_buffer.push(obs, action_idx, reward, next_obs, step_done)
            else:
                reg_buffer.push(obs, action_idx, reward, next_obs, step_done)

            total_reward += reward
            global_step  += 1
            obs = next_obs

            if info.get("deadlock"):
                ep_deadlock = True
                break

            # train_freq 스텝마다 배치 학습 + soft target update
            if global_step % train_freq == 0:
                if len(reg_buffer) >= learn_start:
                    batch         = reg_buffer.sample(batch_size)
                    metrics_reg   = agent.update_regular_batch(batch)
                    last_reg_loss = metrics_reg.get("loss_reg", 0.0)
                    agent.update_regular_target(tau=tau)

                if len(asm_buffer) >= asm_learn_start:
                    batch         = asm_buffer.sample(min(batch_size, len(asm_buffer)))
                    metrics_asm   = agent.update_assembly_batch(batch)
                    last_asm_loss = metrics_asm.get("loss_asm", 0.0)
                    agent.update_assembly_target(tau=tau)

                logger.log_train_step(global_step, last_reg_loss, last_asm_loss)

        wt = env.get_actual_weighted_tardiness()
        ms = env.get_makespan()

        episode_rewards.append(total_reward)
        episode_tardiness.append(wt)
        episode_makespans.append(ms)
        episode_deadlocks.append(int(ep_deadlock))

        agent.decay_epsilon()
        logger.log_episode(ep, wt, ms, total_reward, ep_deadlock, agent.epsilon)
        logger.log_buffer(ep, len(reg_buffer), len(asm_buffer))

        warmup_done = len(reg_buffer) >= learn_start

        # 고정 인스턴스 평가 + 체크포인트 저장 (eval_wt 기준 — 데드락 WT=0 면역)
        if eval_envs and ep % eval_interval == 0:
            eval_wt = run_eval(agent, eval_envs)
            logger.log_eval(ep, eval_wt)
            print(f"  → [EVAL] ep={ep}  eval_wt={eval_wt:.2f}")
            if warmup_done and eval_wt < best_ever_eval_wt:
                best_ever_eval_wt = eval_wt
                torch.save({
                    "ep": ep,
                    "best_eval_wt": best_ever_eval_wt,
                    "reg_online": agent.reg_online.state_dict(),
                    "asm_online": agent.asm_online.state_dict(),
                }, f"{save_dir}/best_model.pt")
                print(f"  → [SAVED] ep={ep}  eval_wt={best_ever_eval_wt:.2f}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFSA Dual DQN (디스패칭) 학습")
    parser.add_argument("--episodes",           type=int,   default=300)
    parser.add_argument("--buffer-size",        type=int,   default=20000)
    parser.add_argument("--batch-size",         type=int,   default=64)
    parser.add_argument("--train-freq",         type=int,   default=4)
    parser.add_argument("--learn-start",        type=int,   default=1000)
    parser.add_argument("--asm-learn-start",    type=int,   default=200)
    parser.add_argument("--tau",                type=float, default=0.005)
    parser.add_argument("--lr",                 type=float, default=2e-4)
    parser.add_argument("--gamma",              type=float, default=0.99)
    parser.add_argument("--epsilon-start",      type=float, default=1.0)
    parser.add_argument("--epsilon-min",        type=float, default=0.05)
    parser.add_argument("--epsilon-decay",      type=float, default=0.995)
    parser.add_argument("--imbalance-weight",     type=float, default=5.0)
    parser.add_argument("--balance-bonus-weight", type=float, default=2.0)
    parser.add_argument("--hidden-dim",           type=int,   default=16)
    parser.add_argument("--device",             type=str,   default="cpu")
    parser.add_argument("--exp-name",           type=str,   default=None)
    parser.add_argument("--test-only",          action="store_true")
    parser.add_argument("--eval-seeds",         type=int,   nargs="+", default=[42, 123, 777])
    parser.add_argument("--eval-interval",      type=int,   default=10)
    args = parser.parse_args()

    exp_name  = args.exp_name or f"dual_dqn_dispatch_buf{args.buffer_size}_lr{args.lr}"

    # 학습: 실행마다 랜덤 시드 추출 → 전체 에피소드 동일 인스턴스 사용
    train_seed = random.randint(0, 99999)
    print(f"학습 인스턴스 시드: {train_seed}")
    config = full_config(seed=train_seed)

    # 평가: 고정 시드 인스턴스 (일반화 확인용, 선택)
    eval_envs = [
        FFSASchedulingEnv(full_config(seed=s))
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
            asm_learn_start=args.asm_learn_start,
            tau=args.tau,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            imbalance_weight=args.imbalance_weight,
            balance_bonus_weight=args.balance_bonus_weight,
            hidden_dim=args.hidden_dim,
            device=args.device,
            exp_name=exp_name,
            eval_envs=eval_envs,
            eval_interval=args.eval_interval,
        )
