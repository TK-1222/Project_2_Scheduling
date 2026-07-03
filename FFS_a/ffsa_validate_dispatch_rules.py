"""
디스패칭룰 검증 스크립트 (FIFO / EDD / MWKR)
=============================================
동일 인스턴스에서 3가지 룰 각각 50회 반복.
결과: 룰별 각 에피소드 WT + 평균, 표준편차 출력 (그래프용 값)

사용법:
  python ffsa_validate_dispatch_rules.py
  python ffsa_validate_dispatch_rules.py --num-orders 6
"""

import argparse
import numpy as np

from ffsa_instance import full_config
from ffsa_env_dispatch import FFSASchedulingEnv

# ── RL 검증 스크립트와 동일한 시드 ──
TEST_SEEDS = list(range(10000, 10050))   # 50개


# ──────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────

def get_on_time_rate(env) -> float:
    """납기 준수율: 납기 내 완료된 주문 수 / 전체 주문 수"""
    orders = env.instance.orders
    on_time = 0
    for order in orders.values():
        ok = True
        for fid in order.final_job_ids:
            last_op = env.operations[env.job_ops[fid][-1]]
            if not last_op.is_done or last_op.completion_time is None:
                ok = False
                break
            if last_op.completion_time > order.due_date:
                ok = False
                break
        if ok:
            on_time += 1
    return on_time / len(orders) if orders else 0.0


# ──────────────────────────────────────────────────────────
# 룰 선택 함수
# ──────────────────────────────────────────────────────────

def _is_assembly(action) -> bool:
    return isinstance(action[0], tuple)


def _job_id_of(action, env) -> int:
    if _is_assembly(action):
        return env.instance.jobs[action[0][0]].order_id
    op_id = action[0]
    return env.operations[op_id].job_id


def _remaining_ops(job_id, env) -> int:
    """job의 미완료 operation 수"""
    return sum(
        1 for oid in env.job_ops[job_id]
        if not env.operations[oid].is_done
    )


def select_fifo(obs, env) -> int:
    """FIFO: 가장 먼저 등록된 주문(order_id 작은 것) 우선"""
    actions = obs["actions"]
    best_idx, best_key = 0, None
    for i, action in enumerate(actions):
        if _is_assembly(action):
            job   = env.instance.jobs[action[0][0]]
            key   = (job.order_id, job.job_id)
        else:
            op_id, _ = action
            op    = env.operations[op_id]
            job   = env.instance.jobs[op.job_id]
            key   = (job.order_id, job.job_id)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


def select_edd(obs, env) -> int:
    """EDD: 납기 가장 빠른 job 우선"""
    actions = obs["actions"]
    best_idx, best_key = 0, None
    for i, action in enumerate(actions):
        if _is_assembly(action):
            job = env.instance.jobs[action[0][0]]
        else:
            op_id, _ = action
            op  = env.operations[op_id]
            job = env.instance.jobs[op.job_id]
        key = (job.due_date, job.order_id)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


def select_mwkr(obs, env) -> int:
    """MWKR: 잔여 operation 수 가장 많은 job 우선"""
    actions = obs["actions"]
    best_idx, best_key = 0, None
    for i, action in enumerate(actions):
        if _is_assembly(action):
            job_id = action[0][0]
        else:
            op_id, _ = action
            job_id   = env.operations[op_id].job_id
        remaining = _remaining_ops(job_id, env)
        key = (-remaining, job_id)   # 잔여 많을수록 우선 → 음수 사용
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


RULES = {
    "FIFO": select_fifo,
    "EDD":  select_edd,
    "MWKR": select_mwkr,
}


# ──────────────────────────────────────────────────────────
# 검증 루프
# ──────────────────────────────────────────────────────────

def run_rule(rule_name, selector, num_orders, seeds=None):
    if seeds is None:
        seeds = TEST_SEEDS
    print(f"\n── {rule_name} ──")
    wt_list, otr_list = [], []
    for i, seed in enumerate(seeds):
        config = full_config(seed=seed, num_regular_orders=num_orders)
        env    = FFSASchedulingEnv(config)
        obs, _ = env.reset()
        done = truncated = False
        max_steps = 50000
        step_count = 0

        while not (done or truncated) and step_count < max_steps:
            action_idx = selector(obs, env)
            obs, _, done, truncated, _ = env.step(action_idx)
            step_count += 1

        if step_count >= max_steps:
            print(f"  [{i+1:2d}]  seed={seed}  WT=TIMEOUT")
            wt_list.append(float('inf'))
            otr_list.append(0.0)
            continue

        wt  = env.get_actual_weighted_tardiness()
        otr = get_on_time_rate(env)
        wt_list.append(wt)
        otr_list.append(otr)
        print(f"  [{i+1:2d}]  seed={seed}  WT={wt:.2f}  납기준수율={otr*100:.1f}%")

    wt_arr  = np.array(wt_list)
    otr_arr = np.array(otr_list)
    n = len(seeds)
    print(f"\n{rule_name} 결과  ({n}회)")
    print(f"  평균 WT        : {wt_arr.mean():.4f}  (표준편차 {wt_arr.std():.4f})")
    print(f"  평균 납기준수율: {otr_arr.mean()*100:.1f}%  (표준편차 {otr_arr.std()*100:.1f}%)")
    return wt_list, otr_list


def main():
    parser = argparse.ArgumentParser(description="디스패칭룰 검증 (FIFO/EDD/MWKR)")
    parser.add_argument("--num-orders", type=int, default=6,
                        help="정규주문 수 (기본 6 ≈ 50 jobs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="단일 seed 지정 시 해당 seed 1회만 실행 (기본: 10000~10049)")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else TEST_SEEDS

    results = {}
    for name, selector in RULES.items():
        results[name] = run_rule(name, selector, args.num_orders, seeds)

    # ── 최종 요약 ──
    print(f"\n{'='*60}")
    print(f"{'룰':<8}  {'평균 WT':>12}  {'표준편차':>10}  {'납기준수율':>10}")
    print(f"{'-'*60}")
    for name, (wt_list, otr_list) in results.items():
        wt_arr  = np.array(wt_list)
        otr_arr = np.array(otr_list)
        print(f"{name:<8}  {wt_arr.mean():>12.4f}  {wt_arr.std():>10.4f}  {otr_arr.mean()*100:>9.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
