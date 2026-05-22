"""
FFSA 이종 그래프 시각화
========================
[태림]기본 그래프 수정.py 스타일 계승:
  - networkx DiGraph 기반
  - nx.draw_networkx_nodes / edges + ax.text 패턴
  - Stage-Column 레이아웃

레이아웃:
  X축  : 스테이지 번호 (stage × X_GAP)
  Y 상단: 기계 노드 (정사각형 ■)  — 스테이지별 묶음
  Y 중단: 오퍼레이션 노드 (삼각형 ▲) — job별 행
  X 중간: 버퍼 노드 (원 ●) — 스테이지 사이

엣지:
  Precedence  : op → op  검정 화살표
  Candidate   : machine → ready op  점선, 기계별 색
  Assigned    : machine → processing op  실선 굵게, 기계별 색

연동:
  draw_hetero_graph(env, ep, ax)       → plt.ion() 팝업 창 실시간 갱신
  log_hetero_graph_to_tensorboard(...) → TensorBoard Images 탭
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager
from typing import Optional, Dict, List, Tuple


# ─────────────────────────────────────────────────────────
# 스타일 상수 (태림 코드 계승)
# ─────────────────────────────────────────────────────────

NODE_SHAPE = {
    "machine":   "s",   # 정사각형 ■
    "operation": "^",   # 삼각형  ▲ (태림 operation 모양)
    "buffer":    "o",   # 원      ● (태림 buffer 모양)
}
NODE_SIZE = {
    "machine":   800,
    "operation": 1000,
    "buffer":    500,
}

# Op 상태별 (fill, edge_color)
OP_STYLE = {
    "done":       ("#1565c0", "#0d47a1"),   # 진파랑   : 완료
    "processing": ("#8ecae6", "#e63946"),   # 파랑+빨강 테두리: 처리 중
    "ready":      ("#8ecae6", "#5b8bd0"),   # 태림 파랑 : 배정 대기
    "waiting":    ("#b0bec5", "#78909c"),   # 회청     : 버퍼 대기
    "inactive":   ("#eeeeee", "#bdbdbd"),   # 연회색   : 비활성
}

MACHINE_IDLE_STYLE = ("#f4a261", "#e76f51")   # 연주황: 유휴
MACHINE_BUSY_STYLE = ("#ef233c", "#9b2226")   # 빨강  : 처리 중
BUFFER_STYLE       = ("#86df7f", "#2e8b57")   # 태림 버퍼 초록

# 기계별 고유 색 (candidate / assigned 엣지 + 범례)
MACHINE_PALETTE = [
    "#e41a1c", "#377eb8", "#ff7f00", "#4daf4a",
    "#984ea3", "#a65628", "#f781bf", "#636363",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]

EDGE_PREC_COLOR = "#222222"   # precedence 화살표 색
LABEL_OFFSET_Y  = -0.30       # 태림과 같은 라벨 아래 오프셋

# 레이아웃
X_GAP          = 4.5    # 스테이지 간 X 간격
BUF_X_OFFSET   = 2.25   # 버퍼 X = stage_x + BUF_X_OFFSET
Y_MACHINE_TOP  = 4.2    # 기계 행 Y
MACHINE_H_GAP  = 1.1    # 같은 스테이지 내 기계 수평 간격
Y_OP_BASE      = 0.5    # 첫 번째 job op Y
Y_OP_GAP       = -1.8   # job 행 간 Y 간격 (아래로)


# ─────────────────────────────────────────────────────────
# 한글 폰트 (태림 코드와 동일한 로직)
# ─────────────────────────────────────────────────────────

def _init_font() -> Optional[FontProperties]:
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None
    fontManager.addfont(path)
    fp = FontProperties(fname=path)
    rcParams["font.family"] = fp.get_name()
    rcParams["axes.unicode_minus"] = False
    return fp


# ─────────────────────────────────────────────────────────
# 그래프 & 좌표 생성
# ─────────────────────────────────────────────────────────

def _build_graph_and_pos(env) -> Tuple[nx.DiGraph, dict, list, Dict[int, str]]:
    """
    env 실시간 상태 → 시각화용 networkx DiGraph + 좌표 반환.

    노드 종류:
      M{mid}   : 기계
      Op{oid}  : 오퍼레이션 (active op만 포함)
      BUF{sid} : 버퍼 (스테이지 경계마다 1개)

    엣지 종류:
      precedence : op → op  (stage 순서)
      candidate  : machine → ready op  (처리 가능 후보, 점선)
      assigned   : machine → op  (현재 배정, 실선)
    """
    G = nx.DiGraph()
    pos: dict = {}

    num_stages = env.instance.num_stages
    mbs = env.instance.machines_by_stage

    # ── 활성 job 목록 (active op이 1개라도 있는 job) ──
    active_jobs = sorted([
        jid for jid in env.job_ops
        if any(env.operations[oid].active for oid in env.job_ops[jid])
    ])
    job_y = {jid: Y_OP_BASE + i * Y_OP_GAP for i, jid in enumerate(active_jobs)}

    # ── 기계별 색상 매핑 ──
    all_mids = sorted(env.machine_states.keys())
    m_color: Dict[int, str] = {
        mid: MACHINE_PALETTE[i % len(MACHINE_PALETTE)]
        for i, mid in enumerate(all_mids)
    }

    # ── 기계 노드 ──
    for sid in range(num_stages):
        mids = sorted(mbs.get(sid, []))
        n = len(mids)
        x_base = sid * X_GAP
        for i, mid in enumerate(mids):
            key = f"M{mid}"
            ms = env.machine_states[mid]
            is_busy = not ms.is_idle
            m_x = x_base + (i - (n - 1) / 2.0) * MACHINE_H_GAP
            G.add_node(key, ntype="machine", machine_id=mid,
                       is_busy=is_busy, stage_id=sid)
            pos[key] = (m_x, Y_MACHINE_TOP)

    # ── Op 노드 ──
    for jid in active_jobs:
        for oid in env.job_ops[jid]:
            op = env.operations[oid]
            if not op.active:
                continue
            key = f"Op{oid}"
            x = op.stage_id * X_GAP
            y = job_y[jid]

            if op.is_done:
                status = "done"
            elif op.is_processing:
                status = "processing"
            elif op.is_ready:
                status = "ready"
            elif op.buffer_waiting:
                status = "waiting"
            else:
                status = "inactive"

            G.add_node(key, ntype="operation", op_id=oid, job_id=jid,
                       stage_id=op.stage_id, status=status,
                       product_id=op.product_id, machine_id=op.machine_id)
            pos[key] = (x, y)

    # ── 버퍼 노드 (스테이지 경계마다 1개) ──
    buf_y = (sum(job_y.values()) / len(job_y)) if job_y else 0.0
    for sid in range(num_stages - 1):
        key = f"BUF{sid}"
        occupancy = len(env.buffers[sid].queue) if sid in env.buffers else 0
        G.add_node(key, ntype="buffer", stage_id=sid, occupancy=occupancy)
        pos[key] = (sid * X_GAP + BUF_X_OFFSET, buf_y)

    # ── Precedence 엣지 (op → BUF → op, 버퍼 경유) ──
    for jid in active_jobs:
        ops_list = env.job_ops[jid]
        for idx, oid in enumerate(ops_list):
            op = env.operations[oid]
            if not op.active:
                continue
            if idx < len(ops_list) - 1:
                next_oid = ops_list[idx + 1]
                next_op = env.operations[next_oid]
                if next_op.active:
                    buf_key = f"BUF{op.stage_id}"
                    if buf_key in G.nodes:
                        G.add_edge(f"Op{oid}", buf_key, etype="precedence")
                        G.add_edge(buf_key, f"Op{next_oid}", etype="precedence")
                    else:
                        G.add_edge(f"Op{oid}", f"Op{next_oid}", etype="precedence")

    # ── Candidate / Assigned 엣지 ──
    for jid in active_jobs:
        for oid in env.job_ops[jid]:
            op = env.operations[oid]
            if not op.active:
                continue
            op_key = f"Op{oid}"
            sid = op.stage_id

            if op.machine_id is not None:
                # Assigned: 현재 배정된 기계 → op
                m_key = f"M{op.machine_id}"
                if m_key in G:
                    G.add_edge(m_key, op_key, etype="assigned",
                               color=m_color[op.machine_id])
            elif op.is_ready:
                # Candidate: ready op에 대해서만 (그래프 과밀 방지)
                for mid in sorted(env.instance.machines_by_stage.get(sid, [])):
                    m_data = env.instance.machines[mid]
                    job = env.instance.jobs[op.job_id]
                    if job.is_component:
                        compatible = (job.product_id, job.component_type_idx) in m_data.compatible_component_ops
                    else:
                        compatible = job.product_id in m_data.compatible_final_ops
                    if compatible:
                        m_key = f"M{mid}"
                        if m_key in G:
                            G.add_edge(m_key, op_key, etype="candidate",
                                       color=m_color[mid])

    return G, pos, active_jobs, m_color


# ─────────────────────────────────────────────────────────
# 노드 그리기 (태림 draw_nodes 패턴)
# ─────────────────────────────────────────────────────────

def _draw_nodes(G: nx.DiGraph, pos: dict, ax):
    # ── 기계 노드 (유휴 / 처리 중 구분) ──
    for is_busy in (False, True):
        nlist = [n for n, d in G.nodes(data=True)
                 if d.get("ntype") == "machine" and d.get("is_busy") == is_busy]
        if not nlist:
            continue
        fill, edge_c = MACHINE_BUSY_STYLE if is_busy else MACHINE_IDLE_STYLE
        nx.draw_networkx_nodes(
            G, pos, nodelist=nlist,
            node_shape=NODE_SHAPE["machine"],
            node_color=fill, edgecolors=edge_c,
            linewidths=2.0, node_size=NODE_SIZE["machine"], ax=ax,
        )

    # ── Op 노드 (상태별) ──
    for status, (fill, edge_c) in OP_STYLE.items():
        nlist = [n for n, d in G.nodes(data=True)
                 if d.get("ntype") == "operation" and d.get("status") == status]
        if not nlist:
            continue
        lw = 3.0 if status == "processing" else 1.5
        nx.draw_networkx_nodes(
            G, pos, nodelist=nlist,
            node_shape=NODE_SHAPE["operation"],
            node_color=fill, edgecolors=edge_c,
            linewidths=lw, node_size=NODE_SIZE["operation"], ax=ax,
        )

    # ── 버퍼 노드 ──
    buf_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "buffer"]
    if buf_nodes:
        fill, edge_c = BUFFER_STYLE
        nx.draw_networkx_nodes(
            G, pos, nodelist=buf_nodes,
            node_shape=NODE_SHAPE["buffer"],
            node_color=fill, edgecolors=edge_c,
            linewidths=1.5, node_size=NODE_SIZE["buffer"], ax=ax,
        )


# ─────────────────────────────────────────────────────────
# 엣지 그리기
# ─────────────────────────────────────────────────────────

def _draw_edges(G: nx.DiGraph, pos: dict, ax):
    # ── Precedence: 검정 화살표 ──
    prec = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "precedence"]
    if prec:
        nx.draw_networkx_edges(
            G, pos, edgelist=prec,
            edge_color=EDGE_PREC_COLOR, width=0.7,
            arrows=True, arrowstyle="-|>", arrowsize=8,
            connectionstyle="arc3,rad=0.0", ax=ax,
        )

    # ── Candidate: 점선, 기계별 색, 반투명 ──
    cand = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("etype") == "candidate"]
    for u, v, d in cand:
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=d["color"], width=0.8, alpha=0.45,
            style="dashed", arrows=False, ax=ax,
        )

    # ── Assigned: 실선 굵게, 기계별 색 ──
    asgn = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("etype") == "assigned"]
    for u, v, d in asgn:
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=d["color"], width=2.0,
            arrows=True, arrowstyle="-|>", arrowsize=10,
            ax=ax,
        )


# ─────────────────────────────────────────────────────────
# 라벨 그리기 (태림 draw_labels_and_markers 패턴)
# ─────────────────────────────────────────────────────────

def _draw_labels(G: nx.DiGraph, pos: dict, env,
                 active_jobs: list, kfont: Optional[FontProperties], ax):
    # ── 기계 라벨 ──
    for n, d in G.nodes(data=True):
        if d.get("ntype") != "machine":
            continue
        x, y = pos[n]
        ax.text(x, y + LABEL_OFFSET_Y, n,
                fontproperties=kfont, ha="center", va="top",
                fontsize=8, color="black")

    # ── Op 라벨: J{jid} / S{sid} ──
    for n, d in G.nodes(data=True):
        if d.get("ntype") != "operation":
            continue
        x, y = pos[n]
        jid = d["job_id"]
        sid = d["stage_id"]
        ax.text(x, y + LABEL_OFFSET_Y, f"J{jid}\nS{sid}",
                fontproperties=kfont, ha="center", va="top",
                fontsize=7, color="black")

    # ── 버퍼 라벨: Buf{sid} (점유) ──
    for n, d in G.nodes(data=True):
        if d.get("ntype") != "buffer":
            continue
        x, y = pos[n]
        sid = d["stage_id"]
        occ = d.get("occupancy", 0)
        ax.text(x, y + LABEL_OFFSET_Y, f"Buf{sid}\n({occ})",
                fontproperties=kfont, ha="center", va="top",
                fontsize=7, color="#2e8b57")

    # ── Stage 제목 (태림의 라인 타이틀에 해당) ──
    for sid in range(env.instance.num_stages):
        x = sid * X_GAP
        ax.text(x, Y_MACHINE_TOP + 1.0, f"Stage {sid}",
                fontproperties=kfont, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#0b1f8a")

    # ── Job 행 라벨 (왼쪽 여백) ──
    for i, jid in enumerate(active_jobs):
        y = Y_OP_BASE + i * Y_OP_GAP
        pid = env.instance.jobs[jid].product_id
        ax.text(-2.0, y, f"J{jid}  (P{pid})",
                fontproperties=kfont, ha="right", va="center",
                fontsize=8, color="#333333")


# ─────────────────────────────────────────────────────────
# 범례 (태림 mpatches 패턴)
# ─────────────────────────────────────────────────────────

def _draw_legend(m_color: Dict[int, str], kfont: Optional[FontProperties], ax):
    patches = [
        mpatches.Patch(facecolor=OP_STYLE["done"][0],
                       edgecolor=OP_STYLE["done"][1],       label="Op: 완료 ▲"),
        mpatches.Patch(facecolor=OP_STYLE["processing"][0],
                       edgecolor=OP_STYLE["processing"][1], label="Op: 처리 중 ▲"),
        mpatches.Patch(facecolor=OP_STYLE["ready"][0],
                       edgecolor=OP_STYLE["ready"][1],      label="Op: 배정 대기 ▲"),
        mpatches.Patch(facecolor=OP_STYLE["waiting"][0],
                       edgecolor=OP_STYLE["waiting"][1],    label="Op: 버퍼 대기 ▲"),
        mpatches.Patch(facecolor=MACHINE_IDLE_STYLE[0],
                       edgecolor=MACHINE_IDLE_STYLE[1],     label="기계: 유휴 ■"),
        mpatches.Patch(facecolor=MACHINE_BUSY_STYLE[0],
                       edgecolor=MACHINE_BUSY_STYLE[1],     label="기계: 처리 중 ■"),
        mpatches.Patch(facecolor=BUFFER_STYLE[0],
                       edgecolor=BUFFER_STYLE[1],           label="버퍼 ●"),
    ]
    for mid, color in sorted(m_color.items()):
        patches.append(mpatches.Patch(color=color, label=f"M{mid} 경로"))

    legend_kw = {"prop": kfont} if kfont else {"fontsize": 8}
    ax.legend(handles=patches, loc="lower right", ncol=2, **legend_kw)


# ─────────────────────────────────────────────────────────
# 통합 시각화 함수
# ─────────────────────────────────────────────────────────

def draw_hetero_graph(env, ep: int,
                      ax=None, title: str = "") -> plt.Figure:
    """
    FFSA 이종 그래프 시각화 (태림 스타일).

    plt.ion() 모드: ax를 넘기면 해당 ax를 지우고 갱신 (Figure 재사용).
    TensorBoard  : ax=None → 새 Figure 생성 후 반환.
    """
    kfont = _init_font()
    G, pos, active_jobs, m_color = _build_graph_and_pos(env)

    if ax is None:
        import matplotlib
        n_jobs   = max(len(active_jobs), 1)
        n_stages = env.instance.num_stages
        fig_w = max(14, n_stages * X_GAP + 4)
        fig_h = max(8,  n_jobs  * abs(Y_OP_GAP) + 5)
        # GUI 팝업 없는 non-interactive Figure (TensorBoard / PNG 저장용)
        fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h))
        ax  = fig.add_subplot(111)
    else:
        ax.clear()
        fig = ax.figure

    _draw_edges(G, pos, ax)
    _draw_nodes(G, pos, ax)
    _draw_labels(G, pos, env, active_jobs, kfont, ax)
    _draw_legend(m_color, kfont, ax)

    wt  = env.get_actual_weighted_tardiness()
    ms  = env.get_makespan()
    t   = env.current_time
    ttl = title or f"[Ep {ep}]  t={t:.1f}  WT={wt:.1f}  MS={ms:.1f}"
    ax.set_title(ttl, fontsize=12, fontproperties=kfont)
    ax.axis("off")

    all_x = [p[0] for p in pos.values()] or [0]
    all_y = [p[1] for p in pos.values()] or [0]
    ax.set_xlim(min(all_x) - 2.8, max(all_x) + 2.8)
    ax.set_ylim(min(all_y) - 1.5, max(all_y) + 2.2)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
# TensorBoard 로깅
# ─────────────────────────────────────────────────────────

def log_hetero_graph_to_tensorboard(writer, env, ep: int):
    """TensorBoard Images 탭에 이종 그래프 기록."""
    fig = draw_hetero_graph(env, ep)
    writer.add_figure("graph/hetero_state", fig, global_step=ep)
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# 간트 차트
# ─────────────────────────────────────────────────────────

def draw_gantt(env, title: str = "Schedule", save_path: str = None):
    """
    에피소드 종료 후 env에서 스케줄을 읽어 간트 차트를 그린다.

    Parameters
    ----------
    env       : 에피소드가 끝난 FFSASchedulingEnv
    title     : 차트 제목
    save_path : 지정하면 PNG 저장, None이면 plt.show()
    """
    kfont = _init_font()

    # ── 기계 목록 및 색상 ──
    num_machines = env.instance.num_machines
    product_ids  = sorted(env.instance.products.keys())
    cmap         = plt.get_cmap("tab10")
    prod_color   = {p: cmap(i % 10) for i, p in enumerate(product_ids)}

    fig, ax = plt.subplots(figsize=(14, max(4, num_machines * 0.55)))

    # ── 각 operation 막대 그리기 ──
    for op in env.operations.values():
        if not op.is_done:
            continue
        if op.start_time is None or op.completion_time is None:
            continue
        if op.machine_id is None:
            continue

        job       = env.instance.jobs[op.job_id]
        order     = env.instance.orders[job.order_id]
        color     = prod_color[job.product_id]
        duration  = op.completion_time - op.start_time
        y         = op.machine_id

        ax.barh(
            y, duration, left=op.start_time,
            color=color, edgecolor="white", linewidth=0.5, alpha=0.85,
        )
        # 납기 초과 여부 표시 (빨간 테두리)
        if op.is_assembly or job.is_final_job:
            if op.completion_time > order.due_date + 1e-6:
                ax.barh(
                    y, duration, left=op.start_time,
                    color="none", edgecolor="red", linewidth=1.5,
                )

    # ── 납기 수직선 ──
    for order in env.instance.orders.values():
        ax.axvline(order.due_date, color="red", linewidth=0.7,
                   linestyle="--", alpha=0.5)

    # ── 기계 y축 레이블 ──
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels(
        [f"M{mid}  (Stage {env.instance.machines[mid].stage_id})"
         for mid in range(num_machines)],
        fontproperties=kfont, fontsize=8,
    )
    ax.invert_yaxis()

    # ── 범례 ──
    patches = [
        mpatches.Patch(color=prod_color[p], label=f"Product {p}")
        for p in product_ids
    ]
    patches.append(mpatches.Patch(color="none", edgecolor="red",
                                  linewidth=1.5, label="납기 초과"))
    ax.legend(handles=patches, loc="upper right",
              prop=kfont if kfont else {}, fontsize=8)

    # ── 통계 출력 ──
    wt = env.get_actual_weighted_tardiness()
    ms = env.get_makespan()
    ax.set_xlabel(f"Time   (Makespan={ms:.1f},  Weighted Tardiness={wt:.1f})",
                  fontproperties=kfont, fontsize=9)
    ax.set_title(title, fontproperties=kfont, fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"간트 차트 저장: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────
# 텍스트 스케줄 리포트
# ─────────────────────────────────────────────────────────

def print_schedule_report(env, title: str = "Schedule Report",
                          save_path: str = None):
    """
    에피소드 종료 후 env에서 스케줄 정보를 텍스트 파일로 저장한다.

    Parameters
    ----------
    env       : 에피소드가 끝난 FFSASchedulingEnv
    title     : 리포트 제목
    save_path : 저장할 파일 경로. None이면 stdout으로 출력.
    """
    import io, os, sys

    inst = env.instance
    SEP  = "=" * 70
    SEP2 = "-" * 70
    lines = []
    w = lines.append   # 한 줄씩 추가하는 단축 함수

    w(f"\n{SEP}")
    w(f"  {title}")
    w(SEP)

    # ── 1. 인스턴스 구성 요약 ──
    w("\n[인스턴스 구성]")
    w(f"  제품 수       : {inst.num_products}")
    w(f"  스테이지 수   : {inst.num_stages}")
    w(f"  기계 수       : {inst.num_machines}")
    w(f"  주문 수       : {len(inst.orders)}")
    w(f"  잡 수         : {len(inst.jobs)}")

    stage_machine_map: dict = {}
    for m in inst.machines.values():
        stage_machine_map.setdefault(m.stage_id, []).append(m.machine_id)
    w(f"\n  스테이지별 기계:")
    for sid in sorted(stage_machine_map):
        mids = sorted(stage_machine_map[sid])
        w(f"    Stage {sid}: M{mids}")

    # ── 2. 주문 구성 ──
    w(f"\n{SEP2}")
    w("[주문 구성]")
    w(f"  {'주문ID':>5}  {'제품ID':>5}  {'수량':>4}  {'납기':>8}  {'가중치':>6}")
    w(f"  {'-'*5}  {'-'*5}  {'-'*4}  {'-'*8}  {'-'*6}")
    for order in sorted(inst.orders.values(), key=lambda o: o.order_id):
        prod   = inst.products[order.product_id]
        weight = prod.weight
        w(f"  {order.order_id:>5}  {order.product_id:>5}  {order.quantity:>4}  "
          f"{order.due_date:>8.1f}  {weight:>6.2f}")

    # ── 3. 잡별 처리 상세 ──
    w(f"\n{SEP2}")
    w("[잡별 처리 상세]")

    for order in sorted(inst.orders.values(), key=lambda o: o.order_id):
        w(f"\n  ▶ 주문 {order.order_id}  (제품 {order.product_id}, "
          f"납기={order.due_date:.1f}, 수량={order.quantity})")

        all_job_ids = list(order.component_job_ids) + list(order.final_job_ids)
        for job_id in all_job_ids:
            job  = inst.jobs[job_id]
            kind = "컴포넌트" if job.is_component else "파이널  "
            w(f"\n    잡 {job_id:>3}  [{kind}]  "
              f"route={list(job.route)}  assembly_stage={job.assembly_stage}")

            op_ids = env.job_ops.get(job_id, [])
            if not op_ids:
                w("      (운영 없음)")
                continue

            w(f"      {'스테이지':>5}  {'기계':>4}  {'시작':>8}  {'종료':>8}  "
              f"{'처리시간':>6}  {'비고'}")
            w(f"      {'-'*5}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*10}")
            for op_id in op_ids:
                op  = env.operations[op_id]
                st  = f"{op.start_time:.1f}"      if op.start_time      is not None else "-"
                ct  = f"{op.completion_time:.1f}" if op.completion_time is not None else "-"
                mid = f"M{op.machine_id}"         if op.machine_id      is not None else "-"
                dur = (f"{op.completion_time - op.start_time:.1f}"
                       if (op.start_time is not None and op.completion_time is not None)
                       else "-")
                note = "조립op" if op.is_assembly else ("미완료" if not op.is_done else "")
                w(f"      {op.stage_id:>5}  {mid:>4}  {st:>8}  {ct:>8}  {dur:>6}  {note}")

    # ── 4. 기계별 스케줄 테이블 ──
    w(f"\n{SEP2}")
    w("[기계별 스케줄 테이블]")

    done_ops = [
        op for op in env.operations.values()
        if op.is_done and op.machine_id is not None
        and op.start_time is not None and op.completion_time is not None
    ]
    done_ops.sort(key=lambda o: (o.machine_id, o.start_time))

    cur_machine = None
    hdr = (f"  {'기계':>4}  {'스테이지':>5}  {'잡ID':>5}  {'주문':>5}  "
           f"{'제품':>4}  {'종류':>6}  {'시작':>8}  {'종료':>8}  {'처리시간':>6}  {'비고'}")
    div = (f"  {'-'*4}  {'-'*5}  {'-'*5}  {'-'*5}  "
           f"{'-'*4}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}")

    for op in done_ops:
        if op.machine_id != cur_machine:
            cur_machine = op.machine_id
            w(f"\n{hdr}")
            w(div)

        job   = inst.jobs[op.job_id]
        order = inst.orders[job.order_id]
        kind  = "파이널" if job.is_final_job else "컴포넌트"
        dur   = op.completion_time - op.start_time
        note  = "조립" if op.is_assembly else ""
        if (op.is_assembly or job.is_final_job) and op.completion_time > order.due_date + 1e-6:
            note += "★지연"
        w(f"  M{op.machine_id:<3}  {op.stage_id:>5}  {op.job_id:>5}  "
          f"{job.order_id:>5}  {job.product_id:>4}  {kind:>6}  "
          f"{op.start_time:>8.1f}  {op.completion_time:>8.1f}  {dur:>6.1f}  {note}")

    # ── 5. 기계 가동률 ──
    w(f"\n{SEP2}")
    w("[기계 가동률]")
    makespan = env.get_makespan()
    ms_denom = makespan if makespan > 0 else 1.0

    stage_busy: dict = {}
    stage_count: dict = {}

    w(f"  {'기계':>4}  {'스테이지':>5}  {'처리시간 합':>9}  {'가동률':>6}")
    w(f"  {'-'*4}  {'-'*5}  {'-'*9}  {'-'*6}")
    for mid in sorted(inst.machines.keys()):
        m_ops  = [op for op in done_ops if op.machine_id == mid]
        busy   = sum(op.completion_time - op.start_time for op in m_ops)
        util   = busy / ms_denom * 100
        stage  = inst.machines[mid].stage_id
        stage_busy[stage]  = stage_busy.get(stage, 0.0) + busy
        stage_count[stage] = stage_count.get(stage, 0) + 1
        w(f"  M{mid:<3}  {stage:>5}  {busy:>9.1f}  {util:>5.1f}%")

    w(f"\n  [스테이지 평균 가동률]")
    w(f"  {'스테이지':>5}  {'기계 수':>5}  {'평균 가동률':>8}")
    w(f"  {'-'*5}  {'-'*5}  {'-'*8}")
    for sid in sorted(stage_busy.keys()):
        n        = stage_count[sid]
        avg_util = (stage_busy[sid] / n) / ms_denom * 100
        w(f"  {sid:>5}  {n:>5}  {avg_util:>7.1f}%")

    # ── 6. 주문별 납기 요약 ──
    w(f"\n{SEP2}")
    w("[납기 준수 요약]")
    w(f"  {'주문ID':>5}  {'납기':>8}  {'최종완료':>8}  {'지연':>8}  {'가중 지연':>9}")
    w(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")

    total_wt = 0.0
    for order in sorted(inst.orders.values(), key=lambda o: o.order_id):
        prod   = inst.products[order.product_id]
        weight = prod.weight

        completion_times = []
        for job_id in list(order.component_job_ids) + list(order.final_job_ids):
            for op_id in env.job_ops.get(job_id, []):
                op = env.operations[op_id]
                if op.is_done and op.completion_time is not None:
                    completion_times.append(op.completion_time)

        if not completion_times:
            w(f"  {order.order_id:>5}  {order.due_date:>8.1f}  {'미완':>8}  {'-':>8}  {'?':>9}")
            continue

        comp_time = max(completion_times)
        tardiness = max(0.0, comp_time - order.due_date)
        wt        = weight * tardiness
        total_wt += wt
        flag      = "  <<지연>>" if tardiness > 0 else ""
        w(f"  {order.order_id:>5}  {order.due_date:>8.1f}  {comp_time:>8.1f}  "
          f"{tardiness:>8.1f}  {wt:>9.2f}{flag}")

    w(f"\n  합계 가중 지연 (WT)  : {total_wt:.4f}")
    wt_actual = env.get_actual_weighted_tardiness()
    w(f"  env.get_actual_WT()  : {wt_actual:.4f}")
    w(f"  메이크스팬           : {makespan:.1f}")
    w(SEP)

    # ── 출력 또는 파일 저장 ──
    text = "\n".join(lines) + "\n"
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"스케줄 리포트 저장: {save_path}")
    else:
        sys.stdout.write(text)
