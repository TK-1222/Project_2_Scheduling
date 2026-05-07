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
                    if op.product_id in m_data.compatible_products:
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
