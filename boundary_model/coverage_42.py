"""42 项策略 × 图的覆盖核对：每项策略落在哪张图、哪条线上，现有图里画了没有。

数据来源：《42种策略_通用边界模型.md》第 2 节“所在图轴”一列，
以及 demo_fig6b.py（fig6b_demo.png）与 luong2024_fig.py（luong2024_dV.png）实际画出的线。
运行：python3 coverage_42.py → coverage_42.png，并打印 Markdown 表与计数。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]

COLS = ["d–V 图\n(Fig. 6b)", "λ 轴\n(Fig. 7)", "材料参数\n(移动 d–V 线)", "P / f 轴\n(Fig. 9–10)"]
COLS_MD = ["d–V 图 (Fig. 6b)", "λ 轴 (Fig. 7)", "材料参数", "P / f 轴"]

# 状态：drawn 已画；partial 线已画但本策略的作用没体现（只是共用/移动该线，或只在一张图里）；
# missing 应在 d–V 图上却没画；todo 属于尚未生成的图；scope 范围线
STATUS = {
    "drawn":   ("已画", "#2a78d6"),
    "partial": ("部分", "#1baf7a"),
    "missing": ("缺（d–V 图上应有）", "#eb6834"),
    "todo":    ("待画（后续图）", "#d9d8d3"),
    "scope":   ("范围线", "#ffffff"),
}

# (编号, 名称, {列号: (状态, 线/式号)})
S = [
    ("3.1-1", "材料与组分", {0: ("partial", "移 A5"), 1: ("todo", "A1/A2, N3")}),
    ("3.1-2", "聚集与取向", {0: ("partial", "移 A5"), 1: ("todo", "A8")}),
    ("3.1-3", "活性层厚度", {0: ("drawn", "A5")}),
    ("3.1-4", "减反", {0: ("partial", "移 A5"), 1: ("todo", "A1 (R_f)")}),
    ("3.1-5", "寄生吸收", {0: ("partial", "移 A5/B5"), 3: ("todo", "B2, K10")}),
    ("3.1-6", "微腔", {0: ("drawn", "A3/A4"), 1: ("todo", "Δλ, θ")}),
    ("3.1-7", "光陷阱/导模", {0: ("partial", "移 A5"), 1: ("todo", "A7")}),
    ("3.1-8", "等离激元", {1: ("todo", "ΔA")}),
    ("3.1-9", "发光反馈", {0: ("missing", "F1 V 上限")}),
    ("3.1-10", "中间态吸收", {1: ("todo", "α_sub"), 3: ("todo", "f_trap")}),
    ("3.1-11", "内光电发射", {1: ("todo", "N5 λ_c")}),
    ("3.1-12", "光谱转换", {1: ("todo", "F2"), 3: ("todo", "τ_em, P_th")}),
    ("3.1-13", "叠层", {1: ("todo", "F3")}),
    ("3.1-14", "杂化吸光层", {1: ("scope", "范围线")}),
    ("A1", "相尺寸", {2: ("todo", "X2 → K3, B4")}),
    ("A2", "扩散长度", {2: ("todo", "X1/X3")}),
    ("A3", "能量转移", {2: ("todo", "X4")}),
    ("A4", "PHJ/低给体比例", {0: ("missing", "X1 最优 d_X")}),
    ("B1", "界面能级差", {2: ("todo", "X5")}),
    ("B2", "界面耦合", {2: ("todo", "|V|² 最优")}),
    ("B3", "非生产性猝灭", {2: ("todo", "k_q·τ")}),
    ("B4", "激子阻挡层", {2: ("todo", "X1 (s)"), 3: ("todo", "R_EBL")}),
    ("C1", "能量景观", {2: ("todo", "ΔG_casc")}),
    ("C2", "CT 离域", {2: ("todo", "C2 r_c")}),
    ("C3", "介电常数", {2: ("todo", "C1 ε_r(ω)")}),
    ("C4", "竞争通道", {3: ("todo", "γn²(V,P)")}),
    ("C5", "外场（产生端）", {0: ("missing", "V = F_sat·d − V₀")}),
    ("C6", "直接 CT 激发", {1: ("todo", "C4 α_CT")}),
    ("C7", "体内产生", {0: ("missing", "V_½")}),
    ("D1", "迁移率", {0: ("drawn", "K3/K5′/B4")}),
    ("D2", "双分子复合", {0: ("partial", "K5′ 仅 demo")}),
    ("D3", "体陷阱", {0: ("partial", "移 N2"), 3: ("todo", "f_trap")}),
    ("D4", "收集长度匹配", {0: ("drawn", "CCN 区 (K4)")}),
    ("D5", "垂直相分离", {2: ("todo", "φ_w")}),
    ("D6", "PPHJ/LbL", {2: ("todo", "X1 + 渗流")}),
    ("D7", "抽取势垒", {3: ("todo", "K9")}),
    ("D8", "接触选择性", {0: ("drawn", "N2")}),
    ("D9", "界面陷阱", {3: ("todo", "K12/K13")}),
    ("D10", "串联电阻", {0: ("partial", "B5 demo 出界"), 3: ("todo", "f_RC")}),
    ("D11", "反偏（收集端）", {0: ("drawn", "N4 V_n")}),
    ("D12", "内场调控", {0: ("missing", "K3 向 V<0 平移")}),
    ("D13", "电荷产生层", {3: ("scope", "增益判别")}),
]
assert len(S) == 42
GROUPS = [(0, "3.1 光吸收 14"), (14, "3.2 激子 8"), (22, "3.3 CT 解离 7"), (29, "3.4 收集 13")]


def best_status(cells):
    order = ["drawn", "partial", "missing", "todo", "scope"]
    return min((st for st, _ in cells.values()), key=order.index)


# ---------------------------------------------------------------- 图
cw, rh, x0 = 1.55, 0.32, 2.5
fig, ax = plt.subplots(figsize=(9.2, 15.5))
for i, (sid, name, cells) in enumerate(S):
    y = -i * rh
    ax.text(0.05, y + rh / 2, sid, va="center", fontsize=8, color="#0b0b0b")
    ax.text(0.75, y + rh / 2, name, va="center", fontsize=8, color="#52514e")
    for j in range(len(COLS)):
        x = x0 + j * cw
        if j in cells:
            st, lab = cells[j]
            col = STATUS[st][1]
            ax.add_patch(Rectangle((x + 0.02, y + 0.02), cw - 0.04, rh - 0.04, facecolor=col,
                                   edgecolor="#9b9a95" if st == "scope" else "white",
                                   hatch="///" if st == "scope" else None, lw=1))
            ink = "white" if st in ("drawn", "missing") else "#0b0b0b"
            ax.text(x + cw / 2, y + rh / 2, lab, ha="center", va="center", fontsize=7, color=ink)
        else:
            ax.add_patch(Rectangle((x + 0.02, y + 0.02), cw - 0.04, rh - 0.04, facecolor="#f4f3f0", lw=0))
for k, (i0, lab) in enumerate(GROUPS):
    y = -i0 * rh + rh
    if k:
        ax.plot([0, x0 + len(COLS) * cw], [y, y], color="#52514e", lw=1)
    ax.text(x0 + len(COLS) * cw + 0.08, y - rh / 2, lab, va="center", fontsize=8, color="#52514e")
for j, c in enumerate(COLS):
    ax.text(x0 + j * cw + cw / 2, rh * 1.25, c, ha="center", va="bottom", fontsize=8.5, color="#0b0b0b")
ax.set_xlim(0, x0 + len(COLS) * cw + 1.4)
ax.set_ylim(-len(S) * rh - 0.9, rh * 3.2)
ax.axis("off")

# 图例（带文字，颜色不是唯一标识）
counts = {k: sum(best_status(c) == k for _, _, c in S) for k in STATUS}
yl = -len(S) * rh - 0.35
for j, k in enumerate(STATUS):
    lab, col = STATUS[k]
    x = 0.05 + j * 2.05
    ax.add_patch(Rectangle((x, yl), 0.28, 0.2, facecolor=col, edgecolor="#9b9a95" if k == "scope" else "white",
                           hatch="///" if k == "scope" else None))
    ax.text(x + 0.36, yl + 0.1, f"{lab}  {counts[k]} 项", va="center", fontsize=8, color="#0b0b0b")
ax.text(0.05, yl - 0.3, "计数按每项策略的最好状态（已画 > 部分 > 缺 > 待画 > 范围线）；格内为对应边界线或式号",
        fontsize=7.5, color="#52514e")
ax.set_title("42 项策略在边界图中的覆盖情况", fontsize=11, color="#0b0b0b", loc="left")
fig.tight_layout()
fig.savefig("coverage_42.png", dpi=160)

# ---------------------------------------------------------------- 表与计数
print("| 编号 | 策略 | " + " | ".join(COLS_MD) + " |")
print("|---|---|" + "---|" * len(COLS_MD))
for sid, name, cells in S:
    row = [f"{STATUS[cells[j][0]][0]}：{cells[j][1]}" if j in cells else "" for j in range(len(COLS))]
    print(f"| {sid} | {name} | " + " | ".join(row) + " |")
print()
print("按最好状态计数：", {STATUS[k][0]: v for k, v in counts.items()})
