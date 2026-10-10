"""d–V 图以外的三张边界图（示意参数）：λ 轴、材料参数、光强/频率轴。

每个小图对应一项策略：蓝线为收益函数，黑色点线为饱和边界，橙色阴影为失效区。
参数全部是示意值，只用于检查每项策略的边界函数能否画出来，不代表任何文献体系。
运行：python3 axis_figures.py → lambda_axis.png、material_axis.png、pf_axis.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import opd_bounds as b

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": ["DejaVu Sans", "WenQuanYi Zen Hei"], "font.size": 8,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "axes.titlesize": 8.5, "axes.unicode_minus": False, "mathtext.fontset": "dejavusans", "axes.titlelocation": "left"})
KT = b.kT()


def sat(ax, x, label="sat", y=0.95):
    """饱和边界：黑色点线。"""
    ax.axvline(x, color=INK, ls=":", lw=1.4)
    ax.text(x, y, f" {label}", transform=ax.get_xaxis_transform(), fontsize=7, color=INK, va="top")


def fail(ax, x0, x1, label="fail", y=0.05):
    """失效区：橙色阴影。"""
    ax.axvspan(x0, x1, color=ORANGE, alpha=0.15, lw=0)
    ax.text((x0 * x1) ** 0.5 if ax.get_xscale() == "log" else (x0 + x1) / 2, y, label,
            transform=ax.get_xaxis_transform(), fontsize=7, color=ORANGE, ha="center")


def legend(ax, loc="best"):
    ax.legend(fontsize=6.5, frameon=False, loc=loc)


# ====================================================================== λ 轴
def lambda_axis():
    fig, axs = plt.subplots(3, 4, figsize=(13, 9))
    ax = axs.ravel()
    lam_t = 1100e-9

    # 3.1-1 材料与组分：E_CT 越低吸收越远，但 J0 ∝ exp(−E_CT/kT) 抬高（互易 N3）
    a = ax[0]
    E = np.linspace(0.7, 1.3, 300)
    Dn = np.exp((E - 1.3) / (2 * KT))           # D*_shot ∝ 1/√J0，按 1.3 eV 归一
    a.plot(E, Dn, color=BLUE, label="D*_shot / D*(1.3 eV)")
    E_edge = b.h * b.c / lam_t / b.q            # 吸收到 λ* 需要 E_opt ≤ hc/λ*
    E_min = 1.3 + 2 * KT * np.log(1e-3)         # D*_min = 1e-3 × 参考值
    fail(a, E_edge - 0.1, 1.3, "λ* not\nabsorbed", y=0.6)
    fail(a, 0.7, E_min, "D* < D*_min", y=0.6)
    sat(a, E_edge - 0.1, "η_A(λ*)≥A*", y=0.98)
    a.text(0.42, 0.04, "E_opt ≈ E_CT + 0.1 eV ≤ hc/λ*", fontsize=6.5, color=MUTED, transform=a.transAxes)
    a.set(yscale="log", xlabel="E_CT (eV)", ylabel="relative", title="3.1-1 材料与组分 (A1/A2, N3)")
    legend(a, "upper left")

    # 3.1-2 取向：吸收因子 (3/2)<sin²θ>
    a = ax[1]
    s2 = np.linspace(0, 1, 200)
    a.plot(s2, b.orientation_factor(s2), color=BLUE, label="α_eff/α_iso (Eq.A8)")
    a.axvline(2 / 3, color=MUTED, lw=1, ls="--")
    a.text(2 / 3, 0.2, " isotropic", fontsize=7, color=MUTED)
    sat(a, 1.0, "max 1.5", y=0.6)
    a.set(xlabel="⟨sin²θ_dip⟩", ylabel="absorption factor", title="3.1-2 聚集与取向 (A8)")
    a.text(0.02, 1.35, "fail: ΔE(θ) shifts η_CT, J0 (S3)\n— needs measured ΔE", fontsize=6.5, color=ORANGE)
    legend(a, "lower right")

    # 3.1-4 减反：外表面 vs 腔前镜
    a = ax[2]
    Rf = np.linspace(0.0, 0.3, 200)
    alpha_w = b.alpha_from_k(0.02, lam_t)
    a.plot(Rf, b.eta_A_incoherent(alpha_w, 200e-9, Rf, 0.9), color=BLUE, label="outer AR (A1)")
    dres = b.d_resonance(1, lam_t, 1.9)
    a.plot(Rf, b.eta_A_cavity(alpha_w, dres, lam_t, 1.9, np.maximum(Rf, 1e-6), 0.9), color=VIOLET,
           label="cavity front mirror (A2)")
    sat(a, b.fresnel_R(1.0, 1.52), "air/glass 4.2%", y=0.45)
    a.set(xlabel="front reflectance R_f", ylabel="η_A at λ*", title="3.1-4 减反 (A1, A9)")
    a.text(0.35, 0.12, "fail: on a cavity mirror,\nlower R_f lowers η_A", fontsize=6.5, color=ORANGE, transform=a.transAxes)
    legend(a, "upper left")

    # 3.1-6 微腔：谱宽与厚度容差随 r 变窄
    a = ax[3]
    r = np.linspace(0.2, 0.97, 200)
    d6 = b.d_resonance(2, lam_t, 1.9)
    a.plot(r, b.lambda_fwhm_cavity(lam_t, 1.9, d6, r) * 1e9, color=BLUE, label="Δλ_FWHM (Eq.P1)")
    a.plot(r, lam_t * (1 - r) / (2 * np.pi * 1.9 * np.sqrt(r)) * 1e9, color=VIOLET, label="Δd_FWHM (A4)")
    a.axhline(10, color=ORANGE, lw=1, ls="--")
    r_fail = r[np.argmax(lam_t * (1 - r) / (2 * np.pi * 1.9 * np.sqrt(r)) < 3 * 10e-9 / 3)]
    fail(a, r_fail, 0.97, "")
    a.text(0.88, 0.55, "Δd < ±10 nm\ntolerance", fontsize=6.5, color=ORANGE, ha="right", transform=a.get_xaxis_transform())
    a.set(yscale="log", xlabel="mirror product r = √(R_f R_b)e^{−αd}", ylabel="width (nm)",
          title="3.1-6 微腔 (A2–A4, P1, P2)")
    th = np.deg2rad(30)
    a.text(0.22, 3, f"30° shifts λ_res by {(lam_t - b.lambda_res_angle(lam_t, 1.9, th))*1e9:.0f} nm (P2)",
           fontsize=6.5, color=MUTED)
    legend(a, "upper right")

    # 3.1-7 光陷阱：A = αdF/(αdF+1)
    a = ax[4]
    ad = np.geomspace(1e-3, 10, 300)
    Fmax = b.path_enhancement_limit(1.8)
    a.plot(ad, 1 - np.exp(-ad), color=MUTED, lw=1.4, label="single pass")
    a.plot(ad, b.eta_A_trapping(ad, 1, Fmax), color=BLUE, label=f"F = 4n² = {Fmax:.0f} (A6, A7)")
    sat(a, 1 / Fmax, "αd·F = 1")
    a.set(xscale="log", xlabel="αd", ylabel="absorption", title="3.1-7 光陷阱/导模 (A6, A7)")
    a.text(1.5e-3, 0.5, "fail: roughness ρ_r\nraises J_d ∝ ρ_r", fontsize=6.5, color=ORANGE)
    legend(a, "upper left")

    # 3.1-8 等离激元：壳厚与猝灭
    a = ax[5]
    t = np.linspace(0.5e-9, 20e-9, 300)
    k0, z0, tauX = 1e12, 1e-9, 1e-9
    a.plot(t * 1e9, 1 / (1 + k0 * (z0 / t) ** 6 * tauX), color=BLUE, label="exciton survival 1/(1+k_q τ_X)")
    ts = b.shell_min_plasmon(z0, k0, tauX, 0.1)
    fail(a, 0.5, ts * 1e9, "quenched")
    sat(a, ts * 1e9, f"t_s,min={ts*1e9:.1f} nm (P3)", y=0.6)
    a.set(xlabel="shell thickness t_s (nm)", ylabel="survival", title="3.1-8 等离激元 (P3)")
    legend(a, "center right")

    # 3.1-10 中间态吸收：EQE_sub ∝ N_t，但陷阱暗电流也 ∝ N_t
    a = ax[6]
    Nt = np.geomspace(1e20, 1e25, 300)
    eqe = b.eqe_subgap(1e-20, Nt, 300e-9, 0.3)
    a.plot(Nt, eqe, color=BLUE, label="EQE_sub (Eq.P4)")
    a.axhline(1e-4, color=MUTED, lw=1, ls="--")
    a.text(1.2e20, 1.3e-4, "target EQE_sub", fontsize=6.5, color=MUTED)
    Jt = b.J_gen_trap(300e-9, Nt, 0.85)
    N_fail = Nt[np.argmax(Jt > 1e-4)]
    fail(a, N_fail, 1e25, "J_gen,trap > J_d*\n(10 nA/cm², D3)", y=0.15)
    sat(a, Nt[np.argmax(eqe >= 1e-4)], "reach target", y=0.3)
    a.set(xscale="log", yscale="log", xlabel="mid-gap trap density N_t (m⁻³)", ylabel="EQE",
          title="3.1-10 中间态吸收 (P4, P10)")
    legend(a, "upper left")

    # 3.1-11 内光电发射：D*_shot 对势垒的最优
    a = ax[7]
    Eph = b.h * b.c / 1550e-9 / b.q
    phi = np.linspace(0.3, Eph - 0.005, 300)
    D = b.dstar_ipe(Eph, phi)
    a.plot(phi, D / D.max(), color=BLUE, label="D*_shot (Eq.P5)")
    sat(a, Eph - 4 * KT, "Φ_B* = hν − 4kT", y=0.5)
    fail(a, Eph - 0.005, Eph + 0.05, "λ* > λ_c")
    a.set(xlim=(0.3, Eph + 0.05), xlabel="barrier Φ_B (eV)", ylabel="D* / max", yscale="log",
          title="3.1-11 内光电发射 〔待核验〕 (N5, P5)")
    a.text(0.03, 0.03, "λ* = 1550 nm", fontsize=6.5, color=MUTED, transform=a.transAxes)
    legend(a, "upper left")

    # 3.1-12 光谱转换：PLQY 收支平衡点
    a = ax[8]
    plqy = np.linspace(0, 1, 200)
    Ac, cpl, eqe_em, eqe_dir = 0.8, 0.5, 0.6, 0.15
    a.plot(plqy, b.eqe_conversion(Ac, plqy, cpl, eqe_em, eqe_dir), color=BLUE, label="with converter (F2)")
    a.axhline(eqe_dir, color=MUTED, lw=1.2, ls="--", label="direct EQE(λ)")
    be = eqe_dir / (cpl * eqe_em)
    fail(a, 0, be, "net loss")
    sat(a, be, f"break-even PLQY={be:.2f}", y=0.55)
    a.set(xlabel="PLQY", ylabel="EQE at λ*", title="3.1-12 光谱转换 〔待核验〕 (F2)")
    legend(a, "upper left")

    # 3.1-13 叠层：串联取小、并联相加
    a = ax[9]
    eb = np.linspace(0, 0.8, 200)
    Ttop, et = 0.7, 0.35
    a.plot(eb, b.eqe_tandem(et, Ttop, eb, True), color=BLUE, label="series: min(EQE₁, T₁EQE₂)")
    a.plot(eb, b.eqe_tandem(et, Ttop, eb, False), color=VIOLET, label="parallel / 3-terminal")
    sat(a, et / Ttop, "current matched", y=0.75)
    a.set(xlabel="bottom-cell EQE₂", ylabel="EQE", title="3.1-13 叠层 〔待核验〕 (F3)")
    a.text(0.02, 0.95, "fail (series): J_d, noise of both junctions add", fontsize=6.5, color=ORANGE,
           transform=a.transAxes)
    legend(a, "lower right")

    # 3.1-14 杂化吸光层：范围线
    a = ax[10]
    lam = np.linspace(800, 1800, 300)
    Eg_org = 1.15
    E = 1239.84 / lam
    alpha_org = 1e7 * np.where(E > Eg_org, 1.0, np.exp((E - Eg_org) / 0.03))
    a.plot(lam, alpha_org, color=BLUE, label="α_org(λ), Urbach tail")
    a.axhline(1e5, color=MUTED, lw=1, ls="--")
    a.text(810, 1.4e5, "α_min", fontsize=6.5, color=MUTED)
    lam_c = lam[np.argmax(alpha_org < 1e5)]
    a.axvspan(lam_c, 1800, color=MUTED, alpha=0.12, hatch="///", lw=0)
    a.text((lam_c + 1800) / 2, 1e3, "out of scope:\nhybrid absorber", ha="center", fontsize=7, color=MUTED)
    a.set(yscale="log", xlabel="λ (nm)", ylabel="α (m⁻¹)", title="3.1-14 杂化吸光层（范围线）")
    legend(a, "upper right")

    # C6 直接 CT 激发
    a = ax[11]
    E = np.linspace(0.9, 1.8, 400)
    E_CT, lam_r = 1.25, 0.15
    ct = b.ct_absorption(E, 1.0, E_CT, lam_r)
    ct = ct / ct.max() * 1e-2
    le = 1 / (1 + np.exp(-(E - 1.55) / 0.03))
    a.plot(E, le + ct, color=BLUE, label="LE + CT absorption (C4)")
    a.plot(E, ct, color=VIOLET, lw=1.4, label="CT band only")
    a.axhline(1e-3, color=MUTED, lw=1, ls="--")
    a.text(0.92, 1.4e-3, "needs cavity / thick film below this", fontsize=6.5, color=MUTED)
    sat(a, E_CT + lam_r, "CT peak", y=0.5)
    a.set(yscale="log", ylim=(1e-5, 2), xlabel="photon energy (eV)", ylabel="relative α",
          title="C6 直接 CT 激发 (C4)")
    legend(a, "lower right")

    fig.suptitle("λ-axis strategies (illustrative) — blue: benefit, dotted: saturation, orange: failure",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig("lambda_axis.png", dpi=150)


# ====================================================================== 材料参数
def material_axis():
    fig, axs = plt.subplots(3, 4, figsize=(13, 9))
    ax = axs.ravel()

    # A1 相尺寸
    a = ax[0]
    x = np.geomspace(0.05, 10, 300)
    eta = b.eta_sphere(x, 1.0)
    a.plot(x, eta, color=BLUE, label="η_sph(R/L) (X2)")
    el = np.array([b.elasticity(lambda R: b.eta_sphere(R, 1.0), xi) for xi in x])
    sat(a, x[np.argmax(np.abs(el) > 0.1)], "|∂lnη/∂lnR|=0.1")
    fail(a, 0.05, 0.15, "percolation lost\n(μ_s → 0, P9)", y=0.4)
    a.set(xscale="log", xlabel="domain radius R / L_D", ylabel="η_reach", title="A1 相尺寸 (X2, P9)")
    legend(a, "lower left")

    # A2 扩散长度
    a = ax[1]
    x = np.geomspace(0.05, 10, 300)
    a.plot(x, b.sensitivity_L(x, 1.0), color=BLUE, label="∂lnη/∂lnL (X3)")
    a.axhline(0.1, color=MUTED, ls="--", lw=1)
    sat(a, 0.40, "d/L = 0.40", y=0.5)
    a.set(xscale="log", xlabel="d / L_D", ylabel="elasticity", title="A2 扩散长度 (X1, X3)")
    a.text(0.06, 0.6, "fail: raising τ\nmay lower k_CT (B1)", fontsize=6.5, color=ORANGE)
    legend(a, "upper left")

    # A3 能量转移
    a = ax[2]
    rr = np.linspace(0.2, 2.5, 300)
    a.plot(rr, b.fret_efficiency(rr, 1.0), color=BLUE, label="Förster E (X4)")
    rmax = b.r_max_fret(1.0, 0.9)
    sat(a, rmax, f"E*=0.9 → r ≤ {rmax:.2f}R₀ (P7)", y=0.6)
    fail(a, rmax, 2.5, "E < E*", y=0.3)
    a.set(xlabel="r / R₀", ylabel="transfer efficiency", title="A3 能量转移 (X4, P7)")
    legend(a, "upper right")

    # B1 能级差
    a = ax[3]
    dE = np.linspace(0.0, 0.4, 300)
    tauX = 0.5e-9
    k = b.marcus_rate(1e-3, 0.25, -dE)
    etaCT = b.eta_competition(k, 1 / tauX)
    a.plot(dE, etaCT, color=BLUE, label="η_CT, Marcus (X5, X6)")
    Dn = np.exp(-dE / (2 * KT)) / 1.0
    a.plot(dE, Dn, color=VIOLET, lw=1.4, label="D*_shot rel. (E_CT ↓ → J0 ↑, N3)")
    sat(a, dE[np.argmax(etaCT >= 0.9)], "η_CT ≥ 0.9")
    fail(a, dE[np.argmax(Dn < 0.01)], 0.4, "D* < 1%", y=0.5)
    a.set(xlabel="ΔE_LE–CT (eV)", ylabel="relative", title="B1 界面能级差 (X5, N3)")
    a.legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.25, 0.25))

    # B2 界面耦合：正向与回传都 ∝ |V|²
    a = ax[4]
    v2 = np.geomspace(1e-3, 1e3, 300)
    etaCT = v2 / (v2 + 1.0)
    etad = 1.0 / (1.0 + 0.05 * v2)
    a.plot(v2, etaCT * etad, color=BLUE, label="η_CT·η_diss")
    a.plot(v2, etaCT, color=MUTED, lw=1, ls="--", label="η_CT")
    a.plot(v2, etad, color=VIOLET, lw=1, ls="--", label="η_diss (back transfer)")
    sat(a, v2[np.argmax(etaCT * etad)], "optimum")
    fail(a, v2[np.argmax(etaCT * etad)], 1e3, "back transfer\ndominates", y=0.3)
    a.set(xscale="log", xlabel="|V_DA|² (relative)", ylabel="yield", title="B2 界面耦合")
    legend(a, "lower left")

    # B3 非生产性猝灭
    a = ax[5]
    kq = np.geomspace(1e-3, 1e2, 300)
    d, L0 = 20e-9, 20e-9
    a.plot(kq, b.eta_slab(d, L0 / np.sqrt(1 + kq)), color=BLUE, label="η_slab with L' = L/√(1+k_qτ)")
    sat(a, 0.1, "k_q τ = ε_s")
    a.set(xscale="log", xlabel="k_q · τ", ylabel="η_reach", title="B3 非生产性猝灭 (X1)")
    legend(a, "lower left")

    # B4 激子阻挡层
    a = ax[6]
    x = np.geomspace(0.1, 10, 300)
    gain = b.eta_slab(x, 1.0, 0.0) / b.eta_slab(x, 1.0, 1e6)
    a.plot(x, gain, color=BLUE, label="η(s=0)/η(s=∞) (X1)")
    sat(a, x[np.argmax(gain < 1.1)], "gain < 10%", y=0.5)
    a.set(xscale="log", xlabel="d / L_D", ylabel="gain", title="B4 激子阻挡层 〔待核验〕 (X1)")
    a.text(0.12, 1.1, "fail: R_EBL adds to R_s\n(see P/f figure)", fontsize=6.5, color=ORANGE)
    legend(a, "upper right")

    # C1 能量景观
    a = ax[7]
    dG = np.linspace(0, 0.4, 300)
    eps_r, a0 = 3.5, 1.5e-9
    Eb = b.binding_energy(eps_r, a0)
    kd = 1e9 * np.exp(-(Eb - dG) / KT)
    eta = kd / (kd + 1e9)
    a.plot(dG, eta, color=BLUE, label="η_diss with cascade")
    sat(a, dG[np.argmax(eta >= 0.9)], "η ≥ 0.9", y=0.6)
    a.text(0.02, 0.75, "sat also needs F-, T-independence", fontsize=6.5, color=MUTED, transform=a.transAxes)
    a.set(xlabel="cascade energy ΔG_casc (eV)", ylabel="η_diss", title="C1 能量景观")
    a.text(0.01, 0.85, "fail: E_CT ↓ → J0 ↑ (N3)", fontsize=6.5, color=ORANGE, transform=a.transAxes)
    legend(a, "lower right")

    # C2 离域
    a = ax[8]
    aa = np.linspace(0.5, 25, 300) * 1e-9
    a.plot(aa * 1e9, b.binding_energy(3.5, aa) / KT, color=BLUE, label="E_b / kT, ε_r = 3.5 (C1)")
    a.axhline(1, color=MUTED, ls="--", lw=1)
    sat(a, b.coulomb_radius(3.5) * 1e9, f"r_c = {b.coulomb_radius(3.5)*1e9:.1f} nm (C2)")
    a.set(yscale="log", xlabel="electron–hole distance a (nm)", ylabel="E_b / kT", title="C2 CT 离域 (C1, C2)")
    a.text(0.3, 0.6, "fail: crystallinity ↑\n→ domains ↑ (A1)", fontsize=6.5, color=ORANGE,
           transform=a.transAxes)
    legend(a, "lower left")

    # C3 介电常数：只在分离时间尺度上有效
    a = ax[9]
    tt = np.geomspace(1e-14, 1e-7, 300)
    a.plot(tt, b.eps_dispersive(tt, 3.5, 10, 1e-9), color=BLUE, label="ε_eff(t), τ_D = 1 ns (P8)")
    req = 55.7 / 1.5
    a.axhline(req, color=MUTED, ls="--", lw=1)
    a.text(2e-14, req + 2, f"E_b = kT at a = 1.5 nm needs ε ≥ {req:.0f}", fontsize=6.5, color=MUTED)
    fail(a, 1e-14, 1e-11, "t_sep ≪ τ_D:\nno screening", y=0.6)
    a.set(xscale="log", ylim=(0, 45), xlabel="separation time scale t (s)", ylabel="ε_r", title="C3 介电常数 (C1, P8)")
    legend(a, "center right")

    # D5 垂直相分离
    a = ax[10]
    pw = np.linspace(0, 0.5, 200)
    a.plot(pw, 1 / (1 + 4 * pw), color=BLUE, label="η_coll vs wrong-phase fraction")
    sat(a, 0.025, "φ_w → 0", y=0.5)
    a.set(xlabel="wrong phase at contact φ_w", ylabel="η_coll", title="D5 垂直相分离")
    a.text(0.25, 0.9, "fail: over-purified → fewer\nD/A interfaces (A1)", fontsize=6.5, color=ORANGE)
    legend(a, "lower left")

    # D6 PPHJ/LbL：低给体比例时空穴通路
    a = ax[11]
    phiD = np.linspace(0.01, 0.6, 300)
    mu = b.percolation_mobility(phiD, 0.12, 1.0)
    reach = b.eta_slab(40e-9, 40e-9 * (1 - phiD))   # 给体越少，受体相越连续
    a.plot(phiD, mu / mu.max(), color=VIOLET, lw=1.4, label="hole μ (P9)")
    a.plot(phiD, reach / reach.max(), color=MUTED, lw=1.4, ls="--", label="exciton reach")
    a.plot(phiD, mu / mu.max() * reach / reach.max(), color=BLUE, label="product")
    fail(a, 0.01, 0.12, "φ_D < φ_c", y=0.5)
    a.set(ylim=(0, 1.45), xlabel="donor fraction φ_D", ylabel="relative", title="D6 PPHJ/LbL 及 A4 低给体 (X1, P9)")
    legend(a, "upper center")

    fig.suptitle("Material-parameter strategies (illustrative) — these move the d–V lines; blue: benefit, dotted: saturation, orange: failure",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig("material_axis.png", dpi=150)


# ====================================================================== 光强 / 频率
def pf_axis():
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.4))
    ax = axs.ravel()
    eps_r, A, d, R_load, f_t = 3.5, 4e-6, 300e-9, 50.0, 5e5

    # 3.1-5 寄生吸收：电极变薄透光↑，电阻↑
    a = ax[0]
    t = np.linspace(2e-9, 30e-9, 300)
    T_el, R_el = b.sheet_tradeoff(t, 2.4e-8, 4.0, 1100e-9, 200.0)   # 条形电极长宽比 200
    fRC = b.f_RC(R_load + R_el, eps_r, A, d)
    a.plot(t * 1e9, fRC, color=BLUE, label="f_RC with electrode R (B2)")
    a.axhline(f_t, color=MUTED, ls="--", lw=1)
    a.text(2.5, f_t * 1.2, "f*", fontsize=7, color=MUTED)
    t_lo = t[np.argmax(fRC >= f_t)]
    t_hi = t[np.argmax(T_el < 0.6)]
    fail(a, 2, t_lo * 1e9, "RC too\nslow", y=0.8)
    fail(a, t_hi * 1e9, 30, "T_el < 0.6\n(parasitic)", y=0.8)
    a.set(yscale="log", xlabel="metal electrode thickness t (nm)", ylabel="f (Hz)", title="3.1-5 寄生吸收 (P12, B2)")
    legend(a, "lower right")

    # D3 / 3.1-10 陷阱释放截止频率
    a = ax[1]
    Et = np.linspace(0.2, 0.7, 300)
    ft = b.f_trap_bulk(Et)
    a.plot(Et, ft, color=BLUE, label="f_trap = ν₀e^{−E_t/kT}/2π (P10)")
    a.axhline(f_t, color=MUTED, ls="--", lw=1)
    Ef = Et[np.argmax(ft < f_t)]
    fail(a, Ef, 0.7, "f_trap < f*")
    a.set(yscale="log", xlabel="trap depth E_t (eV)", ylabel="f (Hz)", title="D3 体陷阱 / 3.1-10 (P10)")
    legend(a, "upper right")

    # 3.1-12 上转换的光强线性
    a = ax[2]
    P = np.geomspace(1e-3, 1e3, 300)
    a.plot(P, b.eta_upconversion(P, 1.0), color=BLUE, label="η ∝ P/(P+P_th) (P6)")
    fail(a, 1e-3, 1.0, "P < P_th:\nnonlinear", y=0.5)
    sat(a, 10, "η within 10%", y=0.5)
    a.set(xscale="log", xlabel="P / P_th", ylabel="conversion η", title="3.1-12 光谱转换 〔待核验〕 (P6)")
    a.text(0.03, 0.9, "also f ≤ 1/(2πτ_em)", fontsize=6.5, color=ORANGE, transform=a.transAxes)
    legend(a, "lower right")

    # B4 阻挡层电阻
    a = ax[3]
    tE = np.linspace(0, 40e-9, 300)
    R_ebl = 1e6 * tE / A                 # ρ = 1e6 Ω·m（示意）
    f3 = b.f_3dB(b.f_RC(R_load + R_ebl, eps_r, A, d), b.f_transit(1e-7, 3.6, d))
    a.plot(tE * 1e9, f3, color=BLUE, label="f₃dB with R_EBL (B3)")
    a.axhline(f_t, color=MUTED, ls="--", lw=1)
    fail(a, tE[np.argmax(f3 < f_t)] * 1e9, 40, "f₃dB < f*", y=0.5)
    a.set(yscale="log", xlabel="EBL thickness (nm)", ylabel="f (Hz)", title="B4 激子阻挡层 〔待核验〕 (B2, K10)")
    legend(a, "lower left")

    # C4 非成对复合随光强
    a = ax[4]
    P = np.geomspace(1e-2, 1e3, 300)                       # W/m²
    Phi = P * 1100e-9 / (b.h * b.c)
    G = Phi * 0.5 / d
    gam = 1e-17
    for V, col in [(0.0, VIOLET), (5.0, BLUE)]:
        n = b.n_steady(G, d, 1e-7, 0.6 + V)
        loss = gam * n / (gam * n + 1e-7 * (0.6 + V) / d**2)
        a.plot(P, loss, color=col, label=f"non-geminate loss, V={V:g} V (P11)")
        if col == VIOLET:
            fail(a, P[np.argmax(loss > 0.1)], 1e3, "", y=0.5)
    a.axhline(0.1, color=MUTED, ls="--", lw=1)
    a.text(1.2e-2, 0.03, "ε_s", fontsize=7, color=MUTED)
    a.set(xscale="log", yscale="log", xlabel="light intensity P (W/m²)", ylabel="loss fraction",
          title="C4 竞争通道 (P11)")
    legend(a, "lower right")

    # D7 抽取势垒
    a = ax[5]
    Jph = 0.4 * P * 1100e-9 / 1.24e-6                      # A/m²，R ≈ 0.35 A/W
    a.plot(P, Jph, color=BLUE, label="J_ph = R·P")
    for phi_ex, col in [(0.55, MUTED), (0.65, VIOLET)]:
        Jm = b.J_extraction_limit(phi_ex)
        a.axhline(Jm, color=col, ls="--", lw=1.2, label=f"J_ex,max, Φ_ex={phi_ex} eV (K9)")
        Pc = P[np.argmax(Jph > Jm)] if (Jph > Jm).any() else None
        if Pc:
            fail(a, Pc, 1e3, "") if phi_ex == 0.65 else None
    a.set(xscale="log", yscale="log", xlabel="light intensity P (W/m²)", ylabel="J (A/m²)", title="D7 抽取势垒 (K9)")
    legend(a, "lower right")

    # D9 界面陷阱
    a = ax[6]
    P = np.geomspace(1e-3, 1e2, 300)
    ftr = b.f_trap_power(P, 2e4, 1e-3, 1.25)
    Pfill = b.P_trap_fill(1e15, 1100e-9, 0.5, 1e-4)
    ftr = np.where(P < Pfill, ftr, np.inf)
    f3 = b.f_3dB(b.f_RC(R_load, eps_r, A, d), b.f_transit(1e-7, 3.6, d), ftr)
    a.plot(P, f3, color=BLUE, label="f₃dB incl. f_trap (K13, B3)")
    sat(a, Pfill, "P_fill (K12)")
    a.text(0.25, 0.5, "passivation helps bandwidth\nonly for P < P_fill (S27)", fontsize=6.5,
           color=ORANGE, transform=a.transAxes)
    a.set(xscale="log", yscale="log", xlabel="light intensity P (W/m²)", ylabel="f (Hz)", title="D9 界面陷阱 (K12, K13)")
    legend(a, "lower right")

    # D10 串联电阻
    a = ax[7]
    R = np.geomspace(1, 1e4, 300)
    fRC = b.f_RC(R, eps_r, A, d)
    ftr = b.f_transit(1e-7, 3.6, d)
    a.plot(R, b.f_3dB(fRC, ftr), color=BLUE, label="f₃dB (B3)")
    a.plot(R, fRC, color=VIOLET, lw=1.2, ls="--", label="f_RC (B2)")
    a.axhline(ftr, color=MUTED, lw=1.2, ls="--", label="f_tr (B1)")
    sat(a, R[np.argmax(fRC < 3 * ftr)], "f_RC = 3f_tr")
    a.set(xscale="log", yscale="log", xlabel="series + load resistance R (Ω)", ylabel="f (Hz)", title="D10 串联电阻 (B1–B3, K10)")
    legend(a, "lower left")

    fig.suptitle("Light-intensity / frequency strategies (illustrative) — blue: benefit, dotted: saturation, orange: failure",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig("pf_axis.png", dpi=150)


if __name__ == "__main__":
    lambda_axis()
    material_axis()
    pf_axis()
    print("saved lambda_axis.png, material_axis.png, pf_axis.png")
