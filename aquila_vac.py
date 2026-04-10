"""
aquila_vac.py — Aquila VAC LOX/C3H8 engine simulation

Reference target (estimated design values):
  Thrust_vac  = 94 kN
  Isp_vac     = 340 s
  Pc          = 9 MPa
  O/F         = 2.9
  mdot        = 28.2 kg/s
  Ae/At       = 80
  Throat area = 0.0065 m²   →  Rt ≈ 45.5 mm
  Exit area   = 0.52 m²
  Burn time   = 300 s
  Cycle       = gas-generator

Propellant: liquid propane (C3H8) + liquid oxygen (LOX)
Cantera mechanism: gri30.yaml (includes propane chemistry)

Run:
    python aquila_vac.py
    python aquila_vac.py --optimize      # also optimise nozzle shape
    python aquila_vac.py --cooling       # also run regen-cooling analysis
"""

import numpy as np
import sys
import os
import json
import argparse
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from injector import Injector
from engine_solver import EngineSolver
from engine_system import nozzle_perf, save_engine_results
from moc import rao_contour
from nozzle_analysis import mach_from_pressure_ratio, area_ratio
from scipy.optimize import brentq

# ── Reference engine specification ───────────────────────────────────────────

TARGET = {
    "name":             "Aquila VAC (estimated)",
    "cycle":            "gas_generator",
    "propellants":      ["LOX", "C3H8"],
    "thrust_vac":       94e3,       # N
    "Isp_vac":          340,        # s
    "chamber_pressure": 9e6,        # Pa
    "mixture_ratio":    2.9,        # O/F
    "mass_flow":        28.2,       # kg/s
    "expansion_ratio":  80,
    "throat_area":      0.0065,     # m²
    "exit_area":        0.52,       # m²
    "burn_time":        300,        # s
}

# ── Derived geometry ──────────────────────────────────────────────────────────

Rt   = np.sqrt(TARGET["throat_area"] / np.pi)   # 0.04549 m
At   = TARGET["throat_area"]
Ae   = TARGET["exit_area"]
Pc   = TARGET["chamber_pressure"]
OF   = TARGET["mixture_ratio"]
mdot = TARGET["mass_flow"]

# Propellant splits
mdot_ox   = mdot * OF / (1.0 + OF)             # 20.97 kg/s
mdot_fuel = mdot / (1.0 + OF)                  # 7.23 kg/s

# Tank / injector pressures (GG cycle: pump delivers ~1.4× Pc)
P_TANK = 1.4 * Pc                               # 12.6 MPa
dP_INJ = P_TANK - Pc                            # 3.6 MPa  (injector drop)
Cd     = 0.80

rho_lox  = 1141.0   # kg/m³
rho_prop = 530.0    # kg/m³  (liquid propane, ~231 K pressurised)

A_ox   = mdot_ox   / (Cd * np.sqrt(2.0 * rho_lox  * dP_INJ))
A_fuel = mdot_fuel / (Cd * np.sqrt(2.0 * rho_prop * dP_INJ))

# Exit pressure from expansion ratio (Cantera will refine Pe later)
# Approximate: solve isentropic area relation for a typical gamma~1.23
_g_approx = 1.23
_Pe_approx = Pc * (1.0 + (_g_approx - 1.0)/2.0 *
             (1.0/area_ratio(brentq(lambda M: area_ratio(M, _g_approx) - TARGET["expansion_ratio"],
                                   1.5, 25.0), _g_approx)
              * TARGET["expansion_ratio"])) ** (-_g_approx/(_g_approx-1.0))
# Simpler: use ~Pe for vacuum nozzle sized at 80:1 ≈ 3000–5000 Pa
Pe = 3500.0   # Pa  (nominal vacuum exit pressure for 80:1 expansion)


def run(optimize=False, cooling=False):
    name = "aquila_vac"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", name)
    if os.path.exists(out_dir):
        i = 2
        while os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "results", f"{name}_{i}")):
            i += 1
        name = f"{name}_{i}"
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", name)
        print(f"[INFO] Using run name '{name}'")
    os.makedirs(out_dir)

    print("\n" + "═"*60)
    print("  AQUILA VAC — LOX/C3H8  Engine Simulation")
    print("═"*60)
    print(f"\n  Reference target:")
    print(f"    Thrust_vac  = {TARGET['thrust_vac']/1e3:.1f} kN")
    print(f"    Isp_vac     = {TARGET['Isp_vac']:.0f} s")
    print(f"    Pc          = {TARGET['chamber_pressure']/1e6:.1f} MPa")
    print(f"    O/F         = {TARGET['mixture_ratio']:.2f}")
    print(f"    mdot        = {TARGET['mass_flow']:.1f} kg/s")
    print(f"    Ae/At       = {TARGET['expansion_ratio']:.0f}")
    print(f"    Rt          = {Rt*1000:.1f} mm")
    print(f"\n  Derived injector parameters:")
    print(f"    P_tank      = {P_TANK/1e6:.2f} MPa")
    print(f"    A_ox        = {A_ox:.5f} m²  ({A_ox*1e4:.2f} cm²)")
    print(f"    A_fuel      = {A_fuel:.5f} m²  ({A_fuel*1e4:.2f} cm²)")

    # ── Step 1: Engine operating point ───────────────────────────────────────

    print("\n[STEP 1] Solving coupled Pc/OF …")
    inj = Injector(
        fuel="C3H8", oxidizer="O2",
        A_f=A_fuel, A_o=A_ox,
        Cd_f=Cd, Cd_o=Cd,
        rho_f=rho_prop, rho_o=rho_lox,
    )
    solver = EngineSolver(
        At=At, injector=inj,
        fuel="C3H8", oxidizer="O2",
        OF_init=OF,
    )
    eng = solver.solve(p_tank_f=P_TANK, p_tank_o=P_TANK)

    print(f"  Pc    = {eng['pc']/1e6:.3f} MPa  (target {Pc/1e6:.1f} MPa)")
    print(f"  O/F   = {eng['OF']:.3f}        (target {OF:.2f})")
    print(f"  T0    = {eng['T0']:.0f} K")
    print(f"  C*    = {eng['Cstar']:.0f} m/s")
    print(f"  γ     = {eng['gamma']:.4f}")
    print(f"  MW    = {eng['MW']:.2f} g/mol")
    print(f"  mdot  = {eng['mdot_total']:.4f} kg/s  (target {mdot:.1f} kg/s)")

    # ── Step 2: Nozzle performance ────────────────────────────────────────────

    print("\n[STEP 2] Computing Rao nozzle at Ae/At = 80 …")
    # Use Pe that gives the target expansion ratio
    g = eng["gamma"]
    Me_target = brentq(lambda M: area_ratio(M, g) - TARGET["expansion_ratio"],
                       3.0, 20.0)
    Pe_sim = eng["pc"] * (1.0 + (g-1.0)/2.0 * Me_target**2) ** (-g/(g-1.0))

    noz = nozzle_perf(eng, Rt, Pe_sim, theta_i=30, theta_e=5, n_lines=12)

    print(f"  Me        = {noz['Me']:.3f}        (Ae/At = {noz['Ae_At']:.1f})")
    print(f"  Isp_vac   = {noz['Isp_vac']:.1f} s   (target {TARGET['Isp_vac']:.0f} s)")
    print(f"  Thrust_vac= {noz['Thrust_vac']/1e3:.2f} kN  (target {TARGET['thrust_vac']/1e3:.1f} kN)")
    print(f"  Cf_vac    = {noz['Cf_vac']:.4f}")
    print(f"  λ         = {noz['divergence_lambda']:.4f}")
    print(f"  Re        = {noz['Re']*1000:.1f} mm")
    print(f"  L_nozzle  = {noz['length']*100:.1f} cm")

    # ── Step 3: Comparison table ──────────────────────────────────────────────

    print("\n[STEP 3] Target vs Simulation:")
    rows = [
        ("Isp_vac [s]",     TARGET["Isp_vac"],              noz["Isp_vac"]),
        ("Thrust_vac [kN]", TARGET["thrust_vac"]/1e3,        noz["Thrust_vac"]/1e3),
        ("Pc [MPa]",        TARGET["chamber_pressure"]/1e6,  eng["pc"]/1e6),
        ("O/F",             TARGET["mixture_ratio"],         eng["OF"]),
        ("mdot [kg/s]",     TARGET["mass_flow"],             eng["mdot_total"]),
        ("Ae/At",           TARGET["expansion_ratio"],       noz["Ae_At"]),
    ]
    print(f"\n  {'Parameter':<20} {'Target':>12} {'Simulated':>12} {'Error':>10}")
    print("  " + "─"*56)
    for label, tgt, sim in rows:
        err = (sim - tgt) / tgt * 100.0
        flag = " ✓" if abs(err) < 3 else " ←"
        print(f"  {label:<20} {tgt:>12.3g} {sim:>12.3g} {err:>+9.1f}%{flag}")

    # ── Step 4: Propellant budget for burn time ───────────────────────────────

    print(f"\n[STEP 4] Propellant budget for {TARGET['burn_time']} s burn:")
    m_ox_total   = eng["mdot_o"] * TARGET["burn_time"]
    m_fuel_total = eng["mdot_f"] * TARGET["burn_time"]
    m_total      = eng["mdot_total"] * TARGET["burn_time"]
    print(f"  LOX consumed  = {m_ox_total:.1f} kg")
    print(f"  C3H8 consumed = {m_fuel_total:.1f} kg")
    print(f"  Total prop    = {m_total:.1f} kg")
    print(f"  Δv potential  = Isp × g₀ × ln(1 + m_prop/m_dry)")
    print(f"  (m_dry needed to estimate Δv — add tank + engine dry mass)")

    # ── Step 5: Optional nozzle shape optimisation ────────────────────────────

    if optimize:
        print("\n[STEP 5] Optimising nozzle shape (DE + Nelder-Mead) …")
        from nozzle_optimizer import run_optimizer
        opt_params = dict(
            name=f"{name}_opt",
            gamma=eng["gamma"], R=eng["R"], T0=eng["T0"],
            P0=eng["pc"], Pe=Pe_sim,
            throat_radius=Rt,
            max_length_m=1.5,
            n_eval=120,
            fuel=None, oxidizer=None, OF=None, optimize_OF=False,
        )
        log = run_optimizer(opt_params)
        print(f"  Optimizer result: {log}")

    # ── Step 6: Optional regenerative cooling ────────────────────────────────

    if cooling:
        print("\n[STEP 6] Regenerative cooling analysis (LOX/C3H8 — propane coolant) …")
        from cooling import CoolingChannelGeometry, CoolantProperties, solve_cooling_loop
        wall = noz["wall"]
        geom = CoolingChannelGeometry(
            n_channels=120, width=2.0e-3, height=4.0e-3,
            wall_thickness=1.2e-3, land_width=1.5e-3,
        )
        # Propane as coolant (fuel-cooled regen)
        coolant = CoolantProperties("CH4")   # use CH4 as proxy (similar cryogenic hydrocarbon)
        cool = solve_cooling_loop(
            engine_result=eng,
            nozzle_result=noz,
            wall_contour=wall,
            channel_geom=geom,
            coolant=coolant,
            wall_material="CuCrZr",
            mdot_coolant=eng["mdot_f"],
            n_stations=80,
        )
        print(f"  Max wall temp  = {cool['max_T_wg']:.0f} K")
        print(f"  Coolant ΔT     = {cool['T_cool_exit'] - coolant.T_boil:.0f} K rise")
        print(f"  Total heat     = {cool['total_heat']/1e6:.2f} MW")
        print(f"  Coolant ΔP     = {cool['dP_total']/1e6:.2f} MPa")
        print(f"  Min margin     = {np.min(cool['margin']):.2f}×  (must be > 1)")

    # ── Step 7: Plots + save ──────────────────────────────────────────────────

    print(f"\n[STEP 7] Generating plots …")
    _make_plots(eng, noz, name, out_dir)

    results = {
        "run_name": name,
        "timestamp": datetime.now().isoformat(),
        "target": TARGET,
        "engine": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                   for k, v in eng.items()},
        "nozzle": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                   for k, v in noz.items() if k != "wall"},
        "comparison": {label: {"target": tgt, "simulated": float(sim),
                                "error_pct": float((sim-tgt)/tgt*100)}
                       for label, tgt, sim in rows},
        "propellant_budget": {
            "burn_time_s": TARGET["burn_time"],
            "m_lox_kg": float(m_ox_total),
            "m_c3h8_kg": float(m_fuel_total),
            "m_total_kg": float(m_total),
        },
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Results saved to: results/{name}/")
    print(f"       results.json + aquila_vac_plots.png")
    return eng, noz


def _make_plots(eng, noz, name, out_dir):
    wall = noz["wall"]
    xs, rs = wall[:, 0], wall[:, 1]
    g = eng["gamma"]
    At = np.pi * Rt ** 2
    AR = np.pi * rs ** 2 / At

    M_ax = []
    for ar, x in zip(AR, xs):
        try:
            M = (brentq(lambda M: area_ratio(M, g) - ar, 1.001, 25.0) if x >= 0
                 else brentq(lambda M: area_ratio(M, g) - ar, 0.01, 0.9999))
        except Exception:
            M = 1.0
        M_ax.append(M)
    M_ax = np.array(M_ax)

    from nozzle_analysis import isentropic
    T_ratio, P_ratio, _ = isentropic(M_ax, g)
    T_ax = eng["T0"] * T_ratio
    P_ax = eng["pc"] * P_ratio

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Aquila VAC — LOX/C3H8  |  {name}", fontsize=13)

    # 1. Nozzle profile
    axes[0, 0].fill_between(xs * 100, -rs * 100, rs * 100, alpha=0.25, color="darkorange")
    axes[0, 0].plot(xs * 100,  rs * 100, color="darkorange", lw=2)
    axes[0, 0].plot(xs * 100, -rs * 100, color="darkorange", lw=2)
    axes[0, 0].axvline(0, color="red", ls="--", lw=1, label="Throat")
    axes[0, 0].set_xlabel("x [cm]"); axes[0, 0].set_ylabel("r [cm]")
    axes[0, 0].set_title(f"Nozzle Contour  (Ae/At = {noz['Ae_At']:.1f})")
    axes[0, 0].set_aspect("equal"); axes[0, 0].legend()

    # 2. Mach number
    axes[0, 1].plot(xs * 100, M_ax, color="steelblue", lw=2)
    axes[0, 1].axvline(0, color="red", ls="--", lw=1)
    axes[0, 1].set_xlabel("x [cm]"); axes[0, 1].set_ylabel("Mach")
    axes[0, 1].set_title(f"Mach Distribution  (Me = {noz['Me']:.2f})")

    # 3. Pressure profile
    axes[0, 2].plot(xs * 100, P_ax / 1e6, color="crimson", lw=2)
    axes[0, 2].axvline(0, color="red", ls="--", lw=1)
    axes[0, 2].set_xlabel("x [cm]"); axes[0, 2].set_ylabel("P [MPa]")
    axes[0, 2].set_title("Pressure Distribution")

    # 4. Temperature profile
    axes[1, 0].plot(xs * 100, T_ax, color="tomato", lw=2)
    axes[1, 0].axvline(0, color="red", ls="--", lw=1)
    axes[1, 0].set_xlabel("x [cm]"); axes[1, 0].set_ylabel("T [K]")
    axes[1, 0].set_title("Temperature Distribution")

    # 5. Target vs simulation bar chart
    labels = ["Isp_vac\n[s]", "Thrust\n[kN]", "Pc\n[MPa]", "mdot\n[kg/s]"]
    targets = [TARGET["Isp_vac"], TARGET["thrust_vac"]/1e3,
               TARGET["chamber_pressure"]/1e6, TARGET["mass_flow"]]
    simulated = [noz["Isp_vac"], noz["Thrust_vac"]/1e3,
                 eng["pc"]/1e6, eng["mdot_total"]]
    x_pos = np.arange(len(labels))
    w = 0.35
    axes[1, 1].bar(x_pos - w/2, targets,   w, label="Target",    color="steelblue", alpha=0.8)
    axes[1, 1].bar(x_pos + w/2, simulated, w, label="Simulated", color="darkorange", alpha=0.8)
    axes[1, 1].set_xticks(x_pos); axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_title("Target vs Simulation"); axes[1, 1].legend()

    # 6. Propellant budget pie
    sizes = [eng["mdot_o"] * TARGET["burn_time"],
             eng["mdot_f"] * TARGET["burn_time"]]
    labels_pie = [f"LOX\n{sizes[0]:.0f} kg", f"C3H8\n{sizes[1]:.0f} kg"]
    axes[1, 2].pie(sizes, labels=labels_pie, colors=["steelblue", "darkorange"],
                   autopct="%1.1f%%", startangle=90)
    axes[1, 2].set_title(f"Propellant Budget\n({TARGET['burn_time']} s burn, "
                          f"{sum(sizes):.0f} kg total)")

    plt.tight_layout()
    path = os.path.join(out_dir, "aquila_vac_plots.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Plot: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aquila VAC LOX/C3H8 engine simulation")
    parser.add_argument("--optimize", action="store_true",
                        help="Also run nozzle shape optimizer (slower)")
    parser.add_argument("--cooling",  action="store_true",
                        help="Also run regenerative cooling analysis")
    args = parser.parse_args()
    run(optimize=args.optimize, cooling=args.cooling)
