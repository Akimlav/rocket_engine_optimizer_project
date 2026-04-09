"""
Parametrized Rocket Nozzle Analytical Code
1D isentropic nozzle performance (ideal gas)
Usage: python nozzle_analysis.py --name my_run --T0 3500 --P0 7e6 --Pe 101325
"""

import numpy as np
import json
import os
import argparse
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULTS = {
    "name":            "nozzle_run",
    "gamma":           1.4,
    "R":               287.0,
    "T0":              3000.0,
    "P0":              5e6,
    "Pe":              101325.0,
    "throat_radius":   0.05,
    "half_angle_deg":  15.0,
    "n_points":        200,
}

def isentropic(M, gamma):
    g = gamma
    T_r = 1.0 / (1.0 + (g-1)/2.0 * M**2)
    P_r = T_r ** (g/(g-1))
    rho_r = T_r ** (1.0/(g-1))
    return T_r, P_r, rho_r

def area_ratio(M, gamma):
    g = gamma
    return (1.0/M) * ((2.0/(g+1)) * (1.0 + (g-1)/2.0 * M**2)) ** ((g+1)/(2*(g-1)))

def mach_from_pressure_ratio(PR, gamma):
    lo, hi = 1.0, 15.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        _, pr, _ = isentropic(mid, gamma)
        if pr > PR: lo = mid
        else:       hi = mid
    return (lo + hi) / 2.0

def mach_from_area_ratio_local(AR, gamma, supersonic=True):
    if abs(AR - 1.0) < 1e-9:
        return 1.0
    g = gamma
    M = 2.5 if supersonic else 0.3
    for _ in range(200):
        f   = area_ratio(M, g) - AR
        Mhi = max(M + 1e-6, 1e-9)
        Mlo = max(M - 1e-6, 1e-9)
        df  = (area_ratio(Mhi, g) - area_ratio(Mlo, g)) / (Mhi - Mlo)
        step = -f/df if abs(df) > 1e-12 else 0.0
        M = max(M + step, 1e-9)
        if abs(step) < 1e-10:
            break
    return M

def run(params):
    g   = params["gamma"]
    R   = params["R"]
    T0  = params["T0"]
    P0  = params["P0"]
    Pe  = params["Pe"]
    Rt  = params["throat_radius"]
    ang = np.radians(params["half_angle_deg"])
    N   = params["n_points"]
    name = params["name"]

    At    = np.pi * Rt**2
    rho0  = P0 / (R * T0)

    T_star   = T0 * 2.0 / (g + 1.0)
    P_star   = P0 * (2.0/(g+1.0))**(g/(g-1.0))
    rho_star = rho0 * (2.0/(g+1.0))**(1.0/(g-1.0))
    V_star   = np.sqrt(g * R * T_star)
    mdot     = rho_star * V_star * At

    Me    = mach_from_pressure_ratio(Pe / P0, g)
    Ae_At = area_ratio(Me, g)
    Re    = Rt * np.sqrt(Ae_At)
    Ae    = At * Ae_At

    T_re, _, rho_re = isentropic(Me, g)
    Te   = T0 * T_re
    Ve   = Me * np.sqrt(g * R * Te)

    Thrust     = mdot * Ve + (Pe - 101325.0) * Ae
    Thrust_vac = mdot * Ve + Pe * Ae
    Isp        = Thrust     / (mdot * 9.80665)
    Isp_vac    = Thrust_vac / (mdot * 9.80665)
    Cf         = Thrust / (P0 * At)

    Nc = N // 3;  Nd = N - Nc
    r_conv = np.linspace(3.0 * Rt, Rt, Nc)
    x_conv = np.linspace(-0.5 * Re / np.tan(ang), 0.0, Nc)
    x_div  = np.linspace(0.0, Re / np.tan(ang), Nd)
    r_div  = np.clip(Rt + x_div * np.tan(ang), Rt, Re)

    x_all  = np.concatenate([x_conv, x_div])
    r_all  = np.concatenate([r_conv, r_div])
    AR_all = (np.pi * r_all**2) / At

    M_all = np.array([
        mach_from_area_ratio_local(ar, g, supersonic=(x >= 0))
        for ar, x in zip(AR_all, x_all)
    ])
    _, P_ratio_all, _ = isentropic(M_all, g)
    T_ratio_all, _, _ = isentropic(M_all, g)
    P_all = P0 * P_ratio_all
    T_all = T0 * T_ratio_all

    return (name, params, g, R, T0, P0, Pe, Rt, Re, At, Ae, Ae_At,
            Me, Te, Ve, mdot, Thrust, Thrust_vac, Isp, Isp_vac, Cf,
            T_star, P_star, V_star, x_all, r_all, M_all, P_all, T_all)

def save_results(result_tuple):
    (name, params, g, R, T0, P0, Pe, Rt, Re, At, Ae, Ae_At,
     Me, Te, Ve, mdot, Thrust, Thrust_vac, Isp, Isp_vac, Cf,
     T_star, P_star, V_star, x_all, r_all, M_all, P_all, T_all) = result_tuple

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", name)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Rocket Nozzle — {name}", fontsize=13)
    axes[0,0].plot(x_all*100, M_all)
    axes[0,0].axvline(0, color='r', ls='--', label='Throat')
    axes[0,0].set_xlabel("x [cm]"); axes[0,0].set_ylabel("Mach")
    axes[0,0].set_title("Mach Number"); axes[0,0].legend()
    axes[0,1].plot(x_all*100, P_all/1e5)
    axes[0,1].axvline(0, color='r', ls='--')
    axes[0,1].set_xlabel("x [cm]"); axes[0,1].set_ylabel("P [bar]")
    axes[0,1].set_title("Pressure")
    axes[1,0].plot(x_all*100, T_all)
    axes[1,0].axvline(0, color='r', ls='--')
    axes[1,0].set_xlabel("x [cm]"); axes[1,0].set_ylabel("T [K]")
    axes[1,0].set_title("Temperature")
    axes[1,1].fill_between(x_all*100, -r_all*100, r_all*100, alpha=0.3, color='steelblue')
    axes[1,1].plot(x_all*100, r_all*100, 'b')
    axes[1,1].plot(x_all*100, -r_all*100, 'b')
    axes[1,1].axvline(0, color='r', ls='--')
    axes[1,1].set_xlabel("x [cm]"); axes[1,1].set_ylabel("r [cm]")
    axes[1,1].set_title("Nozzle Profile"); axes[1,1].set_aspect('equal')
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "nozzle_plots.png")
    plt.savefig(plot_path, dpi=120); plt.close()

    results = {
        "run_name": name, "timestamp": datetime.now().isoformat(),
        "inputs": params,
        "performance": {
            "exit_mach": round(Me, 4), "area_ratio_Ae_At": round(Ae_At, 4),
            "throat_area_m2": round(At, 8), "exit_area_m2": round(Ae, 6),
            "throat_radius_m": round(Rt, 4), "exit_radius_m": round(Re, 4),
            "mass_flow_kg_s": round(mdot, 6), "exit_velocity_m_s": round(Ve, 2),
            "exit_temp_K": round(Te, 2), "exit_pressure_Pa": round(Pe, 2),
            "throat_temp_K": round(T_star, 2), "throat_pressure_Pa": round(P_star, 2),
            "thrust_N": round(Thrust, 2), "thrust_vacuum_N": round(Thrust_vac, 2),
            "Isp_s": round(Isp, 2), "Isp_vacuum_s": round(Isp_vac, 2),
            "thrust_coefficient_Cf": round(Cf, 4),
        },
        "plots": [plot_path],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    md = f"""---
solver: rocket_nozzle_analytical
run_name: {name}
timestamp: {results['timestamp']}
gamma: {g}
R_J_kgK: {R}
T0_K: {T0}
P0_Pa: {P0}
Pe_Pa: {Pe}
throat_radius_m: {Rt}
half_angle_deg: {params['half_angle_deg']}
exit_mach: {round(Me,4)}
Isp_s: {round(Isp,2)}
Isp_vacuum_s: {round(Isp_vac,2)}
thrust_N: {round(Thrust,2)}
thrust_vacuum_N: {round(Thrust_vac,2)}
mass_flow_kg_s: {round(mdot,6)}
---

# Rocket Nozzle Analysis — {name}

## Performance Summary

| Parameter | Value |
|---|---|
| Exit Mach | {Me:.4f} |
| Area Ratio Ae/At | {Ae_At:.4f} |
| Mass Flow | {mdot:.6f} kg/s |
| Exit Velocity | {Ve:.2f} m/s |
| Thrust (SL ref) | {Thrust:.2f} N |
| Thrust (vacuum) | {Thrust_vac:.2f} N |
| Isp (SL ref) | {Isp:.2f} s |
| Isp (vacuum) | {Isp_vac:.2f} s |
| Thrust Coeff Cf | {Cf:.4f} |

## Throat

| Parameter | Value |
|---|---|
| T* | {T_star:.2f} K |
| P* | {P_star/1e5:.4f} bar |
| V* | {V_star:.2f} m/s |

## Exit

| Parameter | Value |
|---|---|
| Te | {Te:.2f} K |
| Pe | {Pe/1e5:.4f} bar |
| Re | {Re*100:.2f} cm |

## Geometry
- Throat radius: {Rt*100:.2f} cm
- Exit radius: {Re*100:.2f} cm
- Half-angle: {params['half_angle_deg']} deg

## Plots
![Nozzle Analysis](nozzle_plots.png)
"""
    log_path = os.path.join(out_dir, f"log.nozzle_{name}")
    with open(log_path, "w") as f:
        f.write(md)

    print(f"[OK] {out_dir}")
    print(f"     Me={Me:.3f}  Isp={Isp:.1f}s  Isp_vac={Isp_vac:.1f}s  Thrust={Thrust:.1f}N  mdot={mdot:.5f}kg/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parametrized Rocket Nozzle Analysis")
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v, help=f"default: {v}")
    args = parser.parse_args()
    params = vars(args)
    result = run(params)
    save_results(result)
