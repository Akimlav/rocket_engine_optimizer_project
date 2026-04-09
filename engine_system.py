"""
engine_system.py — Full rocket engine system simulation + optimization

Pipeline:
  1. Injector model  → mdot_f, mdot_o  (Bernoulli orifice)
  2. EngineSolver    → equilibrium Pc, OF  (coupled Newton + Cantera)
  3. Cantera         → T0, gamma, R, C*  (HP equilibrium)
  4. moc.py          → Rao optimal nozzle contour
  5. nozzle_analysis → Isp, Thrust, Cf

Modes:
  --mode simulate   : single operating point, no optimization
  --mode optimize   : optimize nozzle shape at the engine operating point
  --mode sweep      : sweep one parameter (tank pressure, A_f, etc.)

Usage:
  python engine_system.py --mode simulate \\
      --fuel H2 --oxidizer O2 \\
      --throat_radius 0.04 \\
      --p_tank_f 8e6 --p_tank_o 8e6 \\
      --A_f 2e-4 --A_o 2e-4 \\
      --name lox_h2_engine_01

  python engine_system.py --mode optimize \\
      --fuel H2 --oxidizer O2 \\
      --throat_radius 0.04 \\
      --p_tank_f 8e6 --p_tank_o 8e6 \\
      --A_f 2e-4 --A_o 2e-4 \\
      --n_eval 100 --name lox_h2_opt_01
"""

import numpy as np, json, os, sys, argparse
from datetime import datetime
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from injector import Injector, PROPELLANT_DENSITIES
from engine_solver import EngineSolver
from nozzle_analysis import isentropic, area_ratio, mach_from_pressure_ratio
from moc import rao_contour
from scipy.optimize import brentq

DEFAULTS = dict(
    name="engine_01",
    fuel="H2", oxidizer="O2",
    throat_radius=0.04,
    Pe=101325.0,
    max_length_m=0.5,
    # Injector
    p_tank_f=8e6, p_tank_o=8e6,
    A_f=2e-4, A_o=2e-4,
    Cd_f=0.8, Cd_o=0.8,
    # Optimizer (mode=optimize)
    n_eval=80,
    mode="simulate",
)

# ── Nozzle performance from engine operating point ────────────────────────────

def nozzle_perf(engine_result, Rt, Pe, theta_i=30, theta_e=5, n_lines=10):
    """Given engine operating point, compute nozzle performance with Rao contour."""
    g  = engine_result['gamma']
    R  = engine_result['R']
    T0 = engine_result['T0']
    P0 = engine_result['pc']
    At = np.pi * Rt**2

    Me = mach_from_pressure_ratio(Pe/P0, g)
    wall = rao_contour(g, Rt, Me, theta_i, theta_e, n_lines)

    Re = wall[-1,1]; Ae = np.pi*Re**2; Ae_At = Ae/At
    try:
        Me_check = brentq(lambda M: area_ratio(M,g)-Ae_At, 1.001, 20.0)
    except Exception:
        Me_check = Me

    T_re,_,_ = isentropic(Me_check,g); Te=T0*T_re; Ve=Me_check*np.sqrt(g*R*Te)
    rho0=P0/(R*T0); Ts=T0*2/(g+1)
    rho_s=rho0*(2/(g+1))**(1/(g-1)); Vs=np.sqrt(g*R*Ts)
    mdot_nozzle = rho_s*Vs*At

    Tv = mdot_nozzle*Ve + Pe*Ae
    # Divergence correction
    if len(wall)>=5:
        dx=wall[-1,0]-wall[-5,0]; dr=wall[-1,1]-wall[-5,1]
        lam=0.5*(1+np.cos(np.arctan2(dr,dx) if abs(dx)>1e-12 else 0))
    else:
        lam=1.0
    Tv_eff=Tv*lam; Isp_v=Tv_eff/(mdot_nozzle*9.80665)
    Cf_v=Tv_eff/(P0*At)

    return dict(Me=Me_check, Ve=Ve, mdot_nozzle=mdot_nozzle,
                Ae_At=Ae_At, Re=Re, Thrust_vac=Tv_eff,
                Isp_vac=Isp_v, Cf_vac=Cf_v,
                length=wall[-1,0]-wall[0,0], Te=Te,
                divergence_lambda=lam, wall=wall)


# ── Simulate mode ─────────────────────────────────────────────────────────────

def simulate(params):
    inj = Injector(
        fuel=params['fuel'], oxidizer=params['oxidizer'],
        Cd_f=params['Cd_f'], Cd_o=params['Cd_o'],
        A_f=params['A_f'],   A_o=params['A_o'],
    )
    solver = EngineSolver(
        At=np.pi*params['throat_radius']**2,
        injector=inj,
        fuel=params['fuel'], oxidizer=params['oxidizer'],
    )
    print(f"[ENGINE] Solving coupled Pc/OF ({params['fuel']}/{params['oxidizer']}) …")
    eng = solver.solve(p_tank_f=params['p_tank_f'], p_tank_o=params['p_tank_o'])
    print(f"[ENGINE] Pc={eng['pc']/1e6:.3f}MPa  OF={eng['OF']:.3f}  "
          f"T0={eng['T0']:.0f}K  C*={eng['Cstar']:.0f}m/s  "
          f"mdot={eng['mdot_total']:.4f}kg/s")

    print("[NOZZLE] Computing Rao contour performance …")
    noz = nozzle_perf(eng, params['throat_radius'], params['Pe'])
    print(f"[NOZZLE] Me={noz['Me']:.3f}  Isp_vac={noz['Isp_vac']:.1f}s  "
          f"Thrust_vac={noz['Thrust_vac']:.1f}N  λ={noz['divergence_lambda']:.4f}")

    return eng, noz


# ── Optimize mode ─────────────────────────────────────────────────────────────

def optimize(params):
    from nozzle_optimizer import run_optimizer
    # First get engine operating point
    eng, _ = simulate(params)
    # Then run full shape optimizer with engine-derived chamber conditions
    opt_params = dict(
        name=params['name'],
        gamma=eng['gamma'], R=eng['R'], T0=eng['T0'],
        P0=eng['pc'], Pe=params['Pe'],
        throat_radius=params['throat_radius'],
        max_length_m=params['max_length_m'],
        n_eval=params['n_eval'],
        fuel=None, oxidizer=None, OF=None, optimize_OF=False,
    )
    print(f"\n[OPTIMIZER] Running shape optimizer at engine operating point …")
    return run_optimizer(opt_params), eng


# ── Save + plots ──────────────────────────────────────────────────────────────

def save_engine_results(params, eng, noz, out_dir, name):
    ts = datetime.now().isoformat()
    wall = noz['wall']
    np.savetxt(os.path.join(out_dir,"contour.csv"), wall,
               delimiter=",", header="x_m,r_m", comments="")

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Engine System — {name}  ({params['oxidizer']}/{params['fuel']})", fontsize=13)

    # 1. Nozzle contour
    xs,rs = wall[:,0],wall[:,1]
    axes[0,0].fill_between(xs*100,-rs*100,rs*100,alpha=0.25,color="steelblue")
    axes[0,0].plot(xs*100,rs*100,"b-",lw=2); axes[0,0].plot(xs*100,-rs*100,"b-",lw=2)
    axes[0,0].set_xlabel("x [cm]"); axes[0,0].set_ylabel("r [cm]")
    axes[0,0].set_title("Nozzle Contour (Rao)"); axes[0,0].set_aspect("equal")

    # 2. Mach distribution
    g=eng['gamma']
    AR=np.pi*rs**2/(np.pi*params['throat_radius']**2)
    M_ax=[]
    for ar,x in zip(AR,xs):
        try: M=brentq(lambda M:area_ratio(M,g)-ar,1.001,20.) if x>=0 \
             else brentq(lambda M:area_ratio(M,g)-ar,0.01,0.9999)
        except: M=1.0
        M_ax.append(M)
    axes[0,1].plot(xs*100,M_ax,"g-",lw=2)
    axes[0,1].axvline(0,color="r",ls="--",label="Throat")
    axes[0,1].set_xlabel("x [cm]"); axes[0,1].set_ylabel("Mach")
    axes[0,1].set_title("Mach Distribution"); axes[0,1].legend()

    # 3. Injector dP sensitivity: Isp vs tank pressure
    tank_range = np.linspace(params['p_tank_f']*0.7, params['p_tank_f']*1.3, 20)
    isps=[]
    from injector import Injector as Inj
    for pt in tank_range:
        inj2=Inj(fuel=params['fuel'],oxidizer=params['oxidizer'],
                 Cd_f=params['Cd_f'],Cd_o=params['Cd_o'],
                 A_f=params['A_f'],A_o=params['A_o'])
        try:
            from engine_solver import EngineSolver as ES
            s2=ES(At=np.pi*params['throat_radius']**2,injector=inj2,
                  fuel=params['fuel'],oxidizer=params['oxidizer'])
            e2=s2.solve(p_tank_f=pt,p_tank_o=pt)
            n2=nozzle_perf(e2,params['throat_radius'],params['Pe'])
            isps.append(n2['Isp_vac'])
        except: isps.append(np.nan)
    axes[0,2].plot(tank_range/1e6,isps,"b-o",ms=4)
    axes[0,2].axvline(params['p_tank_f']/1e6,color="r",ls="--",label="Design pt")
    axes[0,2].set_xlabel("Tank Pressure [MPa]"); axes[0,2].set_ylabel("Isp_vac [s]")
    axes[0,2].set_title("Isp vs Tank Pressure"); axes[0,2].legend()

    # 4. OF sensitivity
    OF_range=np.linspace(max(eng['OF']*0.6,1.5), eng['OF']*1.5, 15)
    isps_OF=[]
    from chamber import chamber_conditions
    for OF in OF_range:
        try:
            c=chamber_conditions(params['oxidizer'],params['fuel'],OF,eng['pc'])
            Me_t=mach_from_pressure_ratio(params['Pe']/eng['pc'],c['gamma'])
            w=rao_contour(c['gamma'],params['throat_radius'],Me_t,30,5,10)
            np_=nozzle_perf({**eng,'gamma':c['gamma'],'R':c['R'],'T0':c['T0']},
                            params['throat_radius'],params['Pe'])
            isps_OF.append(np_['Isp_vac'])
        except: isps_OF.append(np.nan)
    axes[1,0].plot(OF_range,isps_OF,"r-o",ms=4)
    axes[1,0].axvline(eng['OF'],color="b",ls="--",label=f"Design OF={eng['OF']:.2f}")
    axes[1,0].set_xlabel("O/F ratio"); axes[1,0].set_ylabel("Isp_vac [s]")
    axes[1,0].set_title("Isp vs O/F"); axes[1,0].legend()

    # 5. Mass flow balance diagram
    Pc_range=np.linspace(eng['pc']*0.3,params['p_tank_f']*0.95,50)
    from injector import Injector as Inj2
    inj3=Inj2(fuel=params['fuel'],oxidizer=params['oxidizer'],
               Cd_f=params['Cd_f'],Cd_o=params['Cd_o'],
               A_f=params['A_f'],A_o=params['A_o'])
    mdot_inj=[inj3.total_mdot(params['p_tank_f'],params['p_tank_o'],pc) for pc in Pc_range]
    mdot_noz=[pc*np.pi*params['throat_radius']**2/eng['Cstar'] for pc in Pc_range]
    axes[1,1].plot(Pc_range/1e6,mdot_inj,"b-",lw=2,label="Injector")
    axes[1,1].plot(Pc_range/1e6,mdot_noz,"r-",lw=2,label="Nozzle (C*)")
    axes[1,1].axvline(eng['pc']/1e6,color="g",ls="--",label=f"Eq. Pc={eng['pc']/1e6:.2f}MPa")
    axes[1,1].set_xlabel("Pc [MPa]"); axes[1,1].set_ylabel("mdot [kg/s]")
    axes[1,1].set_title("Mass Flow Balance"); axes[1,1].legend()

    # 6. Performance summary bar
    metrics=["Isp_vac [s]","Thrust [N]","mdot [kg/s]","Pc [MPa]"]
    vals=[noz['Isp_vac'],noz['Thrust_vac'],eng['mdot_total'],eng['pc']/1e6]
    axes[1,2].barh(metrics,vals,color=["steelblue","orange","green","red"])
    for i,(v,m) in enumerate(zip(vals,metrics)):
        axes[1,2].text(v*1.01,i,f"{v:.2f}",va="center",fontsize=9)
    axes[1,2].set_title("Performance Summary")

    plt.tight_layout()
    p=os.path.join(out_dir,"engine_plots.png")
    plt.savefig(p,dpi=120); plt.close()

    # JSON
    res={"run_name":name,"timestamp":ts,"inputs":{k:v for k,v in params.items()},
         "engine":{k:(float(v) if isinstance(v,float) else v)
                   for k,v in eng.items()},
         "nozzle":{k:(float(v) if isinstance(v,(float,np.floating)) else v)
                   for k,v in noz.items() if k!="wall"}}
    with open(os.path.join(out_dir,"results.json"),"w") as f: json.dump(res,f,indent=2)

    # Markdown log
    md=f"""---
solver: engine_system
run_name: {name}
timestamp: {ts}
fuel: {params['fuel']}
oxidizer: {params['oxidizer']}
throat_radius_m: {params['throat_radius']}
p_tank_f_Pa: {params['p_tank_f']}
p_tank_o_Pa: {params['p_tank_o']}
A_f_m2: {params['A_f']}
A_o_m2: {params['A_o']}
Cd_f: {params['Cd_f']}
Cd_o: {params['Cd_o']}
pc_Pa: {round(eng['pc'],0)}
OF_ratio: {round(eng['OF'],3)}
T0_K: {round(eng['T0'],1)}
Cstar_m_s: {round(eng['Cstar'],1)}
mdot_total_kg_s: {round(eng['mdot_total'],5)}
Isp_vacuum_s: {round(noz['Isp_vac'],2)}
thrust_vacuum_N: {round(noz['Thrust_vac'],2)}
Cf_vac: {round(noz['Cf_vac'],4)}
exit_mach: {round(noz['Me'],4)}
area_ratio_Ae_At: {round(noz['Ae_At'],4)}
divergence_lambda: {round(noz['divergence_lambda'],4)}
---

# Engine System Analysis — {name}

## Operating Point

| Parameter | Value |
|---|---|
| Propellants | {params['oxidizer']}/{params['fuel']} |
| Chamber Pressure Pc | {eng['pc']/1e6:.3f} MPa |
| O/F ratio | {eng['OF']:.3f} |
| Chamber Temp T0 | {eng['T0']:.0f} K |
| C* | {eng['Cstar']:.0f} m/s |
| Total mdot | {eng['mdot_total']:.5f} kg/s |
| mdot fuel | {eng['mdot_f']:.5f} kg/s |
| mdot oxidizer | {eng['mdot_o']:.5f} kg/s |

## Injector

| Parameter | Value |
|---|---|
| Tank pressure (fuel) | {params['p_tank_f']/1e6:.2f} MPa |
| Tank pressure (ox) | {params['p_tank_o']/1e6:.2f} MPa |
| Orifice area fuel | {params['A_f']:.2e} m² |
| Orifice area ox | {params['A_o']:.2e} m² |
| Cd fuel | {params['Cd_f']} |
| Cd ox | {params['Cd_o']} |

## Nozzle Performance (Rao Contour)

| Metric | Value |
|---|---|
| Exit Mach | {noz['Me']:.4f} |
| Area ratio Ae/At | {noz['Ae_At']:.4f} |
| Isp vacuum | {noz['Isp_vac']:.2f} s |
| Thrust vacuum | {noz['Thrust_vac']:.2f} N |
| Cf vacuum | {noz['Cf_vac']:.4f} |
| Divergence λ | {noz['divergence_lambda']:.4f} |
| Nozzle length | {noz['length']:.4f} m |

## Plots
![Engine System](engine_plots.png)
"""
    lp=os.path.join(out_dir,f"log.nozzle_{name}")
    with open(lp,"w") as f: f.write(md)
    print(f"[SAVED] {out_dir}")
    return lp


# ── Main ───────────────────────────────────────────────────────────────────────

def main(params):
    name = params["name"]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", name)
    if os.path.exists(out_dir):
        raise FileExistsError(f"results/{name} exists. Use a different --name.")
    os.makedirs(out_dir)

    if params["mode"] == "optimize":
        log_path, eng = optimize(params)
        # optimize() already saved its own outputs via nozzle_optimizer
        print(f"[SYSTEM DONE] optimizer log: {log_path}")
    else:
        eng, noz = simulate(params)
        log_path = save_engine_results(params, eng, noz, out_dir, name)
        print(f"[DONE] {log_path}")

    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rocket Engine System Simulator")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true", default=False)
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    main(vars(parser.parse_args()))
