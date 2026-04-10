"""
nozzle_optimizer.py — System-level rocket nozzle optimizer

Two modes:
  1. MANUAL  — user provides T0, gamma, R directly (original behaviour)
  2. SYSTEM  — user provides propellant pair + O/F + P0; Cantera computes T0/gamma/R

Free variables in system mode: [theta_i, theta_e, n_lines, OF_ratio]
Free variables in manual mode:  [theta_i, theta_e, n_lines]

Maximizes vacuum Isp with divergence correction.
Results feed into LLM_based_DB via post_processor.

Usage (manual):
    python nozzle_optimizer.py --name rao_01 --n_eval 60

Usage (system):
    python nozzle_optimizer.py --name lox_h2_sys \\
        --fuel H2 --oxidizer O2 --OF 6.0 \\
        --P0 7e6 --throat_radius 0.04 --n_eval 150

Usage (system + optimize O/F):
    python nozzle_optimizer.py --name lox_h2_full \\
        --fuel H2 --oxidizer O2 --optimize_OF \\
        --P0 7e6 --throat_radius 0.04 --n_eval 200
"""

import numpy as np, json, os, sys, argparse
from datetime import datetime
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution, minimize, brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nozzle_analysis import isentropic, area_ratio, mach_from_pressure_ratio
from moc import rao_contour

DEFAULTS = dict(
    name="rao_opt_01",
    # Manual thermodynamic inputs (used when fuel/oxidizer not set)
    gamma=1.4, R=287.0, T0=3000.0, P0=5e6, Pe=101325.0,
    throat_radius=0.05, max_length_m=0.6, n_eval=60,
    # Propellant inputs (system mode — override manual if set)
    fuel=None, oxidizer=None, OF=None, optimize_OF=False,
)

# ── Chamber conditions ─────────────────────────────────────────────────────────

def get_chamber(params, OF_override=None):
    """
    Return (T0, gamma, R) from either Cantera or manual inputs.
    OF_override used during optimization when OF is a free variable.
    """
    fuel = params.get("fuel")
    oxidizer = params.get("oxidizer")

    if fuel is not None:
        from chamber import chamber_conditions
        OF = OF_override if OF_override is not None else params.get("OF", 6.0)
        c = chamber_conditions(oxidizer, fuel, OF, params["P0"])
        return c["T0"], c["gamma"], c["R"], c
    else:
        return params["T0"], params["gamma"], params["R"], None


# ── Physics ────────────────────────────────────────────────────────────────────

def perf_from_contour(wall, params, T0=None, gamma=None, R=None):
    g  = gamma if gamma is not None else params["gamma"]
    R  = R     if R     is not None else params["R"]
    T0 = T0    if T0    is not None else params["T0"]
    P0 = params["P0"]; Pe = params["Pe"]; Rt = params["throat_radius"]

    At = np.pi*Rt**2; Re = wall[-1,1]; Ae = np.pi*Re**2; Ae_At = Ae/At
    if Ae_At <= 1.01:
        return None
    try:
        Me = brentq(lambda M: area_ratio(M,g) - Ae_At, 1.001, 20.0)
    except Exception:
        return None

    T_re,_,_ = isentropic(Me,g); Te=T0*T_re; Ve=Me*np.sqrt(g*R*Te)
    rho0=P0/(R*T0); Ts=T0*2/(g+1); Ps=P0*(2/(g+1))**(g/(g-1))
    rho_s=rho0*(2/(g+1))**(1/(g-1)); Vs=np.sqrt(g*R*Ts); mdot=rho_s*Vs*At

    Tv = mdot*Ve + Pe*Ae
    # Divergence loss correction
    if len(wall) >= 5:
        dx=wall[-1,0]-wall[-5,0]; dr=wall[-1,1]-wall[-5,1]
        exit_ang = np.arctan2(dr,dx) if abs(dx)>1e-12 else 0.0
    else:
        exit_ang = 0.0
    lam = 0.5*(1.0 + np.cos(exit_ang))
    Tv_eff = Tv*lam; Isp_v=Tv_eff/(mdot*9.80665); Cf_v=Tv_eff/(P0*At)

    return dict(Me=Me, Ve=Ve, mdot=mdot, Ae_At=Ae_At, Re=Re,
                Thrust_vac=Tv_eff, Isp_vac=Isp_v, Cf_vac=Cf_v,
                length=wall[-1,0]-wall[0,0], Te=Te,
                divergence_lambda=lam, T0=T0, gamma=g, R=R)


# ── Objective ──────────────────────────────────────────────────────────────────
_hist = []

def objective(x, params, max_length_m, optimize_OF):
    if optimize_OF:
        ti, te, nl, OF = x; nl=max(int(round(nl)),3)
    else:
        ti, te, nl = x; nl=max(int(round(nl)),3); OF=None

    try:
        T0, gamma, R, _ = get_chamber(params, OF_override=OF)
    except Exception as e:
        _hist.append(dict(theta_i=ti,theta_e=te,n_lines=nl,OF=OF,
                          Isp_vac=np.nan,length_m=np.nan,Cf_vac=np.nan))
        return 1e9

    Me_t = mach_from_pressure_ratio(params["Pe"]/params["P0"], gamma)
    try:
        wall = rao_contour(gamma, params["throat_radius"], Me_t, ti, te, nl)
    except Exception:
        _hist.append(dict(theta_i=ti,theta_e=te,n_lines=nl,OF=OF,
                          Isp_vac=np.nan,length_m=np.nan,Cf_vac=np.nan))
        return 1e9

    perf = perf_from_contour(wall, params, T0=T0, gamma=gamma, R=R)
    if perf is None:
        _hist.append(dict(theta_i=ti,theta_e=te,n_lines=nl,OF=OF,
                          Isp_vac=np.nan,length_m=np.nan,Cf_vac=np.nan))
        return 1e9

    penalty = max(0.0, perf["length"]-max_length_m)*1e6
    cost = -perf["Isp_vac"] + penalty
    _hist.append(dict(theta_i=round(ti,3), theta_e=round(te,3), n_lines=nl,
                      OF=round(OF,3) if OF is not None else None,
                      Isp_vac=round(perf["Isp_vac"],3),
                      length_m=round(perf["length"],5),
                      Cf_vac=round(perf["Cf_vac"],5)))
    return cost


# ── Cone baseline ──────────────────────────────────────────────────────────────

def cone_baseline(params, T0, gamma, R):
    import importlib; na = importlib.import_module("nozzle_analysis")
    p = dict(params); p["half_angle_deg"]=15.0; p["n_points"]=100
    p["name"]="_baseline"; p["T0"]=T0; p["gamma"]=gamma; p["R"]=R
    r = na.run(p)
    _,_,g,R_,T0_,P0,Pe,Rt,Re,At,Ae,Ae_At,Me,Te,Ve,mdot,Th,Tv,Isp,Isp_v,Cf,Ts,Ps,Vs,*_ = r
    lam = 0.5*(1.0 + np.cos(np.radians(15.0)))
    return dict(Isp_vac=Isp_v*lam, Thrust_vac=Tv*lam, Cf_vac=Cf*lam, Me=Me)


# ── Plots ──────────────────────────────────────────────────────────────────────

def make_plots(df, wall, perf, params, baseline, out_dir, name):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"System Optimizer — {name}", fontsize=13)
    valid = df[df["Isp_vac"].notna() & (df["Isp_vac"]>0)]

    # 1. Convergence
    if len(valid):
        rb = valid["Isp_vac"].cummax()
        axes[0,0].plot(valid.index, valid["Isp_vac"],".",alpha=0.3,ms=3,color="steelblue")
        axes[0,0].plot(rb.index, rb, "r-", lw=2, label="Best")
        axes[0,0].axhline(baseline["Isp_vac"], color="orange", ls="--", label="Cone baseline")
    axes[0,0].set_xlabel("Eval"); axes[0,0].set_ylabel("Isp_vac [s]")
    axes[0,0].set_title("Convergence"); axes[0,0].legend(fontsize=8)

    # 2. Isp vs length
    if len(valid):
        axes[0,1].scatter(valid["length_m"],valid["Isp_vac"],c=valid.index,cmap="viridis",s=8,alpha=0.5)
        axes[0,1].axvline(params["max_length_m"],color="r",ls="--",label="max L")
        bi=valid["Isp_vac"].idxmax()
        axes[0,1].plot(valid.loc[bi,"length_m"],valid.loc[bi,"Isp_vac"],"r*",ms=14,label="Best")
    axes[0,1].set_xlabel("Length [m]"); axes[0,1].set_ylabel("Isp_vac [s]")
    axes[0,1].set_title("Isp vs Length"); axes[0,1].legend(fontsize=8)

    # 3. O/F sweep (if available)
    if "OF" in df.columns and df["OF"].notna().any():
        vof = valid[valid["OF"].notna()]
        axes[0,2].scatter(vof["OF"],vof["Isp_vac"],c=vof.index,cmap="plasma",s=8,alpha=0.5)
        axes[0,2].set_xlabel("O/F ratio"); axes[0,2].set_ylabel("Isp_vac [s]")
        axes[0,2].set_title("O/F Effect")
    else:
        axes[0,2].axis("off")
        axes[0,2].text(0.5,0.5,"O/F fixed\n(single-point)",ha="center",va="center",fontsize=11)

    # 4. Optimal contour
    xs,rs=wall[:,0],wall[:,1]
    axes[1,0].fill_between(xs*100,-rs*100,rs*100,alpha=0.25,color="steelblue")
    axes[1,0].plot(xs*100,rs*100,"b-",lw=2); axes[1,0].plot(xs*100,-rs*100,"b-",lw=2)
    axes[1,0].set_xlabel("x [cm]"); axes[1,0].set_ylabel("r [cm]")
    axes[1,0].set_title("Optimal Contour"); axes[1,0].set_aspect("equal")

    # 5. Mach distribution
    g=perf["gamma"]
    AR=np.pi*rs**2/(np.pi*params["throat_radius"]**2)
    M_ax=[]
    for ar,x in zip(AR,xs):
        try:
            M=brentq(lambda M:area_ratio(M,g)-ar,1.001,20.) if x>=0 \
              else brentq(lambda M:area_ratio(M,g)-ar,0.01,0.9999)
        except: M=1.0
        M_ax.append(M)
    axes[1,1].plot(xs*100,M_ax,"g-",lw=2)
    axes[1,1].axvline(0,color="r",ls="--",label="Throat")
    axes[1,1].set_xlabel("x [cm]"); axes[1,1].set_ylabel("Mach")
    axes[1,1].set_title("Mach Distribution"); axes[1,1].legend()

    # 6. Performance bar chart
    labels=["Cone baseline","Rao optimum"]
    isps=[baseline["Isp_vac"],perf["Isp_vac"]]
    colors=["orange","steelblue"]
    bars=axes[1,2].bar(labels,isps,color=colors,width=0.5)
    for bar,val in zip(bars,isps):
        axes[1,2].text(bar.get_x()+bar.get_width()/2, val+0.5, f"{val:.1f}s",
                       ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1,2].set_ylabel("Isp_vac [s]"); axes[1,2].set_title("Performance Comparison")
    improv=(perf["Isp_vac"]-baseline["Isp_vac"])/baseline["Isp_vac"]*100
    axes[1,2].set_ylim(min(isps)*0.98, max(isps)*1.03)
    axes[1,2].text(0.5,0.05,f"Improvement: {improv:+.2f}%",
                   transform=axes[1,2].transAxes, ha="center", fontsize=11, color="green")

    plt.tight_layout()
    p=os.path.join(out_dir,"optimizer_plots.png")
    plt.savefig(p,dpi=120); plt.close()
    return p


# ── Save ───────────────────────────────────────────────────────────────────────

def _c(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    return v

def save_outputs(params, perf, wall, best_x, baseline, df,
                 plot_path, out_dir, name, optimize_OF, chamber_info):
    ts = datetime.now().isoformat()
    if optimize_OF:
        ti,te,nl,OF_best = best_x; nl=int(round(nl))
    else:
        ti,te,nl = best_x[:3]; nl=int(round(nl))
        OF_best = params.get("OF")

    improv = (perf["Isp_vac"]-baseline["Isp_vac"])/baseline["Isp_vac"]*100
    np.savetxt(os.path.join(out_dir,"contour.csv"),wall,delimiter=",",header="x_m,r_m",comments="")
    df.to_csv(os.path.join(out_dir,"optimization_history.csv"),index_label="eval_id")

    # Chamber info block for JSON/frontmatter
    cinfo = {}
    if chamber_info:
        cinfo = {k:_c(v) for k,v in chamber_info.items()
                 if k not in ("oxidizer","fuel")}

    res = {"run_name":name,"timestamp":ts,
           "inputs":{k:_c(v) for k,v in params.items() if v is not None},
           "chamber":cinfo,
           "best":{k:_c(v) for k,v in perf.items()},
           "best_params":{"theta_i_deg":round(ti,3),"theta_e_deg":round(te,3),
                          "n_lines":nl,"OF":round(OF_best,3) if OF_best else None},
           "baseline_cone":{k:_c(v) for k,v in baseline.items()},
           "improvement_pct":round(improv,3),"n_evaluations":len(df),
           "contour_points":len(wall)}
    with open(os.path.join(out_dir,"results.json"),"w") as f:
        json.dump(res,f,indent=2)

    # Propellant line for frontmatter
    prop_line = ""
    if params.get("fuel"):
        prop_line = f"propellants: {params.get('oxidizer','')}/{params['fuel']}\nOF_ratio: {round(OF_best,3) if OF_best else 'N/A'}\n"

    w=wall; si=list(range(min(5,len(w))))+list(range(max(len(w)-5,5),len(w)))
    rows="\n".join(f"| {w[i,0]:.5f} | {w[i,1]*100:.3f} |" for i in si)

    md = f"""---
solver: rocket_nozzle_optimizer
run_name: {name}
timestamp: {ts}
{prop_line}gamma: {round(perf['gamma'],4)}
R_J_kgK: {round(perf['R'],2)}
T0_K: {round(perf['T0'],1)}
P0_Pa: {params['P0']}
Pe_Pa: {params['Pe']}
throat_radius_m: {params['throat_radius']}
max_length_m: {params['max_length_m']}
n_evaluations: {len(df)}
best_Isp_vacuum_s: {round(perf['Isp_vac'],2)}
best_thrust_vacuum_N: {round(perf['Thrust_vac'],2)}
best_Cf_vac: {round(perf['Cf_vac'],4)}
best_theta_i_deg: {round(ti,3)}
best_theta_e_deg: {round(te,3)}
best_n_lines: {nl}
contour_points: {len(wall)}
cone_baseline_Isp_vac_s: {round(baseline['Isp_vac'],2)}
improvement_over_cone_pct: {round(improv,3)}
divergence_lambda: {round(perf['divergence_lambda'],4)}
---

# Rocket Nozzle System Optimizer — {name}

## Best Result vs Cone Baseline

| Metric | Rao Optimum | Straight Cone 15° | Improvement |
|---|---|---|---|
| Isp_vac [s] | {perf['Isp_vac']:.2f} | {baseline['Isp_vac']:.2f} | {improv:+.2f}% |
| Thrust_vac [N] | {perf['Thrust_vac']:.2f} | {baseline['Thrust_vac']:.2f} | — |
| Exit Mach | {perf['Me']:.4f} | {baseline['Me']:.4f} | — |
| Divergence λ | {perf['divergence_lambda']:.4f} | 0.9830 | — |

## Chamber Conditions
{"" if not chamber_info else f"- Propellants: {params.get('oxidizer','')}/{params.get('fuel','')}"}
- T0 = {round(perf['T0'],1)} K
- γ  = {round(perf['gamma'],4)}
- R  = {round(perf['R'],2)} J/kg/K
{"- C* = " + str(round(chamber_info['Cstar'],1)) + " m/s" if chamber_info else ""}

## Optimal Shape Parameters

| Parameter | Value |
|---|---|
| theta_i | {ti:.2f} deg |
| theta_e | {te:.2f} deg |
| MOC lines | {nl} |
{"| O/F ratio | " + str(round(OF_best,3)) + " |" if OF_best else ""}
| Nozzle length | {perf['length']:.4f} m |
| Exit radius | {perf['Re']*100:.3f} cm |
| Area ratio Ae/At | {perf['Ae_At']:.4f} |

## Wall Contour Sample

| x [m] | r [cm] |
|---|---|
{rows}

## Plots
![Optimizer Results](optimizer_plots.png)
"""
    lp=os.path.join(out_dir,f"log.nozzle_{name}")
    with open(lp,"w") as f: f.write(md)
    print(f"[SAVED] {out_dir}")
    return lp


# ── Main ───────────────────────────────────────────────────────────────────────

def run_optimizer(params):
    global _hist; _hist = []
    name = params["name"]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "results", name)
    if os.path.exists(out_dir):
        i = 2
        while os.path.exists(os.path.join(base_dir, "results", f"{name}_{i}")):
            i += 1
        name = f"{name}_{i}"
        params = dict(params, name=name)
        out_dir = os.path.join(base_dir, "results", name)
        print(f"[INFO] Name already exists, using '{name}'")
    os.makedirs(out_dir)

    optimize_OF = params.get("optimize_OF", False)
    max_L = params["max_length_m"]; n_eval = params["n_eval"]

    # Get initial chamber conditions (for info/baseline)
    T0_init, gamma_init, R_init, chamber_info = get_chamber(params)
    print(f"[CHAMBER] T0={T0_init:.0f}K  gamma={gamma_init:.4f}  R={R_init:.1f} J/kg/K")
    if chamber_info:
        print(f"          Cstar={chamber_info['Cstar']:.0f} m/s  MW={chamber_info['MW']:.2f}")

    # Bounds
    bounds = [(15.0,45.0),(2.0,15.0),(5.0,20.0)]
    if optimize_OF:
        bounds.append((2.0, 10.0))   # O/F ratio bounds

    popsize = max(5, n_eval//15); maxiter = max(5, n_eval//(popsize*3))
    print(f"[OPT] DE: popsize={popsize} maxiter={maxiter}  optimize_OF={optimize_OF} …")

    de = differential_evolution(objective, bounds,
                                args=(params, max_L, optimize_OF),
                                popsize=popsize, maxiter=maxiter,
                                tol=1e-3, seed=42, polish=False)
    print(f"[OPT] DE done. Best Isp_vac≈{-de.fun:.1f}s  evals={len(_hist)}")

    print("[OPT] Nelder-Mead polish …")
    nm = minimize(objective, de.x, args=(params, max_L, optimize_OF),
                  method="Nelder-Mead",
                  options={"maxiter":300,"xatol":0.1,"fatol":0.5})

    best_x = nm.x if nm.fun <= de.fun else de.x

    # Reconstruct best solution
    if optimize_OF:
        ti,te,nl,OF_b = best_x; nl=max(int(round(nl)),3)
        T0_b,gamma_b,R_b,chamber_info = get_chamber(params, OF_override=OF_b)
    else:
        ti,te,nl = best_x[:3]; nl=max(int(round(nl)),3)
        OF_b = params.get("OF"); T0_b=T0_init; gamma_b=gamma_init; R_b=R_init

    Me_t = mach_from_pressure_ratio(params["Pe"]/params["P0"], gamma_b)
    wall = rao_contour(gamma_b, params["throat_radius"], Me_t, ti, te, nl)
    perf = perf_from_contour(wall, params, T0=T0_b, gamma=gamma_b, R=R_b)

    if perf is None:
        ti,te,nl_f = de.x[:3]; nl=max(int(round(nl_f)),3)
        wall = rao_contour(gamma_b, params["throat_radius"], Me_t, ti, te, nl)
        perf = perf_from_contour(wall, params, T0=T0_b, gamma=gamma_b, R=R_b)

    if perf is None:
        raise RuntimeError("No valid solution found. Try larger max_length_m.")

    print(f"[OPT] Best: ti={ti:.2f}°  te={te:.2f}°  n={nl}"
          + (f"  O/F={OF_b:.3f}" if optimize_OF else "")
          + f"  Isp_vac={perf['Isp_vac']:.2f}s  L={perf['length']:.4f}m  λ={perf['divergence_lambda']:.4f}")

    print("[OPT] Cone baseline …")
    baseline = cone_baseline(params, T0_b, gamma_b, R_b)
    improv = (perf['Isp_vac']-baseline['Isp_vac'])/baseline['Isp_vac']*100
    print(f"[OPT] Improvement over cone: {improv:+.2f}%")

    df = pd.DataFrame(_hist); df.index.name = "eval_id"
    plot_path = make_plots(df, wall, perf, params, baseline, out_dir, name)
    log_path  = save_outputs(params, perf, wall, best_x, baseline, df,
                             plot_path, out_dir, name, optimize_OF, chamber_info)
    print(f"[DONE] {log_path}")
    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rocket Nozzle System Optimizer")
    for k, v in DEFAULTS.items():
        if v is None:
            parser.add_argument(f"--{k}", default=None)
        elif isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true", default=False)
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    p = vars(parser.parse_args())
    # If fuel provided but T0/gamma/R not explicitly changed, they'll be overridden by Cantera
    run_optimizer(p)
