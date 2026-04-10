"""
chamber.py — Cantera-based rocket combustion chamber solver
"""
import numpy as np, cantera as ct, json, os, argparse, warnings

R_UNIVERSAL = ct.gas_constant
MW = {'H2':2.016,'O2':32.0,'CH4':16.043,'C2H6':30.069,'C2H4':28.054,
      'N2':28.014,'C3H8':44.097,'C4H10':58.123}
PROPELLANT_PRESETS = {
    'LOX/H2':   dict(oxidizer='O2', fuel='H2',   OF_stoich=8.0,  OF_opt=6.0),
    'LOX/CH4':  dict(oxidizer='O2', fuel='CH4',  OF_stoich=4.0,  OF_opt=3.4),
    'LOX/C2H6': dict(oxidizer='O2', fuel='C2H6', OF_stoich=3.73, OF_opt=2.8),
    'LOX/C3H8': dict(oxidizer='O2', fuel='C3H8', OF_stoich=3.63, OF_opt=2.9),
    'COLD_N2':  dict(oxidizer=None, fuel='N2',   OF_stoich=None, OF_opt=None),
}

def _mole_fractions(fuel, oxidizer, OF_mass):
    mw_f=MW.get(fuel,28.0); mw_o=MW.get(oxidizer,32.0) if oxidizer else None
    if oxidizer is None or OF_mass is None: return {fuel:1.0}
    mf=1.0/(1.0+OF_mass); mo=OF_mass/(1.0+OF_mass)
    nf=mf/mw_f; no=mo/mw_o; nt=nf+no
    return {fuel:nf/nt, oxidizer:no/nt}

def chamber_conditions(oxidizer, fuel, OF, P0, mechanism='gri30.yaml'):
    """Compute equilibrium chamber conditions via Cantera HP combustion."""
    gas=ct.Solution(mechanism)
    gas.TPX=300.0, P0, _mole_fractions(fuel, oxidizer, OF)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gas.equilibrate('HP')
    T0=gas.T; cp=gas.cp_mass; cv=gas.cv_mass; gamma=cp/cv
    mw=gas.mean_molecular_weight; R=R_UNIVERSAL/mw
    g=gamma; Gamma=np.sqrt(g)*(2.0/(g+1.0))**((g+1.0)/(2.0*(g-1.0)))
    Cstar=np.sqrt(R*T0)/Gamma
    T_star=T0*2.0/(g+1.0)
    gf=ct.Solution(mechanism)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gf.TPX=T_star, P0*(2.0/(g+1.0))**(g/(g-1.0)), gas.X
    gamma_frozen=gf.cp_mass/gf.cv_mass
    return dict(T0=T0,gamma=gamma,R=R,Cstar=Cstar,MW=mw,cp=cp,
                frozen_gamma=gamma_frozen,oxidizer=oxidizer,fuel=fuel,OF=OF,P0=P0)

def of_sweep(oxidizer, fuel, P0, OF_min, OF_max, n_points=20, mechanism='gri30.yaml'):
    results=[]
    for OF in np.linspace(OF_min, OF_max, n_points):
        try: results.append(chamber_conditions(oxidizer,fuel,OF,P0,mechanism))
        except Exception as e: results.append({'OF':OF,'error':str(e)})
    return results

def best_OF(oxidizer, fuel, P0, OF_min, OF_max, n_points=20, mechanism='gri30.yaml'):
    valid=[r for r in of_sweep(oxidizer,fuel,P0,OF_min,OF_max,n_points,mechanism) if 'T0' in r]
    if not valid: raise RuntimeError("No valid equilibrium points")
    return max(valid, key=lambda r: r['T0'])

if __name__=="__main__":
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser()
    parser.add_argument("--fuel",default="H2"); parser.add_argument("--oxidizer",default="O2")
    parser.add_argument("--P0",type=float,default=7e6); parser.add_argument("--OF",type=float,default=None)
    parser.add_argument("--OF_min",type=float,default=2.0); parser.add_argument("--OF_max",type=float,default=10.0)
    parser.add_argument("--n_points",type=int,default=30)
    args=parser.parse_args()
    if args.OF is not None:
        c=chamber_conditions(args.oxidizer,args.fuel,args.OF,args.P0)
        print(f"\n{args.oxidizer}/{args.fuel}  O/F={args.OF}  P0={args.P0/1e6:.2f}MPa")
        print(f"  T0={c['T0']:.1f}K  gamma={c['gamma']:.4f}  R={c['R']:.2f}  Cstar={c['Cstar']:.1f}  MW={c['MW']:.3f}")
    else:
        print(f"Sweeping O/F {args.OF_min}-{args.OF_max} …")
        sweep=of_sweep(args.oxidizer,args.fuel,args.P0,args.OF_min,args.OF_max,args.n_points)
        valid=[r for r in sweep if 'T0' in r]
        best=max(valid,key=lambda r:r['T0'])
        print(f"  Best O/F={best['OF']:.2f}  T0={best['T0']:.0f}K  gamma={best['gamma']:.3f}  Cstar={best['Cstar']:.0f}m/s")
        OFs=[r['OF'] for r in valid]; T0s=[r['T0'] for r in valid]
        gammas=[r['gamma'] for r in valid]; Cstars=[r['Cstar'] for r in valid]
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        fig.suptitle(f"{args.oxidizer}/{args.fuel}  P0={args.P0/1e6:.1f}MPa",fontsize=12)
        axes[0,0].plot(OFs,T0s,'r-o',ms=4); axes[0,0].set_title("T0 [K]"); axes[0,0].set_xlabel("O/F")
        axes[0,1].plot(OFs,gammas,'b-o',ms=4); axes[0,1].set_title("gamma"); axes[0,1].set_xlabel("O/F")
        axes[1,0].plot(OFs,[r['R'] for r in valid],'g-o',ms=4); axes[1,0].set_title("R [J/kg/K]"); axes[1,0].set_xlabel("O/F")
        axes[1,1].plot(OFs,Cstars,'m-o',ms=4); axes[1,1].set_title("C* [m/s]"); axes[1,1].set_xlabel("O/F")
        plt.tight_layout()
        out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"results",
                         f"chamber_{args.fuel}_{args.oxidizer}_sweep.png")
        os.makedirs(os.path.dirname(out),exist_ok=True)
        plt.savefig(out,dpi=120); plt.close(); print(f"  Plot: {out}")
