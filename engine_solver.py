"""
engine_solver.py — Self-consistent engine operating point solver
"""
import numpy as np, sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from injector import Injector

class EngineSolver:
    def __init__(self, At, injector, fuel="H2", oxidizer="O2",
                 mechanism="gri30.yaml", OF_init=6.0,
                 tol_pc=1.0, tol_OF=1e-3, max_outer=30):
        self.At=At; self.injector=injector; self.fuel=fuel; self.oxidizer=oxidizer
        self.mechanism=mechanism; self.OF_init=OF_init
        self.tol_pc=tol_pc; self.tol_OF=tol_OF; self.max_outer=max_outer

    def _cantera_chamber(self, OF, Pc_est):
        import cantera as ct
        R_U=ct.gas_constant
        gas=ct.Solution(self.mechanism)
        mF=1.0/(1.0+OF); mO=OF/(1.0+OF)
        MW={'H2':2.016,'O2':32.0,'CH4':16.043,'C2H6':30.069,'N2':28.014,
            'C3H8':44.097,'C4H10':58.123}
        mwF=MW.get(self.fuel,28.0); mwO=MW.get(self.oxidizer,32.0)
        nF=mF/mwF; nO=mO/mwO; nt=nF+nO
        gas.TPX=300.0, max(Pc_est,1e4), {self.fuel:nF/nt, self.oxidizer:nO/nt}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gas.equilibrate('HP')
        g=gas.cp_mass/gas.cv_mass; R=R_U/gas.mean_molecular_weight; T0=gas.T
        Gamma=np.sqrt(g)*(2.0/(g+1.0))**((g+1.0)/(2.0*(g-1.0)))
        return T0, g, R, np.sqrt(R*T0)/Gamma, gas.mean_molecular_weight

    def _solve_pc(self, Cstar):
        inj=self.injector; Pc=5e6
        for _ in range(80):
            mf,mo=inj.mass_flow(self._ptf,self._pto,Pc)
            f=mf+mo-Pc*self.At/Cstar
            dP=Pc*1e-5+1.0
            mf2,mo2=inj.mass_flow(self._ptf,self._pto,Pc+dP)
            df=(mf2+mo2-(Pc+dP)*self.At/Cstar-f)/dP
            Pc_new=max(Pc-f/(df+1e-30), 1e4)
            if abs(Pc_new-Pc)<self.tol_pc: Pc=Pc_new; break
            Pc=Pc_new
        return Pc

    def solve(self, p_tank_f=8e6, p_tank_o=8e6):
        self._ptf=p_tank_f; self._pto=p_tank_o
        OF=self.OF_init; Pc=5e6
        T0,gamma,R,Cstar,MW=self._cantera_chamber(OF,Pc)
        for _ in range(self.max_outer):
            Pc=self._solve_pc(Cstar)
            mf,mo=self.injector.mass_flow(self._ptf,self._pto,Pc)
            OF_new=mo/max(mf,1e-12)
            T0,gamma,R,Cstar,MW=self._cantera_chamber(OF_new,Pc)
            if abs(OF_new-OF)<self.tol_OF: OF=OF_new; break
            OF=0.5*OF+0.5*OF_new
        mf,mo=self.injector.mass_flow(self._ptf,self._pto,Pc)
        return dict(pc=Pc,T0=T0,gamma=gamma,R=R,Cstar=Cstar,
                    OF=OF,mdot_f=mf,mdot_o=mo,mdot_total=mf+mo,MW=MW,
                    p_tank_f=p_tank_f,p_tank_o=p_tank_o)

if __name__=="__main__":
    inj=Injector(fuel="H2",oxidizer="O2",A_f=2e-4,A_o=2e-4)
    s=EngineSolver(np.pi*0.04**2,inj,fuel="H2",oxidizer="O2")
    r=s.solve(8e6,8e6)
    print(f"Pc={r['pc']/1e6:.3f}MPa  OF={r['OF']:.3f}  T0={r['T0']:.0f}K  C*={r['Cstar']:.0f}  mdot={r['mdot_total']:.4f}kg/s")
