"""
tank.py — Propellant tank model (pressure-fed blowdown)

Models tank pressure decay during burn as ullage gas expands polytropic-
ally (or isentropically) when propellant is expelled.

Supports helium, nitrogen, and argon pressurant.  All units SI.
"""

import numpy as np
from scipy.integrate import solve_ivp

# ── Pressurant gas properties ─────────────────────────────────────────────────
PRESSURANT = {
    "He": dict(gamma=1.667, R=2077.0, MW=4.003),
    "N2": dict(gamma=1.4,   R=296.8,  MW=28.014),
    "Ar": dict(gamma=1.667, R=208.1,  MW=39.948),
}

# Re-use density table from injector.py where available
try:
    from injector import PROPELLANT_DENSITIES as _PROP_RHO
except ImportError:
    _PROP_RHO = {}


class Tank:
    """
    Propellant tank with blowdown pressurization.

    Parameters
    ----------
    V_tank          : float  total tank volume [m^3]
    P_0             : float  initial ullage pressure [Pa]
    propellant      : str    propellant name (for density lookup)
    rho_prop        : float  propellant liquid density [kg/m^3] (overrides lookup)
    ullage_fraction : float  initial ullage volume / total volume [-]  (default 0.05)
    pressurant      : str    pressurant gas name ('He', 'N2', 'Ar')
    T_pressurant    : float  initial pressurant temperature [K]  (default 300)
    polytropic_n    : float  polytropic exponent (None = use pressurant gamma)
    """

    def __init__(self, V_tank, P_0, propellant="LOX", rho_prop=None,
                 ullage_fraction=0.05, pressurant="He",
                 T_pressurant=300.0, polytropic_n=None):
        self.V_tank = V_tank
        self.P_0 = P_0
        self.pressurant_name = pressurant
        self.T_pres_0 = T_pressurant

        gas = PRESSURANT[pressurant]
        self.gamma_pres = gas['gamma']
        self.R_pres = gas['R']
        self.n = polytropic_n if polytropic_n is not None else gas['gamma']

        self.rho_prop = rho_prop if rho_prop is not None else _PROP_RHO.get(propellant, 800.0)
        self.propellant = propellant

        self.V_ullage_0 = V_tank * ullage_fraction
        self.V_prop_0 = V_tank - self.V_ullage_0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def m_prop_0(self):
        """Initial propellant mass [kg]."""
        return self.V_prop_0 * self.rho_prop

    @property
    def m_ullage(self):
        """Pressurant gas mass [kg] (constant during blowdown)."""
        return self.P_0 * self.V_ullage_0 / (self.R_pres * self.T_pres_0)

    # ── Steady-state helpers ──────────────────────────────────────────────────

    def pressure(self, m_expelled):
        """
        Tank pressure after m_expelled [kg] of propellant has left.

        Uses polytropic expansion:  P = P_0 * (V_ull_0 / V_ull)^n

        Parameters
        ----------
        m_expelled : float or ndarray [kg]

        Returns
        -------
        P : float or ndarray [Pa]
        """
        m_expelled = np.asarray(m_expelled, dtype=float)
        V_ull = self.V_ullage_0 + m_expelled / self.rho_prop
        V_ull = np.clip(V_ull, self.V_ullage_0, self.V_tank)
        return self.P_0 * (self.V_ullage_0 / V_ull) ** self.n

    def remaining_mass(self, m_expelled):
        """Propellant remaining [kg]."""
        return np.clip(self.m_prop_0 - np.asarray(m_expelled), 0.0, None)

    # ── Blowdown curves ──────────────────────────────────────────────────────

    def blowdown_curve(self, mdot, t_burn, n_pts=200):
        """
        Pressure-vs-time for constant mass-flow rate.

        Parameters
        ----------
        mdot   : float  mass flow rate [kg/s]
        t_burn : float  total burn time [s]
        n_pts  : int    output resolution

        Returns
        -------
        dict with keys: t [s], P [Pa], m_prop [kg], V_ullage [m^3]
        """
        t = np.linspace(0, t_burn, n_pts)
        m_exp = mdot * t
        m_exp = np.clip(m_exp, 0, self.m_prop_0)
        P = self.pressure(m_exp)
        m_prop = self.remaining_mass(m_exp)
        V_ull = self.V_ullage_0 + m_exp / self.rho_prop
        return dict(t=t, P=P, m_prop=m_prop, V_ullage=V_ull)

    def blowdown_coupled(self, mdot_func, t_burn, dt=0.01):
        """
        Blowdown with time-varying mass flow (coupled to downstream pressure).

        Parameters
        ----------
        mdot_func : callable   mdot = mdot_func(t, P_tank)  [kg/s]
        t_burn    : float      burn duration [s]
        dt        : float      output time step [s]

        Returns
        -------
        dict with keys: t, P, m_prop, mdot (all ndarray)
        """
        def rhs(t, y):
            m_exp = y[0]
            if m_exp >= self.m_prop_0:
                return [0.0]
            P = float(self.pressure(m_exp))
            md = mdot_func(t, P)
            return [md]

        t_span = (0.0, t_burn)
        t_eval = np.arange(0, t_burn + dt, dt)
        sol = solve_ivp(rhs, t_span, [0.0], t_eval=t_eval,
                        max_step=dt, rtol=1e-8, atol=1e-10)
        m_exp = sol.y[0]
        P = self.pressure(m_exp)
        mdots = np.array([mdot_func(t, p) for t, p in zip(sol.t, P)])
        return dict(t=sol.t, P=P, m_prop=self.remaining_mass(m_exp), mdot=mdots)

    def __repr__(self):
        return (f"Tank(V={self.V_tank:.4f}m3  P0={self.P_0/1e6:.2f}MPa  "
                f"{self.propellant} rho={self.rho_prop:.0f}kg/m3  "
                f"pres={self.pressurant_name} n={self.n:.3f}  "
                f"m_prop={self.m_prop_0:.2f}kg)")


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = Tank(V_tank=0.1, P_0=8e6, propellant="LOX", pressurant="He",
             ullage_fraction=0.05)
    print(t)
    print(f"  m_prop_0  = {t.m_prop_0:.2f} kg")
    print(f"  m_ullage  = {t.m_ullage:.4f} kg")

    bd = t.blowdown_curve(mdot=0.5, t_burn=t.m_prop_0/0.5)
    print(f"\n  Blowdown (constant 0.5 kg/s):")
    print(f"    P_start = {bd['P'][0]/1e6:.2f} MPa")
    print(f"    P_end   = {bd['P'][-1]/1e6:.2f} MPa")
    print(f"    t_burn  = {bd['t'][-1]:.1f} s")
