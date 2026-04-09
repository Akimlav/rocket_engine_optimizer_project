"""
injector.py — Incompressible injector orifice model

Models a coaxial / showerhead injector as two sets of orifices
(fuel and oxidizer) using Bernoulli discharge equations.

  mdot = Cd * A * sqrt(2 * rho * dP)

Propellant density presets available via PROPELLANT_DENSITIES.
"""

import numpy as np

# Liquid propellant densities at typical storage conditions [kg/m^3]
PROPELLANT_DENSITIES = {
    "LH2":   71.0,    # liquid hydrogen (20 K)
    "H2":    71.0,    # alias
    "LOX":   1141.0,  # liquid oxygen (90 K)
    "O2":    1141.0,  # alias
    "RP1":   820.0,   # kerosene / RP-1
    "CH4":   424.0,   # liquid methane (111 K)
    "N2O4":  1443.0,  # nitrogen tetroxide
    "UDMH":  791.0,   # unsymmetrical dimethylhydrazine
    "N2":    807.0,   # liquid nitrogen (77 K)
    "H2O2":  1450.0,  # 90% HTP
}


class Injector:
    def __init__(self,
                 Cd_f=0.8, Cd_o=0.8,
                 A_f=1e-4, A_o=1e-4,
                 rho_f=None, rho_o=None,
                 fuel="LH2", oxidizer="LOX"):
        """
        Parameters
        ----------
        Cd_f, Cd_o   : discharge coefficients [-]  (0.6–0.9 typical)
        A_f, A_o     : total orifice area [m^2]
        rho_f, rho_o : densities [kg/m^3]; if None, looked up from fuel/oxidizer names
        fuel         : propellant name string (for density lookup)
        oxidizer     : propellant name string (for density lookup)
        """
        self.Cd_f = Cd_f
        self.Cd_o = Cd_o
        self.A_f  = A_f
        self.A_o  = A_o
        self.fuel     = fuel
        self.oxidizer = oxidizer
        self.rho_f = rho_f if rho_f is not None else PROPELLANT_DENSITIES.get(fuel, 800.0)
        self.rho_o = rho_o if rho_o is not None else PROPELLANT_DENSITIES.get(oxidizer, 1140.0)

    def mass_flow(self, p_tank_f, p_tank_o, p_c):
        """
        Compute fuel and oxidizer mass flow rates.

        Parameters
        ----------
        p_tank_f : fuel tank pressure [Pa]
        p_tank_o : oxidizer tank pressure [Pa]
        p_c      : chamber pressure [Pa]

        Returns
        -------
        mdot_f, mdot_o : mass flow rates [kg/s]
        """
        dp_f = max(p_tank_f - p_c, 1.0)
        dp_o = max(p_tank_o - p_c, 1.0)
        mdot_f = self.Cd_f * self.A_f * np.sqrt(2.0 * self.rho_f * dp_f)
        mdot_o = self.Cd_o * self.A_o * np.sqrt(2.0 * self.rho_o * dp_o)
        return mdot_f, mdot_o

    def total_mdot(self, p_tank_f, p_tank_o, p_c):
        mdot_f, mdot_o = self.mass_flow(p_tank_f, p_tank_o, p_c)
        return mdot_f + mdot_o

    def OF_ratio(self, p_tank_f, p_tank_o, p_c):
        mdot_f, mdot_o = self.mass_flow(p_tank_f, p_tank_o, p_c)
        return mdot_o / max(mdot_f, 1e-12)

    def __repr__(self):
        return (f"Injector(fuel={self.fuel} rho={self.rho_f:.0f}kg/m³ "
                f"Cd_f={self.Cd_f} A_f={self.A_f:.2e}m²  "
                f"ox={self.oxidizer} rho={self.rho_o:.0f}kg/m³ "
                f"Cd_o={self.Cd_o} A_o={self.A_o:.2e}m²)")


if __name__ == "__main__":
    inj = Injector(fuel="LH2", oxidizer="LOX", A_f=2e-4, A_o=2e-4)
    mf, mo = inj.mass_flow(8e6, 8e6, 6e6)
    print(f"mdot_f={mf:.4f} kg/s  mdot_o={mo:.4f} kg/s  OF={mo/mf:.3f}")
    print(inj)
