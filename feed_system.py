"""
feed_system.py — Feed system pressure drop and engine cycle balance

Models:
  - Pipe friction losses (Darcy-Weisbach + Swamee-Jain)
  - Fitting and valve K-factor losses
  - Gas generator cycle power balance
  - Expander cycle power balance
  - Complete feed system pressure budget

All units SI.
"""

import numpy as np
from turbopump import Pump, Turbine, G

try:
    from injector import PROPELLANT_DENSITIES
except ImportError:
    PROPELLANT_DENSITIES = {}

# ── Standard fitting K-factors ────────────────────────────────────────────────

FITTING_K = {
    "elbow_90":    0.9,
    "elbow_45":    0.4,
    "tee_branch":  1.8,
    "tee_run":     0.4,
    "ball_valve":  0.05,
    "gate_valve":  0.10,
    "check_valve": 2.0,
    "orifice":     2.5,
    "entrance":    0.5,
    "exit":        1.0,
}

# Approximate viscosities for common propellants [Pa*s]
PROPELLANT_VISCOSITY = {
    "LOX":  1.9e-4,   "O2":  1.9e-4,
    "LH2":  1.3e-5,   "H2":  1.3e-5,
    "CH4":  1.2e-4,
    "RP1":  1.2e-3,
    "N2O4": 4.2e-4,
    "UDMH": 5.0e-4,
    "N2":   1.6e-4,
}

# Approximate vapor pressures at storage temp [Pa]
PROPELLANT_VAPOR_P = {
    "LOX":  1.0e5,   "O2":  1.0e5,
    "LH2":  1.0e5,   "H2":  1.0e5,
    "CH4":  1.0e5,
    "RP1":  2.0e3,
    "N2O4": 9.6e4,
    "UDMH": 2.0e4,
}


# ── Pipe friction ─────────────────────────────────────────────────────────────

def friction_factor(Re, roughness, D):
    """
    Darcy friction factor via Swamee-Jain (explicit Colebrook).

    Parameters
    ----------
    Re        : float  Reynolds number
    roughness : float  surface roughness [m]
    D         : float  pipe inner diameter [m]

    Returns
    -------
    f : float  Darcy friction factor [-]
    """
    if Re < 2300:
        return 64.0 / max(Re, 1.0)
    eps_D = roughness / D
    return 0.25 / (np.log10(eps_D / 3.7 + 5.74 / Re ** 0.9)) ** 2


def pipe_pressure_drop(mdot, rho, mu, D, L, roughness=4.5e-5):
    """
    Friction pressure drop in a straight pipe [Pa].

    Parameters
    ----------
    mdot      : float  mass flow rate [kg/s]
    rho       : float  fluid density [kg/m^3]
    mu        : float  dynamic viscosity [Pa*s]
    D         : float  pipe inner diameter [m]
    L         : float  pipe length [m]
    roughness : float  surface roughness [m]

    Returns
    -------
    dP : float  pressure drop [Pa]
    """
    A = np.pi * (D / 2.0) ** 2
    v = mdot / (rho * A)
    Re = rho * v * D / max(mu, 1e-12)
    f = friction_factor(Re, roughness, D)
    return f * (L / D) * (rho * v ** 2 / 2.0)


def fitting_pressure_drop(mdot, rho, D, fittings):
    """
    Pressure drop through fittings/valves (K-factor method).

    Parameters
    ----------
    mdot     : float  mass flow rate [kg/s]
    rho      : float  fluid density [kg/m^3]
    D        : float  pipe diameter [m]
    fittings : list   fitting names from FITTING_K, or (name, count) tuples

    Returns
    -------
    dP : float  total fitting pressure drop [Pa]
    """
    A = np.pi * (D / 2.0) ** 2
    v = mdot / (rho * A)
    q = rho * v ** 2 / 2.0

    K_total = 0.0
    for item in fittings:
        if isinstance(item, tuple):
            name, count = item
            K_total += FITTING_K.get(name, 0) * count
        else:
            K_total += FITTING_K.get(item, 0)
    return K_total * q


def line_pressure_drop(mdot, rho, mu, D, L, roughness=4.5e-5, fittings=None):
    """
    Total pressure drop for a feed line (pipe + fittings).

    Returns
    -------
    dict: dP_friction, dP_fittings, dP_total [Pa], velocity [m/s], Re
    """
    dP_f = pipe_pressure_drop(mdot, rho, mu, D, L, roughness)
    dP_fit = 0.0
    if fittings:
        dP_fit = fitting_pressure_drop(mdot, rho, D, fittings)
    A = np.pi * (D / 2.0) ** 2
    v = mdot / (rho * A)
    Re = rho * v * D / max(mu, 1e-12)
    return dict(dP_friction=dP_f, dP_fittings=dP_fit,
                dP_total=dP_f + dP_fit, velocity=v, Re=Re)


# ── Feed line class ───────────────────────────────────────────────────────────

class FeedLine:
    """
    Complete feed line from tank to injector.

    Parameters
    ----------
    D          : float  pipe inner diameter [m]
    L          : float  pipe length [m]
    roughness  : float  surface roughness [m]  (default 4.5e-5 for drawn tubing)
    fittings   : list   fitting descriptors
    propellant : str    propellant name (density/viscosity lookup)
    rho        : float  override density [kg/m^3]
    mu         : float  override viscosity [Pa*s]
    """

    def __init__(self, D=0.02, L=2.0, roughness=4.5e-5,
                 fittings=None, propellant="LOX",
                 rho=None, mu=None):
        self.D = D
        self.L = L
        self.roughness = roughness
        self.fittings = fittings or [("elbow_90", 2), "ball_valve", "check_valve"]
        self.propellant = propellant
        self.rho = rho if rho is not None else PROPELLANT_DENSITIES.get(propellant, 800.0)
        self.mu = mu if mu is not None else PROPELLANT_VISCOSITY.get(propellant, 1e-3)

    def pressure_drop(self, mdot):
        """Compute total pressure drop [Pa] for mass flow mdot [kg/s]."""
        return line_pressure_drop(mdot, self.rho, self.mu,
                                  self.D, self.L, self.roughness,
                                  self.fittings)


# ── Engine cycle models ───────────────────────────────────────────────────────

class GasGeneratorCycle:
    """
    Gas-generator engine cycle model.

    A small gas generator burns a fraction of propellant (fuel-rich) to drive
    the turbine that powers both pumps.  The bleed flow is dumped overboard
    (or into the nozzle extension) and does not contribute to main-chamber Isp.

    Parameters
    ----------
    pump_ox, pump_fuel : Pump instances
    turbine            : Turbine instance
    feed_ox, feed_fuel : FeedLine instances
    T_gg               : float  gas generator temperature [K]  (800-1100 typical)
    gamma_gg           : float  GG exhaust gamma
    cp_gg              : float  GG exhaust cp [J/(kg*K)]
    """

    def __init__(self, pump_ox, pump_fuel, turbine,
                 feed_ox, feed_fuel,
                 T_gg=900.0, gamma_gg=1.3, cp_gg=2000.0):
        self.pump_ox = pump_ox
        self.pump_fuel = pump_fuel
        self.turbine = turbine
        self.feed_ox = feed_ox
        self.feed_fuel = feed_fuel
        self.T_gg = T_gg
        self.gamma_gg = gamma_gg
        self.cp_gg = cp_gg

    def solve(self, P_tank_ox, P_tank_fuel, P_chamber, mdot_total, OF):
        """
        Solve GG cycle balance.

        Parameters
        ----------
        P_tank_ox, P_tank_fuel : float  tank pressures [Pa]
        P_chamber              : float  chamber pressure [Pa]
        mdot_total             : float  total propellant mass flow [kg/s]
        OF                     : float  mixture ratio

        Returns
        -------
        dict with cycle balance results
        """
        mdot_ox = mdot_total * OF / (1.0 + OF)
        mdot_fuel = mdot_total / (1.0 + OF)

        rho_ox = self.feed_ox.rho
        rho_fuel = self.feed_fuel.rho

        # Line losses
        line_ox = self.feed_ox.pressure_drop(mdot_ox)
        line_fuel = self.feed_fuel.pressure_drop(mdot_fuel)

        # Required pump head: pump must raise pressure from tank to
        # chamber + injector dP + line losses
        dP_injector_ox = 0.15 * P_chamber    # ~15% of Pc rule of thumb
        dP_injector_fuel = 0.15 * P_chamber

        P_pump_out_ox = P_chamber + dP_injector_ox + line_ox['dP_total']
        P_pump_out_fuel = P_chamber + dP_injector_fuel + line_fuel['dP_total']

        dP_pump_ox = max(P_pump_out_ox - P_tank_ox, 0)
        dP_pump_fuel = max(P_pump_out_fuel - P_tank_fuel, 0)

        Q_ox = mdot_ox / rho_ox
        Q_fuel = mdot_fuel / rho_fuel

        # Pump powers
        H_ox = dP_pump_ox / (rho_ox * G)
        H_fuel = dP_pump_fuel / (rho_fuel * G)

        # Override pump design point to match
        self.pump_ox.H_design = max(H_ox, 10)
        self.pump_ox.Q_design = Q_ox
        self.pump_fuel.H_design = max(H_fuel, 10)
        self.pump_fuel.Q_design = Q_fuel

        P_shaft_ox = self.pump_ox.power(Q_ox, rho_ox)
        P_shaft_fuel = self.pump_fuel.power(Q_fuel, rho_fuel)
        P_total_pumps = P_shaft_ox + P_shaft_fuel

        # Turbine: GG exhaust drives turbine, exhausts at ~ambient
        P_turb_in = P_chamber * 0.9  # GG operates slightly below Pc
        P_turb_out = max(P_chamber * 0.05, 1e5)  # dump pressure

        mdot_gg = self.turbine.required_mdot(
            P_total_pumps, self.cp_gg, self.T_gg,
            P_turb_in, P_turb_out, self.gamma_gg)
        bleed = mdot_gg / mdot_total

        # NPSH checks
        P_vapor_ox = PROPELLANT_VAPOR_P.get(self.feed_ox.propellant, 1e5)
        P_vapor_fuel = PROPELLANT_VAPOR_P.get(self.feed_fuel.propellant, 1e5)
        npsh_ox = self.pump_ox.check_npsh(P_tank_ox, P_vapor_ox, rho_ox)
        npsh_fuel = self.pump_fuel.check_npsh(P_tank_fuel, P_vapor_fuel, rho_fuel)

        return dict(
            mdot_gg=mdot_gg,
            mdot_main=mdot_total - mdot_gg,
            bleed_fraction=bleed,
            dP_ox_line=line_ox['dP_total'],
            dP_fuel_line=line_fuel['dP_total'],
            dP_ox_pump=dP_pump_ox,
            dP_fuel_pump=dP_pump_fuel,
            P_ox_injector=P_tank_ox + dP_pump_ox - line_ox['dP_total'],
            P_fuel_injector=P_tank_fuel + dP_pump_fuel - line_fuel['dP_total'],
            P_pump_ox=P_shaft_ox,
            P_pump_fuel=P_shaft_fuel,
            P_turbine=P_total_pumps,
            NPSH_ox=npsh_ox,
            NPSH_fuel=npsh_fuel,
        )


class ExpanderCycle:
    """
    Expander engine cycle model.

    Turbine driven by heated fuel from regenerative cooling jacket.
    No gas generator — all propellant flows through main chamber.

    Parameters
    ----------
    pump_ox, pump_fuel : Pump
    turbine            : Turbine
    feed_ox, feed_fuel : FeedLine
    """

    def __init__(self, pump_ox, pump_fuel, turbine,
                 feed_ox, feed_fuel):
        self.pump_ox = pump_ox
        self.pump_fuel = pump_fuel
        self.turbine = turbine
        self.feed_ox = feed_ox
        self.feed_fuel = feed_fuel

    def solve(self, P_tank_ox, P_tank_fuel, P_chamber, mdot_total, OF,
              T_cool_exit, P_cool_exit, cp_cool, gamma_cool):
        """
        Solve expander cycle.

        Parameters
        ----------
        T_cool_exit  : float [K]   coolant temp at cooling jacket exit
        P_cool_exit  : float [Pa]  coolant pressure at jacket exit
        cp_cool      : float [J/(kg*K)]
        gamma_cool   : float       coolant gamma at turbine conditions
        (other params same as GasGeneratorCycle)

        Returns
        -------
        dict with cycle results, including feasibility check
        """
        mdot_ox = mdot_total * OF / (1.0 + OF)
        mdot_fuel = mdot_total / (1.0 + OF)

        rho_ox = self.feed_ox.rho
        rho_fuel = self.feed_fuel.rho

        line_ox = self.feed_ox.pressure_drop(mdot_ox)
        line_fuel = self.feed_fuel.pressure_drop(mdot_fuel)

        dP_injector_ox = 0.15 * P_chamber
        dP_injector_fuel = 0.15 * P_chamber

        P_pump_out_ox = P_chamber + dP_injector_ox + line_ox['dP_total']
        P_pump_out_fuel = P_chamber + dP_injector_fuel + line_fuel['dP_total']

        dP_pump_ox = max(P_pump_out_ox - P_tank_ox, 0)
        dP_pump_fuel = max(P_pump_out_fuel - P_tank_fuel, 0)

        Q_ox = mdot_ox / rho_ox
        Q_fuel = mdot_fuel / rho_fuel

        self.pump_ox.H_design = max(dP_pump_ox / (rho_ox * G), 10)
        self.pump_ox.Q_design = Q_ox
        self.pump_fuel.H_design = max(dP_pump_fuel / (rho_fuel * G), 10)
        self.pump_fuel.Q_design = Q_fuel

        P_shaft_ox = self.pump_ox.power(Q_ox, rho_ox)
        P_shaft_fuel = self.pump_fuel.power(Q_fuel, rho_fuel)
        P_total_pumps = P_shaft_ox + P_shaft_fuel

        # Turbine driven by heated coolant (fuel)
        P_turb_in = P_cool_exit
        P_turb_out = P_chamber * 1.05  # must exceed Pc to inject

        P_turb = self.turbine.power(
            mdot_fuel, cp_cool, T_cool_exit,
            P_turb_in, P_turb_out, gamma_cool)

        feasible = P_turb >= P_total_pumps

        P_vapor_ox = PROPELLANT_VAPOR_P.get(self.feed_ox.propellant, 1e5)
        P_vapor_fuel = PROPELLANT_VAPOR_P.get(self.feed_fuel.propellant, 1e5)
        npsh_ox = self.pump_ox.check_npsh(P_tank_ox, P_vapor_ox, rho_ox)
        npsh_fuel = self.pump_fuel.check_npsh(P_tank_fuel, P_vapor_fuel, rho_fuel)

        return dict(
            mdot_gg=0.0,
            mdot_main=mdot_total,
            bleed_fraction=0.0,
            dP_ox_line=line_ox['dP_total'],
            dP_fuel_line=line_fuel['dP_total'],
            dP_ox_pump=dP_pump_ox,
            dP_fuel_pump=dP_pump_fuel,
            P_ox_injector=P_tank_ox + dP_pump_ox - line_ox['dP_total'],
            P_fuel_injector=P_tank_fuel + dP_pump_fuel - line_fuel['dP_total'],
            P_pump_ox=P_shaft_ox,
            P_pump_fuel=P_shaft_fuel,
            P_turbine=P_turb,
            turbine_inlet_T=T_cool_exit,
            turbine_inlet_P=P_turb_in,
            feasible=feasible,
            power_margin=P_turb - P_total_pumps,
            NPSH_ox=npsh_ox,
            NPSH_fuel=npsh_fuel,
        )


# ── Pressure budget ──────────────────────────────────────────────────────────

def pressure_budget(P_tank, dP_line, dP_pump, dP_injector, P_chamber):
    """
    Verify feed system pressure budget.

    P_tank + dP_pump - dP_line - dP_injector >= P_chamber

    Returns
    -------
    dict: margin [Pa], ratio [-]
    """
    supply = P_tank + dP_pump
    demand = P_chamber + dP_line + dP_injector
    return dict(margin=supply - demand, ratio=supply / max(demand, 1.0))


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Feed system demo — LOX/H2 gas generator cycle\n")

    feed_ox = FeedLine(D=0.025, L=2.5, propellant="LOX",
                       fittings=[("elbow_90", 3), "ball_valve", "check_valve"])
    feed_fuel = FeedLine(D=0.020, L=3.0, propellant="H2",
                         fittings=[("elbow_90", 2), "ball_valve", "check_valve"])

    pump_o = Pump(eta_design=0.70)
    pump_f = Pump(eta_design=0.65)
    turb = Turbine(eta_turbine=0.60)

    gg = GasGeneratorCycle(pump_o, pump_f, turb, feed_ox, feed_fuel,
                           T_gg=900, gamma_gg=1.3, cp_gg=2500)

    result = gg.solve(P_tank_ox=0.3e6, P_tank_fuel=0.3e6,
                      P_chamber=7e6, mdot_total=1.0, OF=6.0)

    print(f"  GG bleed fraction = {result['bleed_fraction']*100:.2f}%")
    print(f"  mdot_gg           = {result['mdot_gg']:.4f} kg/s")
    print(f"  mdot_main         = {result['mdot_main']:.4f} kg/s")
    print(f"  Pump power (ox)   = {result['P_pump_ox']/1e3:.1f} kW")
    print(f"  Pump power (fuel) = {result['P_pump_fuel']/1e3:.1f} kW")
    print(f"  dP ox line        = {result['dP_ox_line']/1e6:.3f} MPa")
    print(f"  dP fuel line      = {result['dP_fuel_line']/1e6:.3f} MPa")
    print(f"  NPSH ox margin    = {result['NPSH_ox']['margin']:.2f}")
    print(f"  NPSH fuel margin  = {result['NPSH_fuel']['margin']:.2f}")

    # Pressure budget check
    pb = pressure_budget(0.3e6, result['dP_ox_line'],
                         result['dP_ox_pump'], 0.15*7e6, 7e6)
    print(f"\n  Pressure budget ratio = {pb['ratio']:.3f}")
