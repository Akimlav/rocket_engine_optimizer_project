"""
turbopump.py — Turbopump performance model

Models centrifugal pump head-rise, efficiency, required shaft power,
gas turbine power balance, and NPSH cavitation check.

For gas-generator and expander cycle engine models.  All units SI.
"""

import numpy as np

G = 9.80665  # standard gravity [m/s^2]


class Pump:
    """
    Centrifugal pump with parabolic head-flow characteristic.

    Parameters
    ----------
    H_design      : float  design-point head [m]  (= dP/(rho*g))
    Q_design      : float  design-point volumetric flow [m^3/s]
    eta_design    : float  design-point isentropic efficiency [-]  (0.6-0.8)
    k_H           : float  head droop coefficient [-]   (default 0.3)
    NPSH_required : float  required net positive suction head [m]  (default 10)
    """

    def __init__(self, H_design=500.0, Q_design=0.01, eta_design=0.70,
                 k_H=0.3, NPSH_required=10.0):
        self.H_design = H_design
        self.Q_design = Q_design
        self.eta_design = eta_design
        self.k_H = k_H
        self.NPSH_required = NPSH_required

    def head(self, Q):
        """
        Pump head at volumetric flow Q.

        H(Q) = H_design * [1 - k_H * (Q/Q_design - 1)^2]

        Parameters
        ----------
        Q : float  volumetric flow rate [m^3/s]

        Returns
        -------
        H : float  pump head [m]
        """
        return self.H_design * (1.0 - self.k_H * ((Q / self.Q_design) - 1.0) ** 2)

    def pressure_rise(self, Q, rho):
        """
        Pressure rise dP [Pa] at flow Q for fluid of density rho [kg/m^3].
        """
        return rho * G * self.head(Q)

    def efficiency(self, Q):
        """
        Pump efficiency at off-design flow (parabolic falloff).

        eta(Q) = eta_design * [1 - 0.5*(Q/Q_design - 1)^2]
        """
        return self.eta_design * (1.0 - 0.5 * ((Q / self.Q_design) - 1.0) ** 2)

    def power(self, Q, rho):
        """
        Required shaft power [W].

        P = rho * g * Q * H(Q) / eta(Q)
        """
        eta = max(self.efficiency(Q), 0.1)
        return rho * G * Q * self.head(Q) / eta

    def check_npsh(self, P_inlet, P_vapor, rho, z_head=0.0, h_loss=0.0):
        """
        Check Net Positive Suction Head margin.

        Parameters
        ----------
        P_inlet  : float  pump inlet static pressure [Pa]
        P_vapor  : float  propellant vapor pressure [Pa]
        rho      : float  propellant density [kg/m^3]
        z_head   : float  elevation head above pump [m]
        h_loss   : float  suction-line friction head loss [m]

        Returns
        -------
        dict:
            NPSH_available : float [m]
            NPSH_required  : float [m]
            margin         : float [-]  (> 1 = OK)
        """
        NPSH_a = (P_inlet - P_vapor) / (rho * G) + z_head - h_loss
        return dict(NPSH_available=NPSH_a,
                    NPSH_required=self.NPSH_required,
                    margin=NPSH_a / max(self.NPSH_required, 1e-6))

    def __repr__(self):
        return (f"Pump(H={self.H_design:.0f}m  Q={self.Q_design:.4f}m3/s  "
                f"eta={self.eta_design:.2f})")


class Turbine:
    """
    Gas turbine model for turbopump drive.

    Parameters
    ----------
    eta_turbine : float  isentropic efficiency [-]  (0.5-0.8)
    P_mech_loss : float  mechanical losses [W]  (bearings, seals, etc.)
    """

    def __init__(self, eta_turbine=0.65, P_mech_loss=0.0):
        self.eta = eta_turbine
        self.P_mech_loss = P_mech_loss

    def power(self, mdot, cp, T_in, P_in, P_out, gamma):
        """
        Turbine shaft power output [W].

        P_shaft = mdot * cp * T_in * eta * [1 - (P_out/P_in)^((gamma-1)/gamma)]
                  - P_mech_loss

        Parameters
        ----------
        mdot   : float  drive-gas mass flow [kg/s]
        cp     : float  specific heat of drive gas [J/(kg*K)]
        T_in   : float  turbine inlet temperature [K]
        P_in   : float  turbine inlet pressure [Pa]
        P_out  : float  turbine exhaust pressure [Pa]
        gamma  : float  ratio of specific heats

        Returns
        -------
        P_shaft : float  net shaft power [W]
        """
        pr = max(P_out / P_in, 1e-6)
        ideal = mdot * cp * T_in * (1.0 - pr ** ((gamma - 1.0) / gamma))
        return ideal * self.eta - self.P_mech_loss

    def required_mdot(self, P_required, cp, T_in, P_in, P_out, gamma):
        """
        Drive-gas mass flow rate needed to deliver P_required [W] shaft power.

        Returns
        -------
        mdot : float [kg/s]
        """
        pr = max(P_out / P_in, 1e-6)
        ideal_per_kg = cp * T_in * (1.0 - pr ** ((gamma - 1.0) / gamma))
        net_per_kg = ideal_per_kg * self.eta
        return (P_required + self.P_mech_loss) / max(net_per_kg, 1.0)

    def __repr__(self):
        return f"Turbine(eta={self.eta:.2f}  mech_loss={self.P_mech_loss:.0f}W)"


def turbopump_balance(pump_ox, pump_fuel, turbine,
                      mdot_ox, rho_ox, mdot_fuel, rho_fuel,
                      turbine_cp, T_turbine_in,
                      P_turbine_in, P_turbine_out, gamma_turbine):
    """
    Solve turbopump power balance:
      P_turbine = P_pump_ox + P_pump_fuel

    Parameters
    ----------
    pump_ox, pump_fuel : Pump instances
    turbine            : Turbine instance
    mdot_ox            : float  oxidizer mass flow [kg/s]
    rho_ox             : float  oxidizer density [kg/m^3]
    mdot_fuel          : float  fuel mass flow [kg/s]
    rho_fuel           : float  fuel density [kg/m^3]
    turbine_cp         : float  turbine drive gas cp [J/(kg*K)]
    T_turbine_in       : float  turbine inlet temperature [K]
    P_turbine_in       : float  turbine inlet pressure [Pa]
    P_turbine_out      : float  turbine exhaust pressure [Pa]
    gamma_turbine      : float  drive gas gamma

    Returns
    -------
    dict with P_pump_ox, P_pump_fuel, P_turbine [W],
         mdot_turbine [kg/s], dP_ox/dP_fuel [Pa], balance_error [W]
    """
    Q_ox = mdot_ox / rho_ox
    Q_fuel = mdot_fuel / rho_fuel
    P_pump_ox = pump_ox.power(Q_ox, rho_ox)
    P_pump_fuel = pump_fuel.power(Q_fuel, rho_fuel)
    P_total = P_pump_ox + P_pump_fuel

    mdot_turb = turbine.required_mdot(
        P_total, turbine_cp, T_turbine_in,
        P_turbine_in, P_turbine_out, gamma_turbine)
    P_turb = turbine.power(
        mdot_turb, turbine_cp, T_turbine_in,
        P_turbine_in, P_turbine_out, gamma_turbine)

    return dict(
        P_pump_ox=P_pump_ox,
        P_pump_fuel=P_pump_fuel,
        P_turbine=P_turb,
        mdot_turbine=mdot_turb,
        dP_ox=pump_ox.pressure_rise(Q_ox, rho_ox),
        dP_fuel=pump_fuel.pressure_rise(Q_fuel, rho_fuel),
        balance_error=P_turb - P_total,
    )


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # LOX/H2 example
    rho_lox = 1141.0; rho_lh2 = 71.0
    mdot_o = 0.8; mdot_f = 0.15
    Q_o = mdot_o / rho_lox; Q_f = mdot_f / rho_lh2

    pump_o = Pump(H_design=600, Q_design=Q_o, eta_design=0.70)
    pump_f = Pump(H_design=1200, Q_design=Q_f, eta_design=0.65)
    turb = Turbine(eta_turbine=0.60)

    bal = turbopump_balance(
        pump_o, pump_f, turb,
        mdot_o, rho_lox, mdot_f, rho_lh2,
        turbine_cp=2500, T_turbine_in=900,
        P_turbine_in=7e6, P_turbine_out=0.5e6,
        gamma_turbine=1.3)

    print("Turbopump balance:")
    for k, v in bal.items():
        if 'dP' in k:
            print(f"  {k} = {v/1e6:.3f} MPa")
        elif 'P_' in k:
            print(f"  {k} = {v/1e3:.1f} kW")
        else:
            print(f"  {k} = {v:.4f}")
