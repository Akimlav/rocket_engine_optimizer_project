"""
cooling.py — Regenerative cooling loop model

Computes wall temperature profile along a rocket engine chamber/nozzle
using:
  - Bartz correlation for gas-side heat transfer coefficient (h_g)
  - Dittus-Boelter correlation for coolant-side heat transfer coefficient (h_c)
  - 1D steady-state radial heat balance through the wall

Coolant marches from nozzle exit (cold inlet) back toward the injector,
picking up heat along the way.

All units SI.
"""

import numpy as np
from materials import thermal_conductivity, tensile_strength
from nozzle_analysis import area_ratio, mach_from_area_ratio_local

# ── Coolant property database ─────────────────────────────────────────────────

_a = np.array

# Simplified temperature-dependent properties for common coolants.
# For each: (T_array_K, property_array), linearly interpolated.
COOLANT_DATA = {
    "H2": dict(
        description="Gaseous/supercritical hydrogen (at ~10 MPa)",
        rho_data=(
            _a([  20,   40,   80,  150,  250,  400,  600,  800, 1000]),
            _a([ 71.0, 40.0, 16.0,  6.5,  3.9,  2.5,  1.65, 1.25, 1.00]),
        ),
        cp_data=(
            _a([  20,   40,   80,  150,  250,  400,  600,  800, 1000]),
            _a([9700, 12000, 14500, 14800, 14500, 14400, 14500, 14600, 14800]),
        ),
        mu_data=(
            _a([  20,   40,   80,  150,  250,  400,  600,  800, 1000]),
            _a([1.3e-6, 2.5e-6, 4.5e-6, 6.5e-6, 8.5e-6, 1.1e-5, 1.4e-5, 1.7e-5, 2.0e-5]),
        ),
        k_data=(
            _a([  20,   40,   80,  150,  250,  400,  600,  800, 1000]),
            _a([0.020, 0.040, 0.080, 0.120, 0.160, 0.210, 0.270, 0.320, 0.370]),
        ),
        T_boil=20.3,
    ),
    "CH4": dict(
        description="Liquid/supercritical methane (at ~10 MPa)",
        rho_data=(
            _a([ 111,  150,  200,  300,  400,  500,  600]),
            _a([ 424,  380,  280,   80,   50,   38,   30]),
        ),
        cp_data=(
            _a([ 111,  150,  200,  300,  400,  500,  600]),
            _a([3500, 3800, 5500, 3200, 2800, 2700, 2700]),
        ),
        mu_data=(
            _a([ 111,  150,  200,  300,  400,  500,  600]),
            _a([1.2e-4, 7.0e-5, 3.0e-5, 1.8e-5, 2.0e-5, 2.2e-5, 2.4e-5]),
        ),
        k_data=(
            _a([ 111,  150,  200,  300,  400,  500,  600]),
            _a([0.200, 0.170, 0.110, 0.050, 0.060, 0.075, 0.090]),
        ),
        T_boil=111.7,
    ),
    "RP1": dict(
        description="RP-1 kerosene",
        rho_data=(
            _a([ 250,  300,  350,  400,  500,  600,  700]),
            _a([ 840,  820,  790,  750,  650,  520,  380]),
        ),
        cp_data=(
            _a([ 250,  300,  350,  400,  500,  600,  700]),
            _a([1900, 2010, 2130, 2260, 2550, 2900, 3300]),
        ),
        mu_data=(
            _a([ 250,  300,  350,  400,  500,  600,  700]),
            _a([2.5e-3, 1.2e-3, 6.5e-4, 4.0e-4, 2.0e-4, 1.2e-4, 8.0e-5]),
        ),
        k_data=(
            _a([ 250,  300,  350,  400,  500,  600,  700]),
            _a([0.120, 0.115, 0.110, 0.105, 0.095, 0.085, 0.075]),
        ),
        T_boil=490.0,
    ),
}


class CoolantProperties:
    """
    Coolant thermophysical properties (temperature-dependent, interpolated).

    Parameters
    ----------
    name : str   coolant name ('H2', 'CH4', 'RP1')
    """

    def __init__(self, name="H2"):
        if name not in COOLANT_DATA:
            raise ValueError(f"Unknown coolant '{name}'. Available: {list(COOLANT_DATA)}")
        self.name = name
        self._d = COOLANT_DATA[name]
        self.T_boil = self._d['T_boil']

    def rho(self, T):
        d = self._d['rho_data']
        return np.interp(T, d[0], d[1])

    def cp(self, T):
        d = self._d['cp_data']
        return np.interp(T, d[0], d[1])

    def mu(self, T):
        d = self._d['mu_data']
        return np.interp(T, d[0], d[1])

    def k(self, T):
        d = self._d['k_data']
        return np.interp(T, d[0], d[1])

    def Pr(self, T):
        return self.mu(T) * self.cp(T) / self.k(T)


# ── Cooling channel geometry ─────────────────────────────────────────────────

class CoolingChannelGeometry:
    """
    Rectangular cooling channel geometry.

    Parameters
    ----------
    n_channels     : int    number of channels around circumference
    width          : float  channel width [m]
    height         : float  channel height [m]  (radial depth)
    wall_thickness : float  gas-side wall thickness [m]
    land_width     : float  rib/land width between channels [m]
    roughness      : float  channel surface roughness [m]
    """

    def __init__(self, n_channels=100, width=1.5e-3, height=2.5e-3,
                 wall_thickness=1.0e-3, land_width=1.0e-3, roughness=1e-5):
        self.n_channels = n_channels
        self.width = width
        self.height = height
        self.wall_thickness = wall_thickness
        self.land_width = land_width
        self.roughness = roughness

    @property
    def hydraulic_diameter(self):
        """D_h = 4*A / P  [m]"""
        A = self.width * self.height
        P = 2.0 * (self.width + self.height)
        return 4.0 * A / P

    @property
    def flow_area(self):
        """Single-channel cross-section area [m^2]."""
        return self.width * self.height

    @property
    def total_flow_area(self):
        """Total flow area across all channels [m^2]."""
        return self.n_channels * self.flow_area


# ── Heat transfer correlations ────────────────────────────────────────────────

def bartz_h_g(T0, P0, gamma, Cstar, cp_gas, Pr_gas, mu_gas,
              Rt, R_curv, A_t, A_local, M_local, T_wg):
    """
    Gas-side heat transfer coefficient (Bartz, 1957).

    Parameters
    ----------
    T0       : float  stagnation temperature [K]
    P0       : float  chamber pressure [Pa]
    gamma    : float  ratio of specific heats
    Cstar    : float  characteristic velocity [m/s]
    cp_gas   : float  gas cp [J/(kg*K)]
    Pr_gas   : float  gas Prandtl number
    mu_gas   : float  gas dynamic viscosity [Pa*s]
    Rt       : float  throat radius [m]
    R_curv   : float  throat radius of curvature [m]
    A_t      : float  throat area [m^2]
    A_local  : float  local cross-section area [m^2]
    M_local  : float  local Mach number
    T_wg     : float  gas-side wall temperature [K] (for sigma correction)

    Returns
    -------
    h_g : float  [W/(m^2*K)]
    """
    D_t = 2.0 * Rt
    g = gamma

    # Bartz sigma correction
    Tw_T0 = T_wg / T0
    fac = 0.5 * Tw_T0 * (1.0 + (g - 1.0) / 2.0 * M_local ** 2) + 0.5
    sigma = fac ** (-0.68) * (1.0 + (g - 1.0) / 2.0 * M_local ** 2) ** (-0.12)

    h = (0.026 / D_t ** 0.2
         * (mu_gas ** 0.2 * cp_gas / Pr_gas ** 0.6)
         * (P0 / Cstar) ** 0.8
         * (D_t / R_curv) ** 0.1
         * (A_t / A_local) ** 0.9
         * sigma)
    return max(h, 1.0)  # floor to avoid numerical issues


def dittus_boelter_h_c(Re, Pr, k_cool, D_h):
    """
    Coolant-side heat transfer coefficient (Dittus-Boelter).

    Nu = 0.023 * Re^0.8 * Pr^0.4
    h_c = Nu * k / D_h

    Parameters
    ----------
    Re     : float  Reynolds number
    Pr     : float  Prandtl number
    k_cool : float  coolant thermal conductivity [W/(m*K)]
    D_h    : float  hydraulic diameter [m]

    Returns
    -------
    h_c : float  [W/(m^2*K)]
    """
    Nu = 0.023 * abs(Re) ** 0.8 * abs(Pr) ** 0.4
    return Nu * k_cool / D_h


def wall_temperature(h_g, h_c, k_wall, t_wall, T_aw, T_cool):
    """
    Solve 1D radial three-layer thermal resistance.

    q = (T_aw - T_cool) / (1/h_g + t_wall/k_wall + 1/h_c)

    Parameters
    ----------
    h_g    : float  gas-side HTC [W/(m^2*K)]
    h_c    : float  coolant-side HTC [W/(m^2*K)]
    k_wall : float  wall thermal conductivity [W/(m*K)]
    t_wall : float  wall thickness [m]
    T_aw   : float  adiabatic wall temperature [K]
    T_cool : float  coolant bulk temperature [K]

    Returns
    -------
    dict: q [W/m^2], T_wg [K], T_wc [K]
    """
    R_total = 1.0 / h_g + t_wall / k_wall + 1.0 / h_c
    q = (T_aw - T_cool) / R_total
    T_wg = T_aw - q / h_g
    T_wc = T_cool + q / h_c
    return dict(q=q, T_wg=T_wg, T_wc=T_wc)


# ── Main solver ───────────────────────────────────────────────────────────────

def solve_cooling_loop(engine_result, nozzle_result, wall_contour,
                       channel_geom, coolant, wall_material,
                       mdot_coolant, T_cool_inlet=None,
                       R_curv_factor=1.5, n_stations=100):
    """
    Solve the full regenerative cooling loop along the nozzle/chamber.

    Marches axially from nozzle exit back toward injector.
    At each station: Bartz h_g, Dittus-Boelter h_c, solve wall temps,
    update coolant temperature and pressure.

    Parameters
    ----------
    engine_result : dict   from EngineSolver.solve()
    nozzle_result : dict   from nozzle_perf()
    wall_contour  : ndarray (N,2) [x_m, r_m]
    channel_geom  : CoolingChannelGeometry
    coolant       : CoolantProperties
    wall_material : str    material name for materials.py
    mdot_coolant  : float  total coolant mass flow [kg/s]
    T_cool_inlet  : float  coolant inlet temperature [K]  (None = boiling point)
    R_curv_factor : float  throat curvature radius = factor * Rt
    n_stations    : int    number of axial stations

    Returns
    -------
    dict with arrays: x, r, T_wg, T_wc, T_cool, q, h_g, h_c, P_cool, M_local
         and scalars: total_heat [W], dP_total [Pa], T_cool_exit [K],
                      max_T_wg [K], margin (array of UTS/applied-stress ratios)
    """
    # Unpack engine state
    T0 = engine_result['T0']
    P0 = engine_result['pc']
    g  = engine_result['gamma']
    R  = engine_result['R']
    Cstar = engine_result['Cstar']

    # Gas transport properties (added by modified chamber.py)
    cp_gas  = engine_result.get('cp', R * g / (g - 1.0))
    mu_gas  = engine_result.get('mu_gas', 5e-5)   # fallback estimate
    Pr_gas  = engine_result.get('Pr_gas', 0.5)     # typical for combustion gases

    Rt = wall_contour[np.argmin(wall_contour[:, 1]), 1]  # throat radius
    At = np.pi * Rt ** 2
    R_curv = R_curv_factor * Rt

    if T_cool_inlet is None:
        T_cool_inlet = coolant.T_boil + 5.0

    # Interpolate contour to uniform axial stations
    x_raw = wall_contour[:, 0]
    r_raw = wall_contour[:, 1]
    x_sta = np.linspace(x_raw[-1], x_raw[0], n_stations)  # nozzle exit -> throat -> chamber
    r_sta = np.interp(x_sta, x_raw, r_raw)

    # Local area ratio and Mach at each station
    A_sta = np.pi * r_sta ** 2
    AR_sta = A_sta / At

    M_sta = np.zeros(n_stations)
    for i, (ar, x) in enumerate(zip(AR_sta, x_sta)):
        try:
            if x >= 0:
                M_sta[i] = mach_from_area_ratio_local(ar, g, supersonic=True)
            else:
                M_sta[i] = mach_from_area_ratio_local(ar, g, supersonic=False)
        except Exception:
            M_sta[i] = 1.0

    # Recovery factor and adiabatic wall temperature
    r_fac = Pr_gas ** (1.0 / 3.0)
    T_aw_sta = T0 * (1.0 + r_fac * (g - 1.0) / 2.0 * M_sta ** 2) / \
               (1.0 + (g - 1.0) / 2.0 * M_sta ** 2)

    # Allocate output arrays
    T_wg_arr  = np.zeros(n_stations)
    T_wc_arr  = np.zeros(n_stations)
    T_cool_arr = np.zeros(n_stations)
    q_arr     = np.zeros(n_stations)
    h_g_arr   = np.zeros(n_stations)
    h_c_arr   = np.zeros(n_stations)
    P_cool_arr = np.zeros(n_stations)

    # Channel geometry
    D_h = channel_geom.hydraulic_diameter
    A_ch = channel_geom.total_flow_area
    t_wall = channel_geom.wall_thickness
    roughness = channel_geom.roughness

    # Initial coolant state (at nozzle exit = coolant inlet)
    T_cool = T_cool_inlet
    P_cool = P0 * 1.2  # coolant enters at ~120% chamber pressure

    # March from nozzle exit toward injector
    for i in range(n_stations):
        A_local = A_sta[i]
        M_local = M_sta[i]
        T_aw = T_aw_sta[i]

        # Coolant state at this station
        rho_c = coolant.rho(T_cool)
        cp_c  = coolant.cp(T_cool)
        mu_c  = coolant.mu(T_cool)
        k_c   = coolant.k(T_cool)

        v_cool = mdot_coolant / (rho_c * A_ch) if A_ch > 0 else 0
        Re_c = rho_c * v_cool * D_h / max(mu_c, 1e-12)
        Pr_c = mu_c * cp_c / max(k_c, 1e-12)

        # Coolant-side HTC
        h_c = dittus_boelter_h_c(Re_c, Pr_c, k_c, D_h)

        # Gas-side HTC (Bartz) — iterate on T_wg
        T_wg_guess = 0.5 * T_aw  # initial guess
        for _iter in range(15):
            k_w = thermal_conductivity(wall_material, T_wg_guess)
            h_g = bartz_h_g(T0, P0, g, Cstar, cp_gas, Pr_gas, mu_gas,
                            Rt, R_curv, At, A_local, M_local, T_wg_guess)
            wt = wall_temperature(h_g, h_c, k_w, t_wall, T_aw, T_cool)
            if abs(wt['T_wg'] - T_wg_guess) < 0.5:
                break
            T_wg_guess = 0.7 * T_wg_guess + 0.3 * wt['T_wg']

        T_wg_arr[i]   = wt['T_wg']
        T_wc_arr[i]   = wt['T_wc']
        T_cool_arr[i] = T_cool
        q_arr[i]      = wt['q']
        h_g_arr[i]    = h_g
        h_c_arr[i]    = h_c
        P_cool_arr[i] = P_cool

        # Advance coolant temperature (energy balance for this segment)
        if i < n_stations - 1:
            dx = abs(x_sta[i + 1] - x_sta[i])
            r_local = r_sta[i]
            dQ = wt['q'] * 2.0 * np.pi * r_local * dx  # heat into coolant [W]
            dT = dQ / (mdot_coolant * cp_c) if mdot_coolant > 0 else 0
            T_cool = T_cool + dT

            # Pressure drop (Darcy-Weisbach)
            Re_f = max(Re_c, 100)
            eps_D = roughness / D_h
            f = 0.25 / (np.log10(eps_D / 3.7 + 5.74 / Re_f ** 0.9)) ** 2
            dP = f * (dx / D_h) * (rho_c * v_cool ** 2 / 2.0)
            P_cool = P_cool - dP

    # Structural margin: UTS(T_wg) / hoop stress from Pc
    sigma_applied = P0 * r_sta / max(t_wall, 1e-6)  # thin-wall hoop stress
    sigma_uts = tensile_strength(wall_material, T_wg_arr)
    margin = sigma_uts / np.clip(sigma_applied, 1.0, None)

    total_heat = np.trapz(q_arr * 2.0 * np.pi * r_sta,
                          np.abs(x_sta - x_sta[0]))

    return dict(
        x=x_sta, r=r_sta,
        T_wg=T_wg_arr, T_wc=T_wc_arr, T_cool=T_cool_arr,
        q=q_arr, h_g=h_g_arr, h_c=h_c_arr,
        P_cool=P_cool_arr, M_local=M_sta,
        total_heat=total_heat,
        dP_total=P_cool_arr[0] - P_cool_arr[-1],
        T_cool_exit=T_cool_arr[-1],
        max_T_wg=float(np.max(T_wg_arr)),
        margin=margin,
    )


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("cooling.py — run via engine_system.py for full integration")
    print("Standalone test with synthetic data:")

    # Synthetic engine result
    eng = dict(T0=3500, pc=7e6, gamma=1.2, R=400, Cstar=2300,
               cp=2000, mu_gas=6e-5, Pr_gas=0.5)

    # Synthetic nozzle contour (straight cone)
    x = np.linspace(-0.05, 0.20, 100)
    Rt = 0.04
    r = np.where(x < 0, Rt + abs(x) * np.tan(np.radians(30)),
                 Rt + x * np.tan(np.radians(15)))
    wall = np.column_stack([x, r])

    noz = dict(Me=3.0, wall=wall)
    geom = CoolingChannelGeometry(n_channels=80, width=1.5e-3, height=3e-3,
                                  wall_thickness=1.0e-3)
    cool = CoolantProperties("H2")

    result = solve_cooling_loop(eng, noz, wall, geom, cool, "CuCrZr",
                                mdot_coolant=0.15, n_stations=60)

    print(f"  Max T_wg     = {result['max_T_wg']:.0f} K")
    print(f"  T_cool exit  = {result['T_cool_exit']:.0f} K")
    print(f"  Total heat   = {result['total_heat']/1e3:.1f} kW")
    print(f"  Coolant dP   = {result['dP_total']/1e6:.2f} MPa")
    print(f"  Min margin   = {np.min(result['margin']):.2f}")
