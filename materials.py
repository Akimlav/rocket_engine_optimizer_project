"""
materials.py — Temperature-dependent material property database

Properties: thermal conductivity k(T), ultimate tensile strength sigma_uts(T),
density rho (constant), specific heat cp(T), melting/service-limit T_melt.

All properties SI.  Temperature-dependent lookups use numpy.interp
(linear interpolation, clamped at edge values).

Reference data: MIL-HDBK-5J, ASM International, MPDB.
"""

import numpy as np

# ── Material database ────────────────────────────────────────────────────────
# Each entry: rho [kg/m3], T_melt [K],
#   k_data   = (T_K, k_WmK)
#   cp_data  = (T_K, cp_JkgK)
#   sigma_uts_data = (T_K, sigma_Pa)

_a = np.array  # shorthand

MATERIALS = {
    "CuCrZr": dict(
        description="Copper-Chromium-Zirconium (C18150) — regen-cooled chambers",
        rho=8900.0,
        T_melt=1350.0,
        k_data=(
            _a([  100,  200,  300,  400,  500,  600,  700,  800,  900, 1000]),
            _a([  340,  335,  320,  305,  290,  275,  260,  250,  240,  230]),
        ),
        cp_data=(
            _a([  100,  200,  300,  400,  500,  600,  700,  800,  900, 1000]),
            _a([  330,  370,  385,  395,  405,  415,  425,  435,  450,  465]),
        ),
        sigma_uts_data=(
            _a([  300,  400,  500,  600,  700,  800,  900, 1000]),
            _a([  470,  440,  400,  340,  260,  170,   90,   40]) * 1e6,
        ),
    ),
    "OFHC-Cu": dict(
        description="Oxygen-Free High-Conductivity Copper (C10200)",
        rho=8940.0,
        T_melt=1356.0,
        k_data=(
            _a([  100,  200,  300,  400,  500,  600,  700,  800,  900, 1000]),
            _a([  420,  405,  395,  385,  375,  370,  360,  350,  340,  330]),
        ),
        cp_data=(
            _a([  100,  200,  300,  400,  500,  600,  700,  800,  900, 1000]),
            _a([  310,  360,  385,  395,  400,  410,  420,  430,  440,  455]),
        ),
        sigma_uts_data=(
            _a([  300,  400,  500,  600,  700,  800]),
            _a([  220,  180,  140,  100,   60,   25]) * 1e6,
        ),
    ),
    "SS304": dict(
        description="Stainless Steel 304 (UNS S30400)",
        rho=8000.0,
        T_melt=1700.0,
        k_data=(
            _a([  100,  200,  300,  400,  500,  600,  800, 1000, 1200]),
            _a([ 9.0, 12.5, 16.2, 19.0, 21.5, 23.5, 27.0, 30.0, 32.5]),
        ),
        cp_data=(
            _a([  100,  200,  300,  400,  500,  600,  800, 1000]),
            _a([  350,  440,  500,  520,  540,  560,  580,  610]),
        ),
        sigma_uts_data=(
            _a([  300,  400,  500,  600,  700,  800,  900, 1000]),
            _a([  515,  480,  450,  420,  380,  310,  230,  140]) * 1e6,
        ),
    ),
    "SS316": dict(
        description="Stainless Steel 316 (UNS S31600)",
        rho=8000.0,
        T_melt=1670.0,
        k_data=(
            _a([  100,  200,  300,  400,  500,  600,  800, 1000, 1200]),
            _a([ 8.5, 11.8, 14.6, 17.0, 19.5, 21.5, 25.0, 28.0, 30.5]),
        ),
        cp_data=(
            _a([  100,  200,  300,  400,  500,  600,  800, 1000]),
            _a([  360,  450,  500,  520,  540,  560,  590,  620]),
        ),
        sigma_uts_data=(
            _a([  300,  400,  500,  600,  700,  800,  900, 1000]),
            _a([  520,  490,  460,  430,  380,  300,  220,  130]) * 1e6,
        ),
    ),
    "Inconel718": dict(
        description="Nickel superalloy Inconel 718 (UNS N07718)",
        rho=8190.0,
        T_melt=1600.0,
        k_data=(
            _a([  100,  200,  300,  400,  500,  600,  800, 1000, 1200]),
            _a([ 8.0, 10.0, 11.4, 13.4, 15.8, 18.0, 22.0, 25.5, 29.0]),
        ),
        cp_data=(
            _a([  100,  200,  300,  400,  500,  600,  800, 1000]),
            _a([  340,  400,  435,  460,  485,  505,  545,  575]),
        ),
        sigma_uts_data=(
            _a([  300,  400,  500,  600,  700,  800,  900, 1000, 1100]),
            _a([ 1240, 1200, 1160, 1100, 1020,  880,  650,  350,  150]) * 1e6,
        ),
    ),
    "CarbonCarbon": dict(
        description="Carbon-Carbon composite (2D layup, through-thickness)",
        rho=1700.0,
        T_melt=3800.0,  # sublimation
        k_data=(
            _a([  300,  500,  800, 1000, 1500, 2000, 2500, 3000]),
            _a([ 15.0, 22.0, 30.0, 35.0, 42.0, 48.0, 50.0, 50.0]),
        ),
        cp_data=(
            _a([  300,  500,  800, 1000, 1500, 2000, 2500, 3000]),
            _a([  710,  960, 1240, 1400, 1650, 1800, 1900, 1950]),
        ),
        sigma_uts_data=(
            _a([  300,  500, 1000, 1500, 2000, 2500, 3000]),
            _a([  150,  160,  175,  190,  180,  150,  120]) * 1e6,
        ),
    ),
    "NbC103": dict(
        description="Niobium alloy C-103 (Nb-10Hf-1Ti) — radiation-cooled thrusters",
        rho=8860.0,
        T_melt=2620.0,
        k_data=(
            _a([  300,  500,  800, 1000, 1200, 1500, 1800, 2000]),
            _a([ 42.0, 44.0, 48.0, 51.0, 55.0, 60.0, 65.0, 68.0]),
        ),
        cp_data=(
            _a([  300,  500,  800, 1000, 1200, 1500, 1800, 2000]),
            _a([  270,  280,  290,  295,  300,  310,  320,  325]),
        ),
        sigma_uts_data=(
            _a([  300,  500,  800, 1000, 1200, 1500, 1800, 2000]),
            _a([  400,  350,  280,  230,  180,  120,   75,   45]) * 1e6,
        ),
    ),
}

# ── Public API ────────────────────────────────────────────────────────────────

def thermal_conductivity(material, T):
    """
    Thermal conductivity at temperature T.

    Parameters
    ----------
    material : str   material name key (see list_materials())
    T        : float or ndarray  temperature [K]

    Returns
    -------
    k : float or ndarray  thermal conductivity [W/(m*K)]
    """
    d = MATERIALS[material]
    return np.interp(T, d['k_data'][0], d['k_data'][1])


def specific_heat(material, T):
    """
    Specific heat capacity at temperature T.

    Parameters
    ----------
    material : str
    T        : float or ndarray  [K]

    Returns
    -------
    cp : float or ndarray  [J/(kg*K)]
    """
    d = MATERIALS[material]
    return np.interp(T, d['cp_data'][0], d['cp_data'][1])


def tensile_strength(material, T):
    """
    Ultimate tensile strength at temperature T.

    Parameters
    ----------
    material : str
    T        : float or ndarray  [K]

    Returns
    -------
    sigma_uts : float or ndarray  [Pa]
    """
    d = MATERIALS[material]
    return np.interp(T, d['sigma_uts_data'][0], d['sigma_uts_data'][1])


def density(material):
    """Density [kg/m^3] (temperature-independent)."""
    return MATERIALS[material]['rho']


def melting_point(material):
    """Melting / maximum service temperature [K]."""
    return MATERIALS[material]['T_melt']


def list_materials():
    """Return list of available material name strings."""
    return list(MATERIALS.keys())


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    T_test = np.array([300, 500, 800, 1000])
    for name in list_materials():
        m = MATERIALS[name]
        print(f"\n{name}: {m['description']}")
        print(f"  rho = {m['rho']:.0f} kg/m3   T_melt = {m['T_melt']:.0f} K")
        for T in T_test:
            if T <= m['T_melt']:
                print(f"  T={T:5.0f} K  k={thermal_conductivity(name,T):7.1f} W/mK  "
                      f"cp={specific_heat(name,T):6.0f} J/kgK  "
                      f"UTS={tensile_strength(name,T)/1e6:7.0f} MPa")
