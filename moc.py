"""
moc.py — Rao Optimum Nozzle Contour (Parabolic/Hermite Approximation)

Implements the standard engineering Rao TIC contour:
  1. Circular arc throat section (0 -> theta_i)
  2. Hermite cubic wall from inflection point to exit (theta_i -> theta_e)

Reference: Rao (1958), Huzel & Huang "Modern Engineering for Design of
           Liquid-Propellant Rocket Engines" Ch.4
"""
import numpy as np

# Rao standard constants
RAO_ARC_RADIUS_RATIO = 0.382   # throat arc radius / throat radius
L_FACTOR_BASE  = 0.7           # base length factor for divergent section
L_FACTOR_SCALE = 0.2           # length factor scaling with theta_i
L_FACTOR_REF   = 30.0          # reference angle [deg] for length factor


def nu_pm(M, gamma):
    g = gamma; gp = (g+1)/(g-1)
    return np.sqrt(gp)*np.arctan(np.sqrt((M**2-1)/gp)) - np.arctan(np.sqrt(M**2-1))


def rao_contour(gamma, Rt, Me, theta_i_deg, theta_e_deg, n_lines, n_pts=80):
    """
    Rao optimum nozzle wall contour.

    Parameters
    ----------
    gamma       : specific heat ratio
    Rt          : throat radius [m]
    Me          : exit Mach number
    theta_i_deg : wall angle at inflection point [deg]  (15-45 typical)
    theta_e_deg : wall angle at nozzle exit [deg]       (2-15 typical)
    n_lines     : controls arc resolution
    n_pts       : total contour points

    Returns
    -------
    wall : np.ndarray (N, 2)  columns [x_m, r_m]
    """
    from nozzle_analysis import area_ratio

    n_lines = max(int(round(n_lines)), 3)
    theta_i = np.radians(np.clip(theta_i_deg, 5.0, 50.0))
    theta_e = np.radians(np.clip(theta_e_deg, 0.5, 20.0))

    Ae_At = area_ratio(Me, gamma)
    Re = Rt * np.sqrt(Ae_At)

    # Section 1: circular arc throat
    R_arc = RAO_ARC_RADIUS_RATIO * Rt
    n_arc = max(n_lines, 5)
    arc_angles = np.linspace(0.0, theta_i, n_arc)
    x_arc = R_arc * np.sin(arc_angles)
    r_arc = Rt + R_arc * (1.0 - np.cos(arc_angles))

    # Inflection point
    x_i = x_arc[-1]; r_i = r_arc[-1]

    # Section 2: Hermite cubic inflection -> exit
    L_cone = (Re - Rt) / np.tan(np.radians(15.0))
    L_factor = L_FACTOR_BASE + L_FACTOR_SCALE * (theta_i / np.radians(L_FACTOR_REF))
    x_e = x_i + L_factor * L_cone

    n_div = n_pts - n_arc
    t = np.linspace(0.0, 1.0, n_div)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2

    dx = x_e - x_i
    m0 = np.tan(theta_i) * dx
    m1 = np.tan(theta_e) * dx

    x_div = x_i + t * dx
    r_div = h00*r_i + h10*m0 + h01*Re + h11*m1

    x_wall = np.concatenate([x_arc[:-1], x_div])
    r_wall = np.concatenate([r_arc[:-1], r_div])

    mask = np.diff(x_wall, prepend=-np.inf) > 0
    x_wall = x_wall[mask]
    r_wall = np.clip(r_wall[mask], Rt, None)

    return np.column_stack([x_wall, r_wall])


if __name__ == "__main__":
    w = rao_contour(1.4, 0.05, 3.2, 30, 8, 10)
    print(f"Contour: {len(w)} pts")
    print(f"  x: {w[0,0]:.4f} -> {w[-1,0]:.4f} m")
    print(f"  r: {w[0,1]*100:.3f} -> {w[-1,1]*100:.3f} cm")
