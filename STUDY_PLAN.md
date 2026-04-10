# Rocket Engine Optimizer — Study Plan

A self-paced learning curriculum to master both the physics and the code.
Each module has: theory to understand → experiments to run → questions to answer.

Estimated total time: **6–10 weeks** at ~5–8 hours/week.

---

## Prerequisites

- Python (NumPy, matplotlib) — intermediate level
- Basic calculus and thermodynamics (ideal gas law, energy conservation)
- No prior aerospace experience needed — everything is built from scratch here

---

## Module 1 — Thermodynamics of Combustion (Week 1)

### What you need to understand first

Rocket propulsion is thermodynamics. All performance ultimately traces back to:
how hot the gas gets, how heavy the molecules are, and how well we expand them.

**Core concepts:**

**Enthalpy** is the energy content of a gas at constant pressure:
```
H = U + PV  (internal energy + pressure-volume work)
```
The HP equilibrium in `chamber.py` conserves enthalpy — the reactants release
chemical energy that heats the products, keeping H constant.

**Gibbs free energy minimization** is how Cantera finds equilibrium:
```
G = H - T·S  (minimize at fixed T, P)
```
At equilibrium, no reaction can lower G further. This determines which species
(H₂O, OH, H, O, CO₂, CO...) exist at what concentrations.

**Specific impulse (Isp)** is the fundamental rocket efficiency metric:
```
Isp = Thrust / (ṁ · g₀)   [seconds]
```
Think of it as "how many seconds one kilogram of propellant produces one newton of thrust."
Higher Isp = less propellant needed for the same mission. A 10% Isp improvement
is enormous — it can double the payload for a given rocket size.

**Why O/F ratio matters:**
- Too oxidizer-rich: heavy O₂ molecules lower Isp even if T₀ is high
- Too fuel-rich: lower T₀, but lighter molecules (H₂) can still give high Isp
- The sweet spot balances temperature against mean molecular weight

### Experiments

**Experiment 1.1 — Explore propellant chemistry**
```python
from chamber import chamber_conditions, of_sweep

# Run an O/F sweep for LOX/H2
results = of_sweep("O2", "H2", P0=7e6, OF_min=2.0, OF_max=12.0, n_points=30)
valid = [r for r in results if "T0" in r]

import matplotlib.pyplot as plt
OFs = [r["OF"] for r in valid]
T0s = [r["T0"] for r in valid]
Rs  = [r["R"]  for r in valid]
Cstars = [r["Cstar"] for r in valid]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(OFs, T0s); axes[0].set(xlabel="O/F", ylabel="T₀ [K]", title="Flame temperature")
axes[1].plot(OFs, Rs);  axes[1].set(xlabel="O/F", ylabel="R [J/kgK]", title="Gas constant (∝ 1/MW)")
axes[2].plot(OFs, Cstars); axes[2].set(xlabel="O/F", ylabel="C* [m/s]", title="Characteristic velocity")
plt.tight_layout(); plt.savefig("study_1_1.png", dpi=120)
```

**Experiment 1.2 — Compare propellant pairs**
```python
pairs = [
    ("O2", "H2",   2.0, 10.0),
    ("O2", "CH4",  1.5,  6.0),
    ("O2", "C3H8", 1.0,  5.0),
]
for ox, fuel, lo, hi in pairs:
    results = of_sweep(ox, fuel, P0=7e6, OF_min=lo, OF_max=hi, n_points=20)
    valid = [r for r in results if "T0" in r]
    best = max(valid, key=lambda r: r["Cstar"])
    print(f"{ox}/{fuel}: best O/F={best['OF']:.2f}  T0={best['T0']:.0f}K  "
          f"C*={best['Cstar']:.0f}m/s  MW={best['MW']:.1f}g/mol")
```

**Experiment 1.3 — Effect of chamber pressure on C***
```python
from chamber import chamber_conditions
pressures = [2e6, 5e6, 10e6, 20e6]
for P0 in pressures:
    c = chamber_conditions("O2", "H2", OF=6.0, P0=P0)
    print(f"  P0={P0/1e6:.0f}MPa  T0={c['T0']:.0f}K  γ={c['gamma']:.4f}  C*={c['Cstar']:.0f}m/s")
```

### Questions to answer

1. Why does LOX/H2 have higher Isp than LOX/CH4 even though LOX/CH4 has a higher flame temperature?
2. In Experiment 1.3, how much does C* change with chamber pressure? Why?
3. What is the physical meaning of C*? If two engines have the same C* but different nozzles, which produces more thrust?
4. Why does the optimal O/F for Isp differ from the stoichiometric O/F?

---

## Module 2 — Isentropic Nozzle Flow (Week 2)

### Theory

The nozzle converts thermal energy (T₀, P₀) into kinetic energy (Ve).
The governing principle is **conservation of energy**:

```
h₀ = h + V²/2      (stagnation enthalpy = static + kinetic)
cₚ·T₀ = cₚ·T + V²/2
```

For an ideal gas with constant γ, this becomes the **isentropic relations**:
```
T/T₀ = [1 + (γ-1)/2 · M²]⁻¹
P/P₀ = (T/T₀)^(γ/(γ-1))
```

**Why does flow accelerate through a nozzle?**
The continuity equation (mass conservation) requires:
```
ṁ = ρ·A·V = constant
```
In supersonic flow, density falls faster than area grows, so V must increase.
The throat (M=1) is where subsonic-to-supersonic transition happens.

**The area-Mach relation** (derived from continuity + energy + isentropic):
```
A/A* = (1/M) · [(2/(γ+1)) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1)))
```
This has two solutions for any A/A* > 1: subsonic (converging) and
supersonic (diverging). The nozzle design selects the supersonic branch.

**Maximum mass flow occurs at M=1 (choked throat):**
```
ṁ_max = P₀ · At · Γ / √(R·T₀)    where Γ = √γ · (2/(γ+1))^((γ+1)/(2(γ-1)))
```
Adding more propellant just raises P₀ — mass flow is set by geometry.

### Experiments

**Experiment 2.1 — Visualize isentropic relations**
```python
import numpy as np
import matplotlib.pyplot as plt
from nozzle_analysis import isentropic, area_ratio

M = np.linspace(0.01, 5.0, 500)
gammas = [1.2, 1.3, 1.4]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for g in gammas:
    T_r, P_r, rho_r = isentropic(M, g)
    AR = area_ratio(M, g)
    axes[0].plot(M, T_r,   label=f"γ={g}")
    axes[1].plot(M, P_r,   label=f"γ={g}")
    axes[2].plot(M, 1/AR,  label=f"γ={g}")   # 1/AR = throat area fraction

for ax, title, ylabel in zip(axes,
    ["Temperature ratio T/T₀", "Pressure ratio P/P₀", "Throat area fraction A*/A"],
    ["T/T₀", "P/P₀", "A*/A"]):
    ax.set(xlabel="Mach", ylabel=ylabel, title=title)
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("study_2_1.png", dpi=120)
```

**Experiment 2.2 — Full nozzle analysis**
```python
from nozzle_analysis import run, save_results

# Run the analytical solver
params = {
    "name": "study_2_2", "gamma": 1.2, "R": 461.5,
    "T0": 3500, "P0": 7e6, "Pe": 101325,
    "throat_radius": 0.04, "half_angle_deg": 15.0, "n_points": 300,
}
result = run(params)
save_results(result)
```
Open `results/study_2_2/nozzle_plots.png`. Trace how T, P, M change from
chamber to exit. Note that most pressure drop occurs in the supersonic section.

**Experiment 2.3 — Effect of expansion ratio**
```python
from nozzle_analysis import run

base = dict(name="er_test", gamma=1.2, R=461.5, T0=3500, P0=7e6,
            throat_radius=0.04, half_angle_deg=15.0, n_points=200)

for Pe in [1e6, 3e5, 1e5, 3e4, 1e4, 3e3]:
    r = run({**base, "Pe": Pe})
    # r[18] = Isp_vac, r[12] = Me, r[11] = Ae_At
    print(f"  Pe={Pe/1e3:7.1f}kPa  Me={r[12]:.2f}  Ae/At={r[11]:.1f}  Isp_vac={r[18]:.1f}s")
```

### Questions to answer

1. At what Mach number does the flow have exactly half the stagnation pressure?
2. If you double the throat area (keeping P₀ and T₀ constant), what happens to thrust? To Isp?
3. Why does a sea-level nozzle have a smaller expansion ratio than a vacuum nozzle?
4. What happens physically when Pe < P_ambient? (over-expanded nozzle)
5. In Experiment 2.3, does Isp_vac keep increasing as Pe → 0? Why or why not?

---

## Module 3 — Rao Nozzle Contour (Week 3)

### Theory

A straight cone wastes thrust because the gas near the wall exits at angle θ
to the axis. The **divergence loss factor**:
```
λ = 0.5 · (1 + cos θ)    →  15° cone: λ = 0.983  (1.7% loss)
```

The **Rao thrust-optimized contour** eliminates this by shaping the wall to
turn all the gas to axial flow at the exit. It uses Method of Characteristics
to trace supersonic wave fronts — the wall is the characteristic that gives
the maximum impulse for a given length.

**Hermite cubic interpolation** gives C¹-continuous (smooth tangent) curves:
```
r(t) = h₀₀·rᵢ + h₁₀·m₀ + h₀₁·Re + h₁₁·m₁
```
The four basis functions h₀₀...h₁₁ are cubic polynomials that satisfy the
boundary conditions at t=0 and t=1. This is the same technique used in
animation, font design, and CAD.

**Why is the circular arc near the throat necessary?**
At M=1, the flow transitions from subsonic to supersonic. A sharp corner would
create a shock. The circular arc (R = 0.382·Rt) provides a smooth curvature
that matches the transonic flow characteristics near M=1.

### Experiments

**Experiment 3.1 — Visualize contour shape vs parameters**
```python
import numpy as np
import matplotlib.pyplot as plt
from moc import rao_contour
from nozzle_analysis import area_ratio
from scipy.optimize import brentq

Rt = 0.04; g = 1.2
Me = brentq(lambda M: area_ratio(M, g) - 40.0, 3.0, 15.0)  # Ae/At=40

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Vary exit angle θₑ
for te in [2, 5, 10, 15]:
    w = rao_contour(g, Rt, Me, theta_i_deg=30, theta_e_deg=te, n_lines=12)
    axes[0].plot(w[:,0]*100, w[:,1]*100, label=f"θₑ={te}°")
axes[0].set(xlabel="x [cm]", ylabel="r [cm]", title="Effect of exit angle θₑ")
axes[0].legend(); axes[0].set_aspect("equal")

# Vary inflection angle θᵢ
for ti in [15, 25, 35, 45]:
    w = rao_contour(g, Rt, Me, theta_i_deg=ti, theta_e_deg=5, n_lines=12)
    axes[1].plot(w[:,0]*100, w[:,1]*100, label=f"θᵢ={ti}°")
axes[1].set(xlabel="x [cm]", ylabel="r [cm]", title="Effect of inflection angle θᵢ")
axes[1].legend(); axes[1].set_aspect("equal")

plt.tight_layout(); plt.savefig("study_3_1.png", dpi=120)
```

**Experiment 3.2 — Divergence loss vs exit angle**
```python
import numpy as np
from moc import rao_contour
from engine_system import nozzle_perf
from nozzle_analysis import mach_from_pressure_ratio

# Use a fixed engine operating point
eng = {"gamma":1.2, "R":461.5, "T0":3500, "pc":7e6, "Cstar":2300}
Rt = 0.04; Pe = 101325

exit_angles = np.linspace(1.0, 20.0, 20)
isps = []; lambdas = []
for te in exit_angles:
    noz = nozzle_perf(eng, Rt, Pe, theta_i=30, theta_e=te, n_lines=10)
    isps.append(noz["Isp_vac"])
    lambdas.append(noz["divergence_lambda"])

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(exit_angles, isps); ax1.set(xlabel="θₑ [deg]", ylabel="Isp_vac [s]")
ax1.set_title("Isp vs exit angle")
ax2.plot(exit_angles, lambdas); ax2.set(xlabel="θₑ [deg]", ylabel="λ")
ax2.axhline(1.0, color="r", ls="--"); ax2.set_title("Divergence loss factor")
plt.tight_layout(); plt.savefig("study_3_2.png", dpi=120)
```

**Experiment 3.3 — Manually try to beat the optimizer**
```python
from engine_system import nozzle_perf

eng = {"gamma":1.2, "R":461.5, "T0":3500, "pc":7e6, "Cstar":2300}

# Your best guess:
my_theta_i = 30.0   # ← tune this
my_theta_e = 5.0    # ← tune this

noz = nozzle_perf(eng, Rt=0.04, Pe=101325, theta_i=my_theta_i, theta_e=my_theta_e)
print(f"Your result: Isp_vac = {noz['Isp_vac']:.2f} s  L = {noz['length']:.3f} m")
# Then run the optimizer and compare
```

### Questions to answer

1. From Experiment 3.2: what θₑ gives maximum Isp? Does the answer match the optimizer's output?
2. Why does making θₑ very small (→0°) not give infinite Isp?
3. What does the nozzle length vs Isp trade-off look like? Plot it.
4. If you increase γ from 1.2 to 1.4, how does the optimal contour change?
5. Read `moc.py` lines 46–73. Why is `clip(r_wall, Rt, None)` needed?

---

## Module 4 — Numerical Optimization (Week 4)

### Theory

The nozzle optimization problem:
- **3 free variables**: θᵢ, θₑ, n_lines
- **1 objective**: maximize Isp_vac (equivalently, minimize −Isp_vac)
- **1 constraint**: nozzle length ≤ L_max (enforced as penalty)

**Why not use gradient descent?**
The objective has:
- Discontinuities (integer n_lines)
- Local optima (multiple good θᵢ/θₑ combinations)
- No closed-form gradient (each evaluation involves a root-finding call)

**Differential Evolution** handles all this. The algorithm:
1. Start with a random population of N candidate solutions (vectors)
2. For each candidate **x**, create a mutant **v** = **a** + F·(**b** − **c**)
   where **a**, **b**, **c** are random population members and F ∈ [0,2]
3. Create trial **u** by mixing **x** and **v** (crossover)
4. Keep **u** if f(**u**) < f(**x**), else keep **x**
5. Repeat until converged

**Nelder-Mead simplex** is fast for local refinement:
- Maintains n+1 points (a "simplex") in parameter space
- Reflects, expands, or contracts the simplex toward the minimum
- No gradient required — pure function evaluations

### Experiments

**Experiment 4.1 — Visualize the objective landscape**
```python
import numpy as np
import matplotlib.pyplot as plt
from moc import rao_contour
from engine_system import nozzle_perf
from nozzle_analysis import mach_from_pressure_ratio

eng = {"gamma":1.2, "R":461.5, "T0":3500, "pc":7e6, "Cstar":2300}
Rt = 0.04; Pe = 101325

theta_i_range = np.linspace(15, 45, 30)
theta_e_range = np.linspace(2, 15, 30)
Isp_grid = np.zeros((30, 30))

for i, ti in enumerate(theta_i_range):
    for j, te in enumerate(theta_e_range):
        try:
            noz = nozzle_perf(eng, Rt, Pe, theta_i=ti, theta_e=te, n_lines=10)
            Isp_grid[i, j] = noz["Isp_vac"] if noz["length"] < 0.6 else 0
        except Exception:
            Isp_grid[i, j] = 0

plt.figure(figsize=(8, 6))
cp = plt.contourf(theta_e_range, theta_i_range, Isp_grid, levels=20, cmap="viridis")
plt.colorbar(cp, label="Isp_vac [s]")
plt.xlabel("θₑ [deg]"); plt.ylabel("θᵢ [deg]")
plt.title("Objective landscape: Isp_vac(θᵢ, θₑ)")
plt.savefig("study_4_1.png", dpi=120)
```

**Experiment 4.2 — Watch DE converge**
```python
# Modify nozzle_optimizer.py to track history, then:
import json
import pandas as pd
import matplotlib.pyplot as plt

# Run the optimizer first, then:
df = pd.read_csv("results/your_run/optimization_history.csv")
valid = df[df["Isp_vac"] > 0]
best_so_far = valid["Isp_vac"].cummax()

plt.figure(figsize=(10, 4))
plt.scatter(valid.index, valid["Isp_vac"], alpha=0.3, s=5, label="All evaluations")
plt.plot(best_so_far.index, best_so_far, "r-", lw=2, label="Best so far")
plt.xlabel("Evaluation #"); plt.ylabel("Isp_vac [s]")
plt.title("DE convergence"); plt.legend()
plt.savefig("study_4_2.png", dpi=120)
```

**Experiment 4.3 — Sensitivity to n_eval**
Run the optimizer with n_eval = 20, 50, 100, 200 and record the best Isp.
At what point does increasing n_eval stop helping?

**Experiment 4.4 — Add a second constraint**
Modify the objective in `nozzle_optimizer.py` to also penalize
exit radius > 0.5 m. How does this change the optimal solution?

### Questions to answer

1. From Experiment 4.1: is the objective landscape smooth? Are there ridges?
2. Why does DE use mutation F·(b−c)? What happens if F=0 or F=2?
3. In the code, why is the Nelder-Mead run on `de.x` (DE result) and not random?
4. The optimizer sometimes finds θᵢ > 45°. Why does the code clip it to 50°?
5. How would you reformulate this as a gradient-based problem?

---

## Module 5 — Heat Transfer & Cooling (Week 5–6)

### Theory

**Why the throat is the hardest point to cool:**
The Bartz correlation shows h_g ∝ (At/A(x))^0.9. At the throat, A(x) = At,
so the heat transfer coefficient is maximum. The throat also has the highest
gas velocity (M=1) and temperature (T* = T₀ × 2/(γ+1)). Typical throat heat
flux: 10–50 MW/m² — more than the surface of the sun (63 MW/m²).

**The three-layer thermal resistance model:**
```
Gas → [gas film] → wall outer surface → [wall] → wall inner surface → [coolant film] → Coolant

q = (T_aw - T_cool) / (1/h_g + t_wall/k_wall + 1/h_c)

T_wg = T_aw - q/h_g       (gas-side wall temp — the critical value)
T_wc = T_cool + q/h_c     (coolant-side wall temp)
```

**Adiabatic wall temperature** is what the wall would reach if it were
perfectly insulated — the gas "stagnates" near the wall:
```
T_aw = T₀ · (1 + r·(γ-1)/2·M²) / (1 + (γ-1)/2·M²)    r = Pr^(1/3)
```
Near M=1 (throat): T_aw ≈ 0.9·T₀. For LOX/H2 that's ~3100 K — way above
copper's melting point of ~1356 K. Hence cooling is essential.

**The Bartz number vs Dittus-Boelter balance:**
- Too few channels → v_cool ↑ → Re ↑ → h_c ↑ ✓ but ΔP ↑↑ (bad)
- Too many channels → v_cool ↓ → Re ↓ → h_c ↓ (bad)
- Wall too thick → conduction resistance k_wall/t_wall ↓ → T_wg ↑ (bad)
- Wall too thin → structural failure under Pc (bad)

This is the central design trade-off of regen cooling.

### Experiments

**Experiment 5.1 — Run the full cooling loop**
```python
from injector import Injector
from engine_solver import EngineSolver
from engine_system import nozzle_perf
from moc import rao_contour
from nozzle_analysis import mach_from_pressure_ratio
from cooling import CoolingChannelGeometry, CoolantProperties, solve_cooling_loop
import numpy as np

# Get engine operating point
inj = Injector(fuel="H2", oxidizer="O2", A_f=2e-4, A_o=2e-4)
solver = EngineSolver(np.pi*0.04**2, inj, fuel="H2", oxidizer="O2")
eng = solver.solve(8e6, 8e6)

# Build nozzle contour
from scipy.optimize import brentq
from nozzle_analysis import area_ratio
Me = brentq(lambda M: area_ratio(M, eng["gamma"]) - 30.0, 3.0, 15.0)
Pe = eng["pc"] * (1 + (eng["gamma"]-1)/2 * Me**2) ** (-eng["gamma"]/(eng["gamma"]-1))
noz = nozzle_perf(eng, 0.04, Pe)
wall = noz["wall"]

# Cooling setup
geom = CoolingChannelGeometry(n_channels=80, width=1.5e-3, height=3e-3,
                               wall_thickness=1.0e-3)
coolant = CoolantProperties("H2")
result = solve_cooling_loop(eng, noz, wall, geom, coolant, "CuCrZr",
                            mdot_coolant=eng["mdot_f"], n_stations=80)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
x = result["x"] * 100  # cm

axes[0,0].plot(x, result["T_wg"], "r-", label="Gas-side wall")
axes[0,0].plot(x, result["T_wc"], "b-", label="Coolant-side wall")
axes[0,0].plot(x, result["T_cool"], "g-", label="Coolant bulk")
axes[0,0].axhline(1350, color="k", ls="--", label="CuCrZr melt limit")
axes[0,0].set(xlabel="x [cm]", ylabel="T [K]", title="Temperature profiles")
axes[0,0].legend(fontsize=8)

axes[0,1].plot(x, result["q"]/1e6, "m-")
axes[0,1].set(xlabel="x [cm]", ylabel="q [MW/m²]", title="Heat flux (peak = throat)")

axes[1,0].plot(x, result["h_g"]/1e3, "r-", label="Gas-side h_g")
axes[1,0].plot(x, result["h_c"]/1e3, "b-", label="Coolant-side h_c")
axes[1,0].set(xlabel="x [cm]", ylabel="h [kW/m²K]", title="Heat transfer coefficients")
axes[1,0].legend()

axes[1,1].plot(x, result["margin"], "k-")
axes[1,1].axhline(1.0, color="r", ls="--", label="Failure threshold")
axes[1,1].axhline(1.5, color="orange", ls="--", label="Design target (1.5×)")
axes[1,1].set(xlabel="x [cm]", ylabel="UTS/σ_applied", title="Structural margin")
axes[1,1].legend()

plt.tight_layout(); plt.savefig("study_5_1.png", dpi=120)
print(f"Max wall temp: {result['max_T_wg']:.0f} K")
print(f"Total heat: {result['total_heat']/1e6:.1f} MW")
print(f"Min margin: {min(result['margin']):.2f}x")
```

**Experiment 5.2 — Channel geometry trade-off**
```python
from cooling import CoolingChannelGeometry, CoolantProperties, solve_cooling_loop

# Sweep n_channels
results = {}
for n_ch in [40, 60, 80, 100, 120, 160]:
    geom = CoolingChannelGeometry(n_channels=n_ch, width=1.5e-3, height=3e-3,
                                   wall_thickness=1e-3)
    r = solve_cooling_loop(eng, noz, wall, geom, CoolantProperties("H2"),
                           "CuCrZr", mdot_coolant=eng["mdot_f"], n_stations=60)
    results[n_ch] = r

import matplotlib.pyplot as plt
ns = list(results.keys())
max_Twg = [results[n]["max_T_wg"] for n in ns]
dP      = [results[n]["dP_total"]/1e6 for n in ns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(ns, max_Twg, "o-"); ax1.axhline(1350, color="r", ls="--", label="Melt limit")
ax1.set(xlabel="n_channels", ylabel="Max T_wg [K]", title="Wall temp vs n_channels")
ax2.plot(ns, dP, "s-", color="steelblue")
ax2.set(xlabel="n_channels", ylabel="ΔP_coolant [MPa]", title="Pressure drop vs n_channels")
plt.tight_layout(); plt.savefig("study_5_2.png", dpi=120)
```

**Experiment 5.3 — Material comparison**
Swap `wall_material` between `"CuCrZr"`, `"SS304"`, and `"Inconel718"`.
Why does copper work much better despite lower strength?

### Questions to answer

1. Where along the nozzle axis is the peak heat flux? Why there?
2. Why does the coolant temperature rise less near the nozzle exit than near the throat?
3. From Experiment 5.2: is there an optimal number of channels? What limits it?
4. If you double the wall thickness, by how much does T_wg increase?
5. Why does Inconel718 (high-strength) perform worse as a chamber wall than CuCrZr (high-conductivity)?

---

## Module 6 — Feed Systems & Engine Cycles (Week 7)

### Theory

**Pressure balance in a feed system:**
```
P_tank + ΔP_pump ≥ P_chamber + ΔP_line + ΔP_injector
          ↑                         ↑           ↑
   what drives flow           friction losses  stability margin
```

**Why GG cycle loses Isp:**
The gas generator burns ~2-5% of propellant to drive the turbine. That flow
never passes through the main nozzle (or passes through a small secondary
nozzle at low Isp). The effective Isp penalty:
```
Isp_eff = Isp_main · (1 - f_bleed) + Isp_dump · f_bleed
```
For f_bleed = 3%, Isp_dump ≈ 150 s, Isp_main = 430 s:
```
Isp_eff ≈ 430 · 0.97 + 150 · 0.03 ≈ 421 s  (2% penalty)
```

**Why turbopump engines use low tank pressure:**
Each MPa of tank pressure requires ~6 kg/m³ of extra tank wall mass (hoop stress).
For a 100-liter tank at 10 MPa vs 0.5 MPa: wall mass difference ≈ 50 kg.
The pump that provides those extra 9.5 MPa might only weigh 5 kg — a 10× mass saving.

### Experiments

**Experiment 6.1 — Blowdown curve**
```python
from tank import Tank
import numpy as np, matplotlib.pyplot as plt

# Compare helium vs nitrogen pressurant
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for pressurant, color in [("He", "steelblue"), ("N2", "orange")]:
    tank = Tank(V_tank=0.05, P_0=8e6, propellant="LOX",
                pressurant=pressurant, ullage_fraction=0.05)
    bd = tank.blowdown_curve(mdot=0.5, t_burn=tank.m_prop_0/0.5)
    axes[0].plot(bd["t"], bd["P"]/1e6, label=pressurant, color=color)
    axes[1].plot(bd["t"], bd["m_prop"], label=pressurant, color=color)

axes[0].set(xlabel="Time [s]", ylabel="Pressure [MPa]", title="Tank pressure (blowdown)")
axes[1].set(xlabel="Time [s]", ylabel="m_prop [kg]", title="Propellant remaining")
for ax in axes: ax.legend()
plt.tight_layout(); plt.savefig("study_6_1.png", dpi=120)
```

**Experiment 6.2 — Pipe losses**
```python
from feed_system import line_pressure_drop

# What fraction of tank pressure is lost in the feed line?
mdot = 0.5    # kg/s
rho  = 1141.0 # LOX
mu   = 1.9e-4 # Pa·s

for D in [0.010, 0.015, 0.020, 0.025, 0.030]:
    r = line_pressure_drop(mdot, rho, mu, D=D, L=2.0)
    frac = r["dP_total"] / 8e6 * 100
    print(f"  D={D*1000:.0f}mm  Re={r['Re']:.0f}  v={r['velocity']:.1f}m/s  "
          f"ΔP={r['dP_total']/1e3:.1f}kPa  ({frac:.2f}% of 8MPa tank)")
```

**Experiment 6.3 — GG cycle balance**
```python
from turbopump import Pump, Turbine
from feed_system import FeedLine, GasGeneratorCycle

feed_ox   = FeedLine(D=0.025, L=2.0, propellant="LOX")
feed_fuel = FeedLine(D=0.020, L=2.5, propellant="H2")
pump_o = Pump(eta_design=0.70); pump_f = Pump(eta_design=0.65)
turb   = Turbine(eta_turbine=0.60)

gg = GasGeneratorCycle(pump_o, pump_f, turb, feed_ox, feed_fuel,
                        T_gg=900, gamma_gg=1.3, cp_gg=2500)

# Sweep turbine inlet temperature (hotter GG = less bleed needed)
import matplotlib.pyplot as plt
T_ggs = [700, 800, 900, 1000, 1100]
bleeds = []
for T in T_ggs:
    gg.T_gg = T
    r = gg.solve(0.3e6, 0.3e6, 7e6, mdot_total=1.0, OF=6.0)
    bleeds.append(r["bleed_fraction"] * 100)

plt.plot(T_ggs, bleeds, "o-")
plt.xlabel("GG temperature [K]"); plt.ylabel("Bleed fraction [%]")
plt.title("Propellant bleed vs GG temperature")
plt.savefig("study_6_3.png", dpi=120)
```

### Questions to answer

1. In Experiment 6.1: why does N₂ cause a steeper pressure drop than He?
2. From Experiment 6.2: what pipe diameter do you need to keep ΔP < 1% of tank P?
3. Why is helium usually chosen as pressurant despite being expensive?
4. In Experiment 6.3: what is the limit of raising T_gg? What prevents using T_gg = 3000 K?
5. Why does the expander cycle only work with hydrogen (or methane) and not RP-1?

---

## Module 7 — Full System Integration (Week 8)

### Capstone project: Design your own engine

Design a small vacuum thruster for a satellite. Constraints:
- Propellants: LOX/CH4 (modern, storable enough)
- Target thrust: 500 N vacuum
- Burn time: 60 s
- Max nozzle length: 15 cm

**Task 1** — Find the required throat area
```
At = F_vac / (Pc · Cf_vac)    where Cf_vac ≈ 1.8 for a typical vacuum nozzle
```
Estimate At, Rt. Then verify with the code.

**Task 2** — Injector sizing
Choose Pc = 2 MPa, OF = 3.4 (optimal for LOX/CH4).
```
mdot = F_vac / (Isp_vac · g₀)
mdot_ox = mdot · OF/(1+OF)
mdot_f  = mdot / (1+OF)
```
Choose P_tank = 3 MPa. Compute A_ox and A_fuel.

**Task 3** — Run the simulation
```bash
python engine_system.py --mode simulate \
    --fuel CH4 --oxidizer O2 \
    --throat_radius YOUR_RT \
    --p_tank_f 3e6 --p_tank_o 3e6 \
    --A_f YOUR_A_FUEL --A_o YOUR_A_OX \
    --Pe 100  \
    --name my_500N_thruster
```

**Task 4** — Optimize the nozzle
```bash
python engine_system.py --mode optimize \
    --fuel CH4 --oxidizer O2 \
    --throat_radius YOUR_RT \
    --p_tank_f 3e6 --p_tank_o 3e6 \
    --A_f YOUR_A_FUEL --A_o YOUR_A_OX \
    --Pe 100 --max_length_m 0.15 \
    --n_eval 100 --name my_500N_opt
```

**Task 5** — Propellant budget
```python
from tank import Tank
Isp = ...  # from your simulation
F   = 500  # N
mdot = F / (Isp * 9.80665)
print(f"mdot = {mdot:.4f} kg/s")

tank_ox   = Tank(V_tank=0.010, P_0=3e6, propellant="LOX",   pressurant="He")
tank_fuel = Tank(V_tank=0.006, P_0=3e6, propellant="CH4",   pressurant="He")
print(f"LOX tank:  {tank_ox.m_prop_0:.2f} kg  (need {mdot*OF/(1+OF)*60:.2f} kg)")
print(f"Fuel tank: {tank_fuel.m_prop_0:.2f} kg (need {mdot/(1+OF)*60:.2f} kg)")
```

**Task 6** — Write a short report (1–2 pages):
- What Isp did you achieve? How does it compare to state-of-the-art?
- What limited your performance (geometry constraint? propellant? pressure?)?
- What would you change to improve Isp by 5%?

---

## Module 8 — Going Deeper (Optional, Week 9–10)

Once you complete the above, these are natural next steps:

### 8.1 Add a Bartz Nusselt correction for variable properties
The Bartz correlation assumes uniform gas properties. In reality, viscosity
and conductivity vary significantly between the hot core flow and the cooler
boundary layer near the wall. Research the **Eckert reference temperature
method** and implement a correction factor in `cooling.py`.

### 8.2 Implement a real-gas equation of state
At 10+ MPa, the ideal gas law has errors of 1–3%. Look up the
**Redlich-Kwong** or **Peng-Robinson** equations of state and add them as
an option to `nozzle_analysis.py`. Compare Isp predictions.

### 8.3 Combustion efficiency (η_c*)
Real engines don't achieve 100% theoretical C*. Typical values:
- Small engines: η_c* = 0.92–0.95
- Large engines: η_c* = 0.97–0.99

Add an `eta_cstar` parameter to `engine_system.py` that scales the effective
C*. How much does 95% combustion efficiency affect Isp?

### 8.4 Specific speed and turbopump scaling
Research the **specific speed** Ns and **specific diameter** Ds for
turbopumps. Plot a Cordier diagram. For the Aquila VAC engine, what
impeller diameter and rotation speed would your pumps require?

### 8.5 Parametric mass estimation
Rocket engine mass can be estimated from empirical correlations.
Research the **Huzel & Huang** mass estimation equations and add a
`mass_budget()` function to `engine_system.py` that estimates:
- Chamber mass (cylindrical pressure vessel)
- Nozzle mass (thin shell)
- Injector mass
- Total engine dry mass

Then compute thrust-to-weight ratio for your 500 N thruster.

---

## Recommended Reading

| Resource | Covers | Where to find |
|----------|--------|---------------|
| Sutton & Biblarz — *Rocket Propulsion Elements* (9th ed.) | All of the above, authoritative | Library / O'Reilly |
| Huzel & Huang — *Modern Engineering for Design of Liquid-Propellant Rocket Engines* | Design methodology, sizing | NASA NTRS (free) |
| Anderson — *Modern Compressible Flow* (3rd ed.) | Isentropic flow, Method of Characteristics | Library |
| Bartz (1957) — *A Simple Equation for Rapid Estimation of Rocket Nozzle Convective Heat Transfer Coefficients* | Bartz correlation derivation | NASA NTRS (free) |
| Rao (1958) — *Exhaust Nozzle Contour for Optimum Thrust* | Rao contour theory | NASA NTRS (free) |
| Cantera documentation — cantera.org | Chemical kinetics, HP equilibrium | Online (free) |
| Storn & Price (1997) — *Differential Evolution* | The DE algorithm | Journal of Global Optimization |

---

## Progress Checklist

- [ ] Module 1: O/F sweep, understand C* and T₀ trade-off
- [ ] Module 2: Can reproduce isentropic flow by hand for one Mach number
- [ ] Module 3: Generated 4 contours with different θᵢ/θₑ, understand λ
- [ ] Module 4: Plotted objective landscape, ran optimizer with 3 different n_eval
- [ ] Module 5: Full cooling loop, found minimum channel count before wall melts
- [ ] Module 6: Blowdown curve for your tank sizing, GG cycle balance
- [ ] Module 7: Designed and simulated your own 500 N thruster
- [ ] Module 8: ≥1 extension implemented

**You understand the project when you can answer:**
> *"Why does the Raptor engine achieve higher Isp than the Merlin despite
>  both using similar propellants (LOX/CH4 vs LOX/RP-1)?"*

Hint: chamber pressure (300 bar vs 100 bar), expansion ratio, and the
staged-combustion vs gas-generator cycle all play a role. You now have
the tools to quantify each factor.
