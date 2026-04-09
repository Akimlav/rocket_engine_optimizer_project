# Rocket Engine Optimizer — Technical Manual

A complete reference for the physics, architecture, and usage of every module in this project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture and Data Flow](#2-architecture-and-data-flow)
3. [Theory: Combustion Chemistry — `chamber.py`](#3-theory-combustion-chemistry--chamberpy)
4. [Theory: Injector Orifice Flow — `injector.py`](#4-theory-injector-orifice-flow--injectorpy)
5. [Theory: Self-Consistent Engine Solver — `engine_solver.py`](#5-theory-self-consistent-engine-solver--engine_solverpy)
6. [Theory: 1D Isentropic Nozzle — `nozzle_analysis.py`](#6-theory-1d-isentropic-nozzle--nozzle_analysispy)
7. [Theory: Rao Optimum Contour — `moc.py`](#7-theory-rao-optimum-contour--mocpy)
8. [Theory: Shape Optimization — `nozzle_optimizer.py`](#8-theory-shape-optimization--nozzle_optimizerpy)
9. [Theory: Material Properties — `materials.py`](#9-theory-material-properties--materialspy)
10. [Theory: Regenerative Cooling — `cooling.py`](#10-theory-regenerative-cooling--coolingpy)
11. [Theory: Propellant Tank (Blowdown) — `tank.py`](#11-theory-propellant-tank-blowdown--tankpy)
12. [Theory: Turbopump — `turbopump.py`](#12-theory-turbopump--turbopumpy)
13. [Theory: Feed System & Engine Cycles — `feed_system.py`](#13-theory-feed-system--engine-cycles--feed_systempy)
14. [System Integration — `engine_system.py`](#14-system-integration--engine_systempy)
15. [Quick-Start Examples](#15-quick-start-examples)
16. [Parameter Reference](#16-parameter-reference)
17. [Output Files Reference](#17-output-files-reference)
18. [Common Propellant Reference Data](#18-common-propellant-reference-data)

---

## 1. Project Overview

This project is a **1D rocket engine simulation and optimization toolkit**. Starting from propellant chemistry, it calculates engine operating conditions, designs the optimal nozzle shape, computes thermal loads on the chamber wall, and balances the propellant feed system—covering the full design chain from tanks to exhaust.

**What it can do:**
- Compute chemical equilibrium chamber conditions for any propellant pair (via Cantera)
- Solve the coupled chamber pressure / mixture ratio operating point
- Analyze 1D isentropic nozzle performance
- Generate and optimize Rao parabolic nozzle contours for maximum Isp
- Compute wall temperature profiles under regenerative cooling
- Model propellant tank blowdown curves
- Simulate turbopump power balance and feed system pressure drops
- Balance gas-generator and expander engine cycles

**What it does NOT do (yet):**
- 2D or 3D flow (no CFD)
- Boundary layer or viscous losses
- Combustion instability
- Transient startup / shutdown sequences

---

## 2. Architecture and Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                     engine_system.py  (orchestrator)               │
└───────────────────────────────┬────────────────────────────────────┘
                                │
            ┌───────────────────▼───────────────────┐
            │         OPERATING POINT               │
            │                                       │
            │  injector.py  ──► engine_solver.py    │
            │  (orifice mdot)    (Newton + Cantera)  │
            │       │                  │             │
            │       └─────────────┐   │             │
            │                     ▼   ▼             │
            │                  chamber.py           │
            │               (HP equilibrium)        │
            └───────────────────────────────────────┘
                                │
                   T0, Pc, γ, R, C*, mdot
                                │
            ┌───────────────────▼───────────────────┐
            │           NOZZLE                      │
            │                                       │
            │  moc.py  ──────► nozzle_analysis.py   │
            │  (Rao contour)    (1D isentropic perf) │
            │                                       │
            │  nozzle_optimizer.py                   │
            │  (DE + Nelder-Mead shape search)       │
            └───────────────────────────────────────┘
                                │
                    Isp, Thrust, wall[x,r]
                                │
            ┌───────────────────▼───────────────────┐
            │   PHASE 1: THERMAL                    │
            │                                       │
            │  materials.py ──► cooling.py          │
            │  (k(T), UTS(T))   (Bartz + D-B loop)  │
            └───────────────────────────────────────┘
                                │
                    T_wall, q, T_coolant
                                │
            ┌───────────────────▼───────────────────┐
            │   PHASE 2: FEED SYSTEM                │
            │                                       │
            │  tank.py ──► turbopump.py             │
            │  (blowdown)    (pump + turbine)        │
            │       └──────► feed_system.py         │
            │                (GG / expander cycle)  │
            └───────────────────────────────────────┘
```

**Data units throughout:** SI (Pa, K, m, kg, s, W, J)

---

## 3. Theory: Combustion Chemistry — `chamber.py`

### 3.1 What it solves

Given a propellant pair (e.g. LOX/H2), mixture ratio O/F, and chamber pressure P₀, it finds the chemical equilibrium composition and temperature inside the combustion chamber.

### 3.2 HP Equilibrium

The combustion process is modeled as a **constant-enthalpy, constant-pressure (HP) process**. The reactants enter at 300 K, react, and reach a final state where the Gibbs free energy of the product mixture is minimized at fixed H and P.

Cantera solves the full nonlinear minimization across all possible product species (H₂O, CO₂, CO, OH, O, H, N₂, etc.). This gives realistic dissociation effects at high temperature—something that simple stoichiometric calculations miss.

### 3.3 Key outputs

| Symbol | Name | Equation |
|--------|------|----------|
| T₀ | Adiabatic flame temperature [K] | Cantera equilibrium |
| γ | Specific heat ratio | γ = cₚ/cᵥ |
| R | Specific gas constant [J/kg/K] | R = R_universal / MW |
| C* | Characteristic velocity [m/s] | C* = √(R·T₀) / Γ |
| MW | Mean molecular weight [g/mol] | Cantera |

where Γ (the throat flow function) is:

```
Γ = √γ · (2/(γ+1))^((γ+1)/(2(γ-1)))
```

### 3.4 Characteristic velocity C*

C* is the most important single figure of merit for a combustion chamber. It depends only on the propellant thermochemistry—not on nozzle geometry. A high C* means the chamber generates a lot of pressure per unit mass flow, indicating efficient combustion.

```
C* = P₀ · At / ṁ
```

Typical values: LOX/H2 ≈ 2350–2430 m/s, LOX/CH4 ≈ 1850 m/s, LOX/RP-1 ≈ 1770 m/s.

### 3.5 O/F optimization

The code can sweep over O/F ratios and find the mixture that maximizes T₀ (`best_OF()`). The optimal O/F is usually slightly fuel-rich compared to stoichiometry—because excess fuel keeps molecular weight low and γ closer to 1.2, both of which increase C* and Isp.

---

## 4. Theory: Injector Orifice Flow — `injector.py`

### 4.1 Bernoulli orifice model

Each propellant enters the chamber through a set of orifices. The mass flow through a single orifice is modeled using the incompressible Bernoulli equation with a discharge coefficient:

```
ṁ = Cd · A · √(2 · ρ · ΔP)
```

where:
- `Cd` = discharge coefficient (0.6–0.9, accounts for vena contracta and viscosity)
- `A` = total orifice area [m²]
- `ρ` = propellant liquid density [kg/m³]
- `ΔP = P_tank − P_chamber` = pressure drop across the injector [Pa]

### 4.2 O/F ratio control

The mixture ratio is set by the relative orifice areas and propellant densities:

```
O/F = ṁ_ox / ṁ_fuel = (Cd_o · A_o · √(ρ_o · ΔP_o)) / (Cd_f · A_f · √(ρ_f · ΔP_f))
```

In a pressure-fed engine, O/F drifts as tank pressures decay (blowdown). This is one reason turbopump engines have better mixture ratio control.

### 4.3 Injector pressure drop

Real injectors need a significant pressure drop (typically 10–20% of chamber pressure) to stabilize combustion and prevent feed-coupled instabilities. Too small a drop → backflow or chugging.

---

## 5. Theory: Self-Consistent Engine Solver — `engine_solver.py`

### 5.1 The coupling problem

The injector and combustion chamber are coupled: the chamber pressure determines the injector ΔP, which determines the mass flow, which in turn determines the chamber pressure via C*.

The steady-state condition requires simultaneous satisfaction of:

```
ṁ_injector(P_tank, Pc) = Pc · At / C*(Pc, OF)
ṁ_ox / ṁ_fuel = OF
```

### 5.2 Solution method

The solver uses a **nested Newton iteration**:

**Outer loop** — converge O/F:
1. Start with a guessed O/F
2. Call Cantera to get C*(OF, Pc_guess)
3. Run inner loop to find Pc
4. Compute new O/F from injector flows at that Pc
5. Update O/F (damped: 50% old + 50% new) and repeat

**Inner loop** — converge Pc for fixed C*:
Newton's method on the residual:
```
f(Pc) = ṁ_injector(P_tank, Pc) - Pc · At / C*
```

Derivative computed by finite difference. Typically converges in <10 iterations.

---

## 6. Theory: 1D Isentropic Nozzle — `nozzle_analysis.py`

### 6.1 Assumptions

- **Steady, 1D flow** — all properties uniform across each cross-section
- **Isentropic** — no friction, no heat transfer, no shocks
- **Ideal gas** — constant γ, constant R
- **Choked throat** — M = 1 at minimum area

These assumptions are the foundation of classical rocket nozzle design (Huzel & Huang). Real performance is 1–5% lower due to boundary layer losses, combustion inefficiency, and divergence.

### 6.2 Isentropic relations

For Mach number M and ratio of specific heats γ:

```
T/T₀  = [1 + (γ-1)/2 · M²]⁻¹

P/P₀  = (T/T₀)^(γ/(γ-1))

ρ/ρ₀  = (T/T₀)^(1/(γ-1))
```

At the throat (M = 1):
```
T*  = T₀ · 2/(γ+1)
P*  = P₀ · (2/(γ+1))^(γ/(γ-1))
ρ*  = ρ₀ · (2/(γ+1))^(1/(γ-1))
V*  = √(γ · R · T*)   (= speed of sound at throat)
```

### 6.3 Area-Mach relation

The cross-sectional area required to sustain a given Mach number is:

```
A/A* = (1/M) · [(2/(γ+1)) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1)))
```

This is monotonically decreasing for subsonic flow and monotonically increasing for supersonic flow—which is why a converging-diverging nozzle works.

### 6.4 Mass flow (choked throat)

```
ṁ = ρ* · V* · At = P₀ · At · Γ / √(R · T₀)
```

where Γ = √γ · (2/(γ+1))^((γ+1)/(2(γ-1))). This is the maximum mass flow the nozzle can pass—adding more propellant only raises chamber pressure.

### 6.5 Thrust and Isp

Vacuum thrust (no ambient pressure subtraction):
```
F_vac = ṁ · Ve + Pe · Ae
```

Vacuum specific impulse (propellant efficiency metric):
```
Isp_vac = F_vac / (ṁ · g₀)   [seconds]
```

Isp is independent of engine scale—it depends only on propellant chemistry and nozzle expansion ratio.

### 6.6 Divergence loss

A straight-cone nozzle has a divergence angle θ. The gas does not all exit axially, so the net axial thrust is:

```
λ = 0.5 · (1 + cos θ)
```

For a 15° cone: λ = 0.983 (1.7% thrust loss). The Rao contour eliminates most of this.

---

## 7. Theory: Rao Optimum Contour — `moc.py`

### 7.1 Why the nozzle shape matters

A straight-cone nozzle wastes thrust through divergence losses—the gas near the wall exits at a large angle to the nozzle axis. The **Rao Thrust-Optimized Contour (TOC)** shapes the wall to turn the gas to axial flow at the exit, maximizing thrust for a given length.

### 7.2 Contour geometry

The Rao contour has two sections:

**Section 1: Circular arc throat**

Starting from the throat (x = 0), the wall follows a circular arc of radius:
```
R_arc = 0.382 · Rt   (Rao standard)
```

The arc sweeps from 0° to θᵢ (the inflection angle), giving:
```
x(θ) = R_arc · sin(θ)
r(θ) = Rt + R_arc · (1 - cos(θ))
```

**Section 2: Hermite cubic divergent section**

From the inflection point (xᵢ, rᵢ) to the exit (xe, Re), the wall follows a Hermite cubic spline:

```
r(t) = h₀₀(t)·rᵢ + h₁₀(t)·m₀ + h₀₁(t)·Re + h₁₁(t)·m₁
```

where t ∈ [0,1] and the cubic basis functions are:
```
h₀₀(t) = 2t³ - 3t² + 1    (value at start)
h₁₀(t) = t³ - 2t² + t     (tangent at start)
h₀₁(t) = -2t³ + 3t²       (value at end)
h₁₁(t) = t³ - t²          (tangent at end)
```

Tangent slopes encode the wall angles:
```
m₀ = tan(θᵢ) · dx   (slope at inflection point)
m₁ = tan(θₑ) · dx   (slope at exit)
```

### 7.3 Prandtl-Meyer function

The method-of-characteristics background uses the Prandtl-Meyer function ν(M) to relate Mach number to flow turning angle in supersonic flow:

```
ν(M) = √((γ+1)/(γ-1)) · arctan(√((γ-1)/(γ+1) · (M²-1))) - arctan(√(M²-1))
```

### 7.4 Design variables

| Variable | Range | Effect |
|----------|-------|--------|
| θᵢ | 15–45° | Controls curvature near throat; larger = more aggressive turn |
| θₑ | 2–15° | Exit angle; smaller = more axial flow = higher λ |
| n_lines | 5–20 | Arc resolution; affects smoothness |

Typical optimal: θᵢ ≈ 25–35°, θₑ ≈ 2–6°. Improvement over 15° cone: ~1.5–2%.

---

## 8. Theory: Shape Optimization — `nozzle_optimizer.py`

### 8.1 Problem statement

Maximize vacuum Isp subject to a maximum nozzle length constraint:

```
maximize    Isp_vac(θᵢ, θₑ, n, [O/F])
subject to  L(θᵢ, θₑ, n) ≤ L_max
```

The length constraint is enforced as a penalty:
```
cost = -Isp_vac + max(0, L - L_max) · 10⁶
```

### 8.2 Differential evolution (global search)

Differential Evolution (DE) is a population-based stochastic algorithm that works well for non-smooth, multi-modal objectives without gradients.

Each generation:
1. For each member **x** in population, create trial vector **u** by combining **x** with two random members
2. Accept **u** if f(**u**) < f(**x**)
3. Repeat until convergence

Population size: `popsize × n_params` (default popsize ≈ n_eval/15).

DE finds the global basin efficiently but converges slowly near the optimum.

### 8.3 Nelder-Mead polish (local refinement)

After DE, Nelder-Mead simplex refinement polishes the solution. It works by reflecting, expanding, or contracting a simplex of n+1 points toward the minimum.

**Why two stages?**
- DE explores the global landscape but is slow near convergence
- Nelder-Mead converges fast near a smooth minimum but can get stuck in local optima
- Together they give reliable global optimization with high final accuracy

---

## 9. Theory: Material Properties — `materials.py`

### 9.1 Why temperature-dependent properties matter

A regeneratively cooled chamber sees a wall temperature gradient from ~500 K (coolant side) to ~800–1200 K (gas side). Using room-temperature material properties introduces significant errors:

- **Copper (CuCrZr)**: thermal conductivity drops ~25% from 300 K to 800 K
- **Inconel 718**: tensile strength drops >50% from 300 K to 900 K
- **Carbon-carbon**: strength actually *increases* with temperature up to ~2000 K

### 9.2 Available materials

| Material | Application | T_melt [K] | ρ [kg/m³] |
|----------|-------------|------------|-----------|
| CuCrZr | Chamber/throat liner (regen-cooled) | 1350 | 8900 |
| OFHC-Cu | High-conductivity applications | 1356 | 8940 |
| SS304 | General structure | 1700 | 8000 |
| SS316 | Corrosion-resistant structure | 1670 | 8000 |
| Inconel718 | Hot-section structure, turbopump | 1600 | 8190 |
| CarbonCarbon | Radiation-cooled nozzle extension | 3800 | 1700 |
| NbC103 | Radiation-cooled small thrusters | 2620 | 8860 |

### 9.3 Interpolation

All temperature-dependent properties use `numpy.interp` (linear interpolation, clamped at edge values). Data tabulated from MIL-HDBK-5J and ASM International.

---

## 10. Theory: Regenerative Cooling — `cooling.py`

### 10.1 Why regenerative cooling is necessary

At chamber pressures of 5–10 MPa, the hot gas temperature is 3000–3600 K—far above the melting point of any structural metal. The solution is to route a coolant (usually the fuel itself) through channels in the chamber wall, absorbing heat before injection.

The heat flux at the throat is typically **5–50 MW/m²**—comparable to a nuclear reactor. Without cooling, the wall would melt in seconds.

### 10.2 Bartz correlation (gas-side heat transfer)

The gas-side heat transfer coefficient h_g is computed using the Bartz correlation (1957):

```
h_g = (0.026 / D_t^0.2) · (μ₀^0.2 · cₚ / Pr^0.6) · (P₀/C*)^0.8 · (D_t/R_c)^0.1 · (At/A(x))^0.9 · σ
```

where σ is the Bartz correction factor for wall temperature effects:

```
σ = [0.5·(Tw/T₀)·(1 + (γ-1)/2·M²) + 0.5]^(-0.68) · [1 + (γ-1)/2·M²]^(-0.12)
```

The `(At/A(x))^0.9` term shows that heat flux is highest at the **throat**, where the area is smallest and velocity (and heat transfer) is maximum.

### 10.3 Dittus-Boelter correlation (coolant-side heat transfer)

Inside the cooling channels, turbulent forced convection:

```
Nu = 0.023 · Re^0.8 · Pr^0.4

h_c = Nu · k_cool / D_h
```

where Re = ρ·v·D_h/μ is the Reynolds number and D_h = 4·A_ch/P_ch is the hydraulic diameter.

Higher coolant velocity → higher Re → higher h_c → lower wall temperature. But higher velocity also means higher pressure drop.

### 10.4 Wall temperature (three-layer resistance)

The wall separates the hot gas from the coolant. In steady state, the same heat flux flows through all three layers:

```
q = (T_aw - T_cool) / (1/h_g + t_wall/k_wall + 1/h_c)
```

where T_aw is the **adiabatic wall temperature** (what the wall would reach if perfectly insulated):

```
T_aw = T₀ · (1 + r·(γ-1)/2·M²) / (1 + (γ-1)/2·M²)
```

with recovery factor r = Pr^(1/3) for turbulent flow.

Gas-side wall temperature:
```
T_wg = T_aw - q / h_g
```

Coolant-side wall temperature:
```
T_wc = T_cool + q / h_c
```

### 10.5 Axial marching

The solver marches from nozzle exit (cold coolant inlet) back toward the injector, updating coolant temperature at each step:

```
dT_cool/dx = q · (2π·r) · dx / (ṁ_cool · cₚ_cool)
```

The `σ` iteration: T_wg appears inside the Bartz σ correction—a fixed-point iteration (15 steps) is used at each axial station.

### 10.6 Structural margin

Hoop stress in the thin-wall approximation:
```
σ_applied = Pc · r / t_wall
```

Safety margin:
```
margin = UTS(T_wg) / σ_applied
```

If margin < 1 anywhere, the wall will fail. Typical design target: margin > 1.5.

---

## 11. Theory: Propellant Tank (Blowdown) — `tank.py`

### 11.1 Pressure-fed blowdown

In a **pressure-fed** engine, the propellant is forced to the injector by pressurant gas (usually helium) in the tank ullage. As propellant is expelled, the ullage volume grows and the pressure drops—this is the **blowdown**.

### 11.2 Polytropic expansion model

The ullage gas expands polytropic-ally:

```
P(t) = P₀ · (V_ull_0 / V_ull(t))^n
```

where n is the polytropic exponent:
- **n = γ** (isentropic): fast blowdown with large pressure drop — accurate for short burns
- **n = 1** (isothermal): heat exchanges with surroundings — accurate for slow burns
- **n = 1.2–1.4**: typical practical range

Ullage volume at time t:
```
V_ull(t) = V_ull_0 + m_expelled(t) / ρ_prop
```

### 11.3 Tank sizing rule of thumb

Initial ullage fraction: 5% of tank volume. This allows for thermal expansion and ensures the outlet remains submerged.

### 11.4 Effect on engine operation

As tank pressure drops, the injector ΔP decreases, reducing mass flow and potentially shifting the O/F ratio (since fuel and oxidizer densities differ). This is the main reason pressure-fed engines have shorter burn times at rated thrust.

---

## 12. Theory: Turbopump — `turbopump.py`

### 12.1 Why turbopumps?

A pressure-fed engine needs tank pressures of 1.5–2× the chamber pressure to drive the injector. For a 10 MPa chamber, that means 15–20 MPa tanks—massively heavy pressure vessels.

A turbopump raises the propellant pressure mechanically, allowing low-pressure tanks (0.2–0.5 MPa), saving enormous mass.

### 12.2 Centrifugal pump model

The pump is modeled with a parabolic head-flow curve:

```
H(Q) = H_design · [1 - k_H · (Q/Q_design - 1)²]
```

Pressure rise:
```
ΔP = ρ · g · H(Q)
```

Shaft power required:
```
P_pump = ρ · g · Q · H(Q) / η(Q)
```

Efficiency falls off parabolically from the design point.

### 12.3 Specific speed and affinity laws

Centrifugal pump similarity (at constant impeller diameter):
```
Q ~ N          (flow scales with speed)
H ~ N²         (head scales with speed²)
P ~ N³         (power scales with speed³)
```

High specific speed pumps (Ns > 1000) → axial/mixed flow, low head, high flow (typical for LOX pump).
Low specific speed pumps (Ns < 500) → radial flow, high head, low flow (typical for H2 pump, which needs very high head due to low density).

### 12.4 NPSH — cavitation prevention

Net Positive Suction Head Available:
```
NPSH_a = (P_inlet - P_vapor) / (ρ·g) + z_head - h_loss
```

If NPSH_a < NPSH_required, the fluid vaporizes at the pump inlet (cavitation), causing vibration, erosion, and loss of flow.

Cryogenic propellants (LH2, LOX) are close to their boiling points in the tank—they have very little NPSH margin. This drives the requirement for inducer stages in rocket turbopumps.

### 12.5 Turbine power

The turbine converts enthalpy of hot gas into shaft power:

```
P_turbine = ṁ_t · cₚ · T_in · η_t · [1 - (P_out/P_in)^((γ-1)/γ)]
```

where η_t = isentropic efficiency (50–80%). The turbine outlet connects to either the chamber (staged combustion) or overboard (gas generator).

---

## 13. Theory: Feed System & Engine Cycles — `feed_system.py`

### 13.1 Pipe friction losses

**Darcy-Weisbach equation:**
```
ΔP_friction = f · (L/D) · (ρ·v²/2)
```

Friction factor via Swamee-Jain (explicit approximation to Colebrook):
```
f = 0.25 / [log₁₀(ε/(3.7·D) + 5.74/Re^0.9)]²
```

For laminar flow (Re < 2300): f = 64/Re.

Typical roughness: drawn tubing ε = 4.5×10⁻⁵ m; commercial steel ε = 4.6×10⁻⁵ m.

**Fittings (K-factor method):**
```
ΔP_fitting = K · ρ·v²/2
```

| Fitting | K |
|---------|---|
| 90° elbow | 0.9 |
| Ball valve (open) | 0.05 |
| Check valve | 2.0 |
| Entrance | 0.5 |

### 13.2 Gas generator cycle

The most common turbopump cycle (used in F-1, Merlin, RD-170):

```
┌─────────┐    small bleed    ┌──────────────┐
│  Tanks  │──────────────────►│ Gas Generator│
└────┬────┘                   └──────┬───────┘
     │                               │ hot gas
     │    pumps                      ▼
     ├───────────────────────►  ┌──────────┐
     │                          │  Turbine │
     │                          └────┬─────┘
     │   injector                    │ exhaust (overboard)
     ▼                               ▼
┌─────────────────────────────────────────┐
│           Main Chamber                  │
└─────────────────────────────────────────┘
```

- 1–5% of propellant is burned in the gas generator to drive the turbine
- Turbine exhaust is dumped overboard (efficiency penalty)
- Simple and reliable
- Isp penalty from bleed: ~5–10 s

### 13.3 Expander cycle

Used in Vinci (Ariane 6), RL-10 (Atlas, Delta):

```
┌─────────┐                ┌──────────────────────┐
│  Tanks  │───────────────►│  Cooling Jacket (regen)│
└────┬────┘                └───────────┬───────────┘
     │                                 │ hot fuel
     │    pumps                        ▼
     ├──────────────────────────►  ┌──────────┐
     │                              │  Turbine │
     │                              └────┬─────┘
     │    injector                       │ to chamber
     ▼                                   ▼
┌─────────────────────────────────────────────┐
│               Main Chamber                   │
└──────────────────────────────────────────────┘
```

- No gas generator needed — turbine driven by fuel heated in cooling jacket
- Higher Isp than GG cycle (no bleed loss)
- Limited to moderate chamber pressures (turbine needs enthalpy from cooling jacket)
- LOX/H2 only: hydrogen has very high cₚ, making it ideal as the working fluid

### 13.4 Pressure budget

The entire feed system must satisfy the pressure balance:

```
P_tank + ΔP_pump ≥ Pc + ΔP_line + ΔP_injector
```

Each term:
| Term | Typical value |
|------|---------------|
| P_tank (pressure-fed) | 1.5–2× Pc |
| P_tank (turbopump) | 0.2–0.5 MPa |
| ΔP_pump | 0 (pressure-fed) or ~Pc + losses - P_tank |
| ΔP_line | 1–5% of Pc |
| ΔP_injector | 10–20% of Pc |

---

## 14. System Integration — `engine_system.py`

The main entry point. Three modes:

### Mode: `simulate`

Runs the full pipeline once at the design point:
1. Build `Injector` from orifice parameters
2. `EngineSolver.solve()` → Pc, T0, γ, R, C*, mdot
3. `nozzle_perf()` with Rao contour → Isp, Thrust, wall
4. Generates 6-panel plot + JSON + Markdown log

```bash
python engine_system.py --mode simulate \
    --fuel H2 --oxidizer O2 \
    --throat_radius 0.04 \
    --p_tank_f 8e6 --p_tank_o 8e6 \
    --A_f 2e-4 --A_o 2e-4 \
    --name my_engine_01
```

### Mode: `optimize`

Finds the optimal Rao contour shape:
1. Runs `simulate` to get the engine operating point
2. Calls `nozzle_optimizer.run_optimizer()` with DE + Nelder-Mead
3. Saves optimal contour, convergence plot, and comparison vs straight cone

```bash
python engine_system.py --mode optimize \
    --fuel H2 --oxidizer O2 \
    --throat_radius 0.04 \
    --p_tank_f 8e6 --p_tank_o 8e6 \
    --A_f 2e-4 --A_o 2e-4 \
    --n_eval 150 --name my_opt_01
```

### Mode: `sweep`

Sweeps one parameter (tank pressure, orifice area, etc.) and plots Isp sensitivity.

---

## 15. Quick-Start Examples

### Example 1: Simple LOX/H2 simulation

```python
from injector import Injector
from engine_solver import EngineSolver
from engine_system import nozzle_perf
import numpy as np

inj = Injector(fuel="H2", oxidizer="O2", A_f=2e-4, A_o=2e-4)
solver = EngineSolver(At=np.pi*0.04**2, injector=inj, fuel="H2", oxidizer="O2")

eng = solver.solve(p_tank_f=8e6, p_tank_o=8e6)
print(f"Pc={eng['pc']/1e6:.2f} MPa  OF={eng['OF']:.2f}  Isp_ideal≈{eng['Cstar']/9.8:.0f}s")

noz = nozzle_perf(eng, throat_radius=0.04, Pe=101325)
print(f"Isp_vac={noz['Isp_vac']:.1f}s  Thrust={noz['Thrust_vac']:.0f}N")
```

### Example 2: Regenerative cooling analysis

```python
from cooling import CoolingChannelGeometry, CoolantProperties, solve_cooling_loop
from moc import rao_contour
from nozzle_analysis import mach_from_pressure_ratio

# Build contour
Me = mach_from_pressure_ratio(101325/eng['pc'], eng['gamma'])
wall = rao_contour(eng['gamma'], 0.04, Me, theta_i_deg=30, theta_e_deg=5, n_lines=10)

# Set up cooling
geom = CoolingChannelGeometry(n_channels=80, width=1.5e-3, height=3e-3, wall_thickness=1e-3)
coolant = CoolantProperties("H2")

result = solve_cooling_loop(
    engine_result=eng,
    nozzle_result=noz,
    wall_contour=wall,
    channel_geom=geom,
    coolant=coolant,
    wall_material="CuCrZr",
    mdot_coolant=eng['mdot_f'],  # fuel-cooled
)

print(f"Max wall temp: {result['max_T_wg']:.0f} K")
print(f"Coolant ΔT:    {result['T_cool_exit'] - coolant.T_boil:.0f} K")
print(f"Heat load:     {result['total_heat']/1e6:.2f} MW")
print(f"Coolant ΔP:    {result['dP_total']/1e6:.2f} MPa")
print(f"Min margin:    {min(result['margin']):.2f}x")
```

### Example 3: Tank blowdown curve

```python
from tank import Tank
import matplotlib.pyplot as plt

tank = Tank(V_tank=0.05, P_0=8e6, propellant="LOX", pressurant="He",
            ullage_fraction=0.05)
print(f"Initial propellant: {tank.m_prop_0:.2f} kg")

bd = tank.blowdown_curve(mdot=0.5, t_burn=tank.m_prop_0/0.5)
plt.plot(bd['t'], bd['P']/1e6)
plt.xlabel("Time [s]");  plt.ylabel("Tank pressure [MPa]")
plt.title("Oxidizer tank blowdown")
plt.savefig("blowdown.png", dpi=120)
```

### Example 4: Gas generator cycle

```python
from turbopump import Pump, Turbine
from feed_system import FeedLine, GasGeneratorCycle

feed_ox   = FeedLine(D=0.025, L=2.5, propellant="LOX")
feed_fuel = FeedLine(D=0.020, L=3.0, propellant="H2")
pump_o    = Pump(eta_design=0.70)
pump_f    = Pump(eta_design=0.65)
turb      = Turbine(eta_turbine=0.60)

cycle = GasGeneratorCycle(pump_o, pump_f, turb, feed_ox, feed_fuel,
                           T_gg=900, gamma_gg=1.3, cp_gg=2500)
result = cycle.solve(P_tank_ox=0.4e6, P_tank_fuel=0.4e6,
                     P_chamber=eng['pc'], mdot_total=eng['mdot_total'],
                     OF=eng['OF'])

print(f"GG bleed:   {result['bleed_fraction']*100:.1f}%")
print(f"Pump power: {(result['P_pump_ox']+result['P_pump_fuel'])/1e3:.0f} kW")
print(f"NPSH ox:    {result['NPSH_ox']['margin']:.2f}x")
```

---

## 16. Parameter Reference

### `engine_system.py` (CLI flags)

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | engine_01 | Run name (must be unique, used as output folder) |
| `--fuel` | H2 | Fuel species name |
| `--oxidizer` | O2 | Oxidizer species name |
| `--throat_radius` | 0.04 m | Throat radius [m] |
| `--Pe` | 101325 Pa | Nozzle exit pressure [Pa] |
| `--max_length_m` | 0.5 m | Maximum nozzle length constraint [m] |
| `--p_tank_f` | 8×10⁶ Pa | Fuel tank pressure [Pa] |
| `--p_tank_o` | 8×10⁶ Pa | Oxidizer tank pressure [Pa] |
| `--A_f` | 2×10⁻⁴ m² | Fuel injector total orifice area [m²] |
| `--A_o` | 2×10⁻⁴ m² | Oxidizer injector total orifice area [m²] |
| `--Cd_f` | 0.8 | Fuel discharge coefficient |
| `--Cd_o` | 0.8 | Oxidizer discharge coefficient |
| `--n_eval` | 80 | Optimizer evaluation budget (mode=optimize) |
| `--mode` | simulate | `simulate` / `optimize` / `sweep` |

### `cooling.py` — `CoolingChannelGeometry`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_channels` | 100 | Number of channels around circumference |
| `width` | 1.5 mm | Channel width [m] |
| `height` | 2.5 mm | Channel radial depth [m] |
| `wall_thickness` | 1.0 mm | Gas-side wall thickness [m] |
| `land_width` | 1.0 mm | Rib width between channels [m] |
| `roughness` | 10 μm | Channel surface roughness [m] |

### `tank.py` — `Tank`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `V_tank` | — | Total tank volume [m³] |
| `P_0` | — | Initial tank pressure [Pa] |
| `propellant` | LOX | Propellant name (for density lookup) |
| `ullage_fraction` | 0.05 | Initial ullage / tank volume [-] |
| `pressurant` | He | Pressurant gas (He / N2 / Ar) |
| `T_pressurant` | 300 K | Pressurant initial temperature [K] |
| `polytropic_n` | None | Polytropic exponent (None = γ of pressurant) |

### `turbopump.py` — `Pump`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `H_design` | 500 m | Design-point head [m] |
| `Q_design` | 0.01 m³/s | Design-point volumetric flow [m³/s] |
| `eta_design` | 0.70 | Design efficiency [-] |
| `k_H` | 0.3 | Head droop coefficient [-] |
| `NPSH_required` | 10 m | Required NPSH [m] |

---

## 17. Output Files Reference

Each run creates a folder `results/<name>/` containing:

| File | Description |
|------|-------------|
| `log.nozzle_<name>` | Markdown with YAML frontmatter — input for LLM_based_DB |
| `results.json` | All inputs, engine state, and performance metrics |
| `contour.csv` | Wall coordinates `x_m, r_m` (for CFD mesh) |
| `engine_plots.png` | 6-panel system-level plot |
| `optimization_history.csv` | Per-evaluation θᵢ, θₑ, n, Isp log |
| `optimizer_plots.png` | 6-panel optimizer convergence and results plot |

### `results.json` structure

```json
{
  "run_name": "lox_h2_01",
  "timestamp": "2025-04-09T...",
  "engine": {
    "pc": 6248000.0,
    "T0": 3412.0,
    "gamma": 1.21,
    "R": 414.2,
    "Cstar": 2381.0,
    "OF": 4.01,
    "mdot_total": 0.9823
  },
  "nozzle": {
    "Isp_vac": 430.96,
    "Thrust_vac": 4152.0,
    "Me": 3.82,
    "Ae_At": 6.42
  }
}
```

---

## 18. Common Propellant Reference Data

| Propellant Pair | O/F_opt | T₀ [K] | γ | C* [m/s] | Isp_vac [s] |
|-----------------|---------|--------|---|----------|-------------|
| LOX / LH2 | 6.0 | 3500 | 1.20 | 2390 | 450 |
| LOX / CH4 | 3.4 | 3550 | 1.22 | 1860 | 380 |
| LOX / RP-1 | 2.6 | 3670 | 1.24 | 1770 | 358 |
| N2O4 / UDMH | 2.0 | 3100 | 1.25 | 1720 | 340 |
| LOX / C2H6 | 2.8 | 3480 | 1.22 | 1810 | 365 |

Values are approximate at P₀ = 7 MPa, Ae/At ≈ 40 (vacuum nozzle).

---

## Dependencies

```bash
pip install numpy scipy matplotlib pandas cantera
```

- **Python** 3.10+
- **Cantera** 3.x — combustion chemistry
- **SciPy** — optimization (DE, Nelder-Mead, solve_ivp)
- **NumPy** — array math
- **Matplotlib** — plotting (headless, Agg backend)
- **Pandas** — optimization history DataFrame

---

## License

For educational and research use. Physical constants and correlations from:
- Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed.
- Huzel & Huang, *Modern Engineering for Design of Liquid-Propellant Rocket Engines*
- Bartz (1957), *A Simple Equation for Rapid Estimation of Rocket Nozzle Convective Heat Transfer Coefficients*
- Rao (1958), *Exhaust Nozzle Contour for Optimum Thrust*
