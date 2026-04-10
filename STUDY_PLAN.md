# Rocket Engine Optimizer — Study Plan

A self-paced learning curriculum to master both the physics and the code.
Each module has: theory to understand → worked examples → experiments to run → questions to answer.

Estimated total time: **8–12 weeks** at ~5–8 hours/week.

---

## Prerequisites

- Python (NumPy, matplotlib) — intermediate level
- Basic calculus (derivatives, integrals, chain rule)
- High-school physics (Newton's laws, energy conservation)
- No prior aerospace or thermodynamics experience needed — everything is derived from scratch

---

## Module 1 — Thermodynamics of Combustion (Week 1–2)

### 1.1 Why thermodynamics matters for rockets

A rocket engine is fundamentally a heat engine: it converts chemical energy into
kinetic energy of the exhaust gas. Every performance number (thrust, Isp,
efficiency) traces back to three thermodynamic properties of the combustion gas:

| Property | Symbol | What it controls |
|----------|--------|-----------------|
| Temperature | T₀ | How much energy is available |
| Mean molecular weight | MW | How heavy the exhaust molecules are |
| Ratio of specific heats | γ | How efficiently we convert heat to velocity |

**Intuitive analogy:** Think of blowing air through a straw. Temperature is how
hard you blow. Molecular weight is how heavy the air is (imagine blowing sand
vs. helium). γ is how "springy" the gas is — does it expand easily or resist?

### 1.2 The Ideal Gas Law — where everything starts

You know `PV = nRT` from chemistry. In engineering, we rewrite it per unit mass:

```
PV = nRT          (chemistry form: n = moles, R = 8.314 J/mol·K)
                   ↓ divide both sides by mass m
P(V/m) = (n/m)RT
P·v = (1/MW)·R_u·T      where v = specific volume [m³/kg], MW = molar mass
P = ρ·(R_u/MW)·T
P = ρ·R·T          where R = R_u/MW is the SPECIFIC gas constant [J/kg·K]
```

**Why R matters:** R = R_u / MW. Lighter molecules → bigger R → higher exhaust velocity.

| Gas | MW [g/mol] | R [J/kg·K] |
|-----|-----------|------------|
| H₂O | 18.015 | 461.5 |
| CO₂ | 44.01 | 188.9 |
| H₂ | 2.016 | 4124 |

This is why hydrogen gives the best Isp: its exhaust products (mostly H₂O + excess H₂)
are light, giving a large R.

### 1.3 Enthalpy and why we use H, not U

**Internal energy U** counts only the energy stored inside molecules (kinetic
energy of vibration, rotation, translation). But in a flowing gas, molecules also
do work pushing other molecules out of their way — **pressure-volume work**.

**Enthalpy H** wraps both together:

```
H = U + PV

For an ideal gas with constant specific heats:
  h = u + Pv = cᵥT + RT = (cᵥ + R)T = cₚT

So:   h = cₚ·T     (enthalpy per unit mass)
```

**Where does cₚ = cᵥ + R come from?** It's a direct consequence of the ideal gas law:

```
Start:    h = u + Pv                    (definition)
For an ideal gas: Pv = RT,  u = cᵥT
So:       h = cᵥT + RT = (cᵥ + R)T
But also: h = cₚT                      (definition of cₚ for ideal gas)
Therefore: cₚ = cᵥ + R                 (Mayer's relation)
```

**The ratio of specific heats γ:**

```
γ = cₚ/cᵥ

From Mayer's relation: cₚ = cᵥ + R
  → cₚ/cᵥ = 1 + R/cᵥ = γ
  → cᵥ = R/(γ-1)
  → cₚ = γR/(γ-1)
```

For a monatomic gas (He, Ar): γ = 5/3 ≈ 1.667
For a diatomic gas (N₂, O₂): γ ≈ 1.4
For combustion products: γ ≈ 1.1–1.3 (complex molecules, more degrees of freedom)

### 1.4 Conservation of Energy in the Combustion Chamber

When fuel and oxidizer react, chemical bonds break and form. The energy released
is called the **heat of reaction ΔH_rxn**. In Cantera, we use **HP equilibrium**:

```
Given: reactants at known H and P
Find: products at the SAME H and P that minimize Gibbs free energy G

Reactants:   fuel (e.g. H₂) + oxidizer (e.g. O₂)
             H_reactants = Σ nᵢ·hf,ᵢ     (sum of formation enthalpies)

Products:    H₂O, OH, H, O, etc.
             H_products = H_reactants     (enthalpy conserved!)
             → This determines T₀, the adiabatic flame temperature
```

**Gibbs free energy minimization** is how nature "decides" which species form:

```
G = H - T·S

At equilibrium, G is minimized. No spontaneous reaction can lower G further.
This determines the concentration of each species (H₂O, OH, H, O, CO₂, CO...).

For example, at 3500 K:
  2H₂O ⇌ 2H₂ + O₂     (dissociation — absorbs energy, lowers T₀)

This is why T₀ doesn't just keep rising with more fuel — dissociation steals energy.
```

**Numerical example — by hand:**

Suppose we burn H₂ + ½O₂ → H₂O at 7 MPa.
- Heat of formation: H₂O(g) = −241.8 kJ/mol, H₂ = 0, O₂ = 0
- ΔH_rxn = −241.8 kJ/mol released
- Average cₚ of products ≈ 45 J/mol·K (varies with T, but let's estimate)
- ΔT = 241800 / 45 ≈ 5370 K above inlet temperature
- If reactants enter at 300 K: T₀ ≈ 5670 K (simplified, no dissociation)
- Real answer with dissociation at 7 MPa: T₀ ≈ 3430 K — dissociation eats ~40% of the energy!

### 1.5 Specific Impulse — the rocket efficiency number

**Specific impulse (Isp)** is the most important number in rocketry:

```
Isp = F / (ṁ · g₀)   [seconds]

Where:
  F = thrust [N]
  ṁ = mass flow rate [kg/s]
  g₀ = 9.80665 m/s² (standard gravity — just a unit conversion)
```

**What does it mean physically?** "How many seconds one kilogram of propellant can
produce one Newton of thrust." Alternatively: if you had 1 kg of propellant
and burned it slowly enough to produce exactly 1 N of thrust, how long would it last?

```
LOX/H₂:  Isp ≈ 450 s   → 1 kg lasts 450 seconds at 1 N
LOX/CH₄: Isp ≈ 360 s   → 1 kg lasts 360 seconds at 1 N
LOX/RP1: Isp ≈ 310 s   → 1 kg lasts 310 seconds at 1 N
```

**Why a 10% Isp improvement is enormous:** the Tsiolkovsky rocket equation:

```
Δv = Isp · g₀ · ln(m_initial / m_final)

For a rocket with Δv = 9.4 km/s (orbit) and Isp = 350 s:
  m_initial / m_final = exp(9400 / (350×9.81)) = exp(2.74) = 15.5
  → 93.5% of the rocket is propellant

Now increase Isp to 385 s (+10%):
  m_initial / m_final = exp(9400 / (385×9.81)) = exp(2.49) = 12.1
  → 91.7% is propellant

The payload fraction went from 6.5% to 8.3% — a 28% increase in payload!
```

### 1.6 Characteristic Velocity C*

**C* (pronounced "C-star")** isolates propellant performance from nozzle quality:

```
C* = P₀ · Aₜ / ṁ    [m/s]

Or equivalently (derivation in §2):
C* = √(R · T₀) / Γ

Where Γ = √γ · (2/(γ+1))^((γ+1)/(2(γ-1)))
```

**What C* tells you:** if two engines have the same C*, their propellant chemistry
is equally good. Different C* means different propellant performance regardless
of nozzle design. C* only depends on T₀, MW, and γ.

**Numerical example:**

```
LOX/H₂ at O/F=6, Pc=7 MPa:
  T₀ = 3430 K, γ = 1.20, R = 461.5 J/kg·K
  Γ = √1.20 · (2/2.20)^(2.20/0.40) = 1.095 · 0.9091^5.5 = 1.095 · 0.5824 = 0.6378
  C* = √(461.5 × 3430) / 0.6378 = √1,582,945 / 0.6378 = 1258.2 / 0.6378 = 1972 m/s
```

### 1.7 Why the Optimal O/F ≠ Stoichiometric O/F

Stoichiometric O/F gives maximum temperature (all fuel burned, maximum heat released).
But Isp depends on **both** temperature and molecular weight:

```
Isp ∝ √(T₀/MW)   (simplified)

Stoichiometric LOX/H₂:  O/F = 8.0, T₀ ≈ 3500 K, MW ≈ 18 → √(3500/18) = 13.9
Fuel-rich LOX/H₂:       O/F = 4.0, T₀ ≈ 2900 K, MW ≈ 10 → √(2900/10) = 17.0  ← higher!

The excess hydrogen (MW=2) dramatically lowers the average MW, more than
compensating for the lower temperature. That's why LOX/H₂ engines run fuel-rich.
```

For LOX/CH₄, the effect is smaller because excess CH₄ (MW=16) is much heavier
than excess H₂ (MW=2), so optimal O/F is closer to stoichiometric.

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
MWs = [r["MW"] for r in valid]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(OFs, T0s);    axes[0,0].set(xlabel="O/F", ylabel="T₀ [K]", title="Flame temperature")
axes[0,1].plot(OFs, Rs);     axes[0,1].set(xlabel="O/F", ylabel="R [J/kgK]", title="Gas constant R = Rᵤ/MW")
axes[1,0].plot(OFs, MWs);    axes[1,0].set(xlabel="O/F", ylabel="MW [g/mol]", title="Mean molecular weight")
axes[1,1].plot(OFs, Cstars); axes[1,1].set(xlabel="O/F", ylabel="C* [m/s]", title="Characteristic velocity")
for ax in axes.flat: ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("study_1_1.png", dpi=120)
print("Open study_1_1.png. Notice: T₀ peaks near O/F=8, but C* peaks near O/F=5.")
print("That shift is because lighter molecules (lower MW) at O/F=5 give better C*.")
```

**Experiment 1.2 — Compare propellant pairs**
```python
from chamber import of_sweep

pairs = [
    ("O2", "H2",   2.0, 10.0),   # Highest Isp, cryogenic, expensive
    ("O2", "CH4",  1.5,  6.0),   # Modern choice (Raptor, Aquila)
    ("O2", "C3H8", 1.0,  5.0),   # Propane — good density, moderate Isp
]
print(f"{'Pair':>12s}  {'Best O/F':>8s}  {'T₀ [K]':>7s}  {'C* [m/s]':>8s}  {'MW':>5s}  {'γ':>5s}")
print("-" * 60)
for ox, fuel, lo, hi in pairs:
    results = of_sweep(ox, fuel, P0=7e6, OF_min=lo, OF_max=hi, n_points=20)
    valid = [r for r in results if "T0" in r]
    best = max(valid, key=lambda r: r["Cstar"])
    print(f"{ox}/{fuel:>4s}  {best['OF']:8.2f}  {best['T0']:7.0f}  {best['Cstar']:8.0f}  "
          f"{best['MW']:5.1f}  {best['gamma']:5.3f}")
```

**Experiment 1.3 — Effect of chamber pressure on C***
```python
from chamber import chamber_conditions

pressures = [1e6, 2e6, 5e6, 10e6, 20e6, 30e6]
print(f"{'Pc [MPa]':>10s}  {'T₀ [K]':>7s}  {'γ':>6s}  {'C* [m/s]':>8s}  {'MW':>5s}")
print("-" * 50)
for P0 in pressures:
    c = chamber_conditions("O2", "H2", OF=6.0, P0=P0)
    print(f"{P0/1e6:10.0f}  {c['T0']:7.0f}  {c['gamma']:6.4f}  {c['Cstar']:8.0f}  {c['MW']:5.1f}")
print()
print("Notice: C* increases slightly with Pc. Why?")
print("At higher pressure, dissociation is SUPPRESSED (Le Chatelier's principle).")
print("Less dissociation → more complete combustion → higher T₀ → higher C*.")
```

**Experiment 1.4 — Hand-calculate and verify (do this on paper first!)**
```python
# Verify your hand calculation of C* from §1.6
import numpy as np
from chamber import chamber_conditions

c = chamber_conditions("O2", "H2", OF=6.0, P0=7e6)
T0, gamma, R = c["T0"], c["gamma"], c["R"]

# Your hand calculation:
Gamma = np.sqrt(gamma) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
Cstar_hand = np.sqrt(R * T0) / Gamma

print(f"T₀ = {T0:.0f} K,  γ = {gamma:.4f},  R = {R:.1f} J/kg·K")
print(f"Γ = {Gamma:.4f}")
print(f"C* (hand)   = {Cstar_hand:.1f} m/s")
print(f"C* (Cantera) = {c['Cstar']:.1f} m/s")
print(f"Difference: {abs(Cstar_hand - c['Cstar'])/c['Cstar']*100:.2f}%")
print("(Small difference is because Cantera uses variable γ, not constant)")
```

**Experiment 1.5 — Visualize dissociation**
```python
# See how dissociation limits flame temperature
import numpy as np
import matplotlib.pyplot as plt
from chamber import chamber_conditions

# Compare low vs high pressure (high P suppresses dissociation)
pressures = [0.5e6, 2e6, 7e6, 20e6]
OFs = np.linspace(3.0, 10.0, 25)

plt.figure(figsize=(10, 5))
for P0 in pressures:
    T0s = []
    for of in OFs:
        try:
            c = chamber_conditions("O2", "H2", OF=of, P0=P0)
            T0s.append(c["T0"])
        except:
            T0s.append(np.nan)
    plt.plot(OFs, T0s, label=f"Pc = {P0/1e6:.1f} MPa")

plt.xlabel("O/F ratio"); plt.ylabel("T₀ [K]")
plt.title("Flame temperature: higher Pc → less dissociation → higher T₀")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("study_1_5.png", dpi=120)
```

### Questions to answer

1. Why does LOX/H₂ have higher Isp than LOX/CH₄ even though LOX/CH₄ has a similar flame temperature?
   (Hint: compute MW for each and use Isp ∝ √(T₀/MW).)
2. In Experiment 1.3, how much does C* change between 1 MPa and 30 MPa? Explain why using Le Chatelier's principle.
3. What is the physical meaning of C*? If two engines have the same C* but different nozzles, which produces more thrust?
4. Why does the optimal O/F for best C* differ from stoichiometric O/F?
5. If you could burn Li + F₂ (lithium + fluorine), would you expect higher or lower Isp than H₂ + O₂? Why?
   (Hint: LiF has MW = 26, Hf° = −616 kJ/mol. Compare with H₂O: MW = 18, Hf° = −242 kJ/mol.)
6. **Hand calculation:** For LOX/CH₄ at O/F = 3.5, Pc = 7 MPa, the code gives T₀ = 3530 K, γ = 1.15, R = 348 J/kg·K. Calculate C* by hand and compare with the code output.

---

## Module 2 — Isentropic Nozzle Flow (Week 3)

### 2.1 The Big Idea: Converting Heat to Speed

The nozzle is the most critical part of the engine. Its job: convert the
high-pressure, high-temperature gas in the chamber into a high-velocity jet.

**Analogy:** Think of a garden hose with a nozzle. When you squeeze the end,
the water speeds up. A rocket nozzle does the same thing, but with
supersonic gas — and the physics are much more subtle.

### 2.2 Deriving the Energy Equation

Start with **conservation of energy** for a flowing gas (no heat transfer,
no work — the gas just flows through a shaped tube):

```
Total energy in = Total energy out

For a steady flow between two points:
  h₁ + V₁²/2 = h₂ + V₂²/2

(enthalpy + kinetic energy = constant along a streamline)
```

Define **stagnation (total) enthalpy** h₀ as the enthalpy when V = 0:

```
h₀ = h + V²/2         (definition)

For an ideal gas (h = cₚT):
  cₚT₀ = cₚT + V²/2

Solve for T:
  T = T₀ - V²/(2cₚ)    →  as V ↑, T ↓  (kinetic energy comes from thermal energy)
```

Now introduce Mach number M = V/a, where a = √(γRT) is the speed of sound:

```
V² = M²·a² = M²·γRT

Substitute into energy equation:
  cₚT₀ = cₚT + M²γRT/2

Divide by cₚT:
  T₀/T = 1 + M²γR/(2cₚ)

Since cₚ = γR/(γ-1):
  γR/(2cₚ) = γR / (2·γR/(γ-1)) = (γ-1)/2

Therefore:
  ┌─────────────────────────────────────┐
  │  T/T₀ = [1 + (γ-1)/2 · M²]⁻¹      │  ← Temperature ratio
  └─────────────────────────────────────┘
```

**Numerical example:** At M = 2, γ = 1.2:

```
T/T₀ = [1 + 0.1 × 4]⁻¹ = 1/1.4 = 0.714
If T₀ = 3500 K → T = 2500 K

The gas cooled by 1000 K — that energy went into kinetic energy!
```

### 2.3 Deriving the Pressure Ratio

For an **isentropic** (no friction, no shocks) flow, pressure and temperature
are related by the isentropic relation:

```
P/P₀ = (T/T₀)^(γ/(γ-1))

Derivation from entropy:
  ds = cₚ dT/T - R dP/P = 0   (isentropic: ds = 0)
  → cₚ ln(T₂/T₁) = R ln(P₂/P₁)
  → ln(P₂/P₁) = (cₚ/R) ln(T₂/T₁)
  → P₂/P₁ = (T₂/T₁)^(cₚ/R)

Since cₚ/R = γ/(γ-1):
  ┌─────────────────────────────────────────────────┐
  │  P/P₀ = [1 + (γ-1)/2 · M²]^(-γ/(γ-1))         │  ← Pressure ratio
  └─────────────────────────────────────────────────┘
```

**Numerical example:** At M = 2, γ = 1.2:

```
P/P₀ = (0.714)^(1.2/0.2) = (0.714)^6 = 0.132
If P₀ = 7 MPa → P = 0.924 MPa

The pressure dropped by 87%! Most of that pressure pushed the gas to Mach 2.
```

### 2.4 The Area-Mach Relation — Why a Converging-Diverging Shape?

This is the key equation that explains why rockets need a "waist" (throat).

Start with **continuity** (mass conservation):

```
ṁ = ρ · A · V = constant

Take the logarithmic differential:
  dρ/ρ + dA/A + dV/V = 0    ... (1)
```

From the **momentum equation** (Euler's equation) for 1D steady flow:

```
ρV dV = -dP
→ dV/V = -dP/(ρV²) = -(1/M²)·dρ/ρ    (using dP = a²dρ for isentropic flow)
... (2)
```

Substitute (2) into (1):

```
dρ/ρ + dA/A - (1/M²)·dρ/ρ = 0
dρ/ρ·(1 - 1/M²) = -dA/A
```

From (2): dρ/ρ = -M²·dV/V, so:

```
-M²·(dV/V)·(1 - 1/M²) = -dA/A
dV/V · (M² - 1) = dA/A

  ┌──────────────────────────────────────────────┐
  │  dA/A = (M² - 1) · dV/V                      │  ← Area-velocity relation
  └──────────────────────────────────────────────┘
```

**This equation is profound.** Read it carefully:

| Regime | M² - 1 | dA/A vs dV/V | Physical meaning |
|--------|--------|-------------|-----------------|
| Subsonic (M < 1) | Negative | Opposite signs | To speed up, DECREASE area (converging) |
| Sonic (M = 1) | Zero | dA = 0 | Area must be at a MINIMUM (throat) |
| Supersonic (M > 1) | Positive | Same sign | To speed up, INCREASE area (diverging) |

**Analogy:** It's like traffic on a highway:
- Subsonic = normal traffic: narrow the road → cars speed up (they bunch together)
- Supersonic = "opposite traffic": widen the road → cars speed up (density drops faster than area grows)
- The throat = the transition point where the rules flip

### 2.5 The Complete Area-Mach Relation

Integrating the above with isentropic relations gives the full equation:

```
Starting from ṁ = ρAV and using isentropic relations:

  ṁ = P₀/√(RT₀) · A · M · √γ · [1 + (γ-1)/2·M²]^(-(γ+1)/(2(γ-1)))

At the throat (M=1), A = A*:

  ṁ = P₀/√(RT₀) · A* · √γ · [2/(γ+1)]^((γ+1)/(2(γ-1)))

Dividing:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  A/A* = (1/M) · [(2/(γ+1)) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1))) │
  └──────────────────────────────────────────────────────────────────────┘
```

**Important property:** For any A/A* > 1, there are TWO solutions:
- One subsonic M < 1 (the converging section)
- One supersonic M > 1 (the diverging section)

The nozzle design selects the supersonic branch in the diverging section.

**Worked example — find Mach number at exit:**

```
Given: γ = 1.2, expansion ratio Ae/At = 30

We need to solve: 30 = (1/M)·[(2/2.2)·(1 + 0.1·M²)]^(2.2/0.4)
                  30 = (1/M)·[0.9091·(1 + 0.1M²)]^5.5

This is transcendental — no closed-form solution. Use Newton-Raphson or brentq.
Answer: Me ≈ 4.69

Verify: T/T₀ = [1 + 0.1×4.69²]⁻¹ = 1/3.2 = 0.312 → T = 1092 K
        P/P₀ = (0.312)^6 = 9.2×10⁻⁴ → P = 6440 Pa (near vacuum!)
        Ve = Me × a = Me × √(γRT) = 4.69 × √(1.2×461.5×1092) = 4.69 × 777 = 3644 m/s
```

### 2.6 Thrust and Thrust Coefficient

**Thrust** comes from two sources — momentum and pressure:

```
F = ṁ·Ve + (Pe - Pa)·Ae

  momentum thrust   pressure thrust
  (dominant term)   (matters in vacuum)

In vacuum (Pa = 0):
  F_vac = ṁ·Ve + Pe·Ae
```

The **thrust coefficient Cf** normalizes thrust by throat area and pressure:

```
Cf = F / (Pc · At)

For vacuum:
  Cf_vac = Γ·√(2γ/(γ-1)·[1-(Pe/Pc)^((γ-1)/γ)]) + (Pe/Pc)·(Ae/At)

Typical values: Cf_vac ≈ 1.5–2.0
```

**The connection between Isp and C*:**

```
F = Cf · Pc · At
ṁ = Pc · At / C*         (from definition of C*)

Isp = F/(ṁg₀) = Cf·Pc·At / (Pc·At/C* · g₀) = Cf·C* / g₀

  ┌───────────────────────────┐
  │  Isp = Cf · C* / g₀       │  ← This separates chemistry (C*) from nozzle design (Cf)
  └───────────────────────────┘
```

### 2.7 Choked Flow — Maximum Mass Flow

When M = 1 at the throat, the flow is **choked**. No matter what you do
downstream, the mass flow rate cannot increase:

```
ṁ_max = P₀ · At · Γ / √(R·T₀)

where Γ = √γ · (2/(γ+1))^((γ+1)/(2(γ-1)))
```

**Numerical example:**

```
P₀ = 7 MPa, At = π×0.04² = 5.03×10⁻³ m², T₀ = 3430 K, R = 461.5, γ = 1.2

Γ = √1.2 × (2/2.2)^(2.2/0.4) = 1.095 × 0.9091^5.5 = 0.638

ṁ = 7×10⁶ × 5.03×10⁻³ × 0.638 / √(461.5 × 3430)
  = 22460 / 1258.2
  = 17.85 kg/s

If you increase Pc to 14 MPa: ṁ doubles to 35.7 kg/s (linear in Pc).
If you increase Rt to 0.06 m: ṁ increases by (0.06/0.04)² = 2.25×.
```

### Experiments

**Experiment 2.1 — Visualize isentropic relations**
```python
import numpy as np
import matplotlib.pyplot as plt
from nozzle_analysis import isentropic, area_ratio

M = np.linspace(0.01, 5.0, 500)
gammas = [1.15, 1.2, 1.3, 1.4]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for g in gammas:
    T_r = (1 + (g-1)/2 * M**2)**(-1)
    P_r = T_r**(g/(g-1))
    rho_r = T_r**(1/(g-1))
    AR = area_ratio(M, g)

    axes[0,0].plot(M, T_r,   label=f"γ={g}")
    axes[0,1].plot(M, P_r,   label=f"γ={g}")
    axes[1,0].plot(M, rho_r, label=f"γ={g}")
    axes[1,1].semilogy(M, AR, label=f"γ={g}")

titles = ["T/T₀", "P/P₀", "ρ/ρ₀", "A/A* (log scale)"]
ylabels = ["T/T₀", "P/P₀", "ρ/ρ₀", "A/A*"]
for ax, t, yl in zip(axes.flat, titles, ylabels):
    ax.set(xlabel="Mach number", ylabel=yl, title=t)
    ax.axvline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig("study_2_1.png", dpi=120)
print("Key takeaway: at M=1 (throat), T/T₀ ≈ 0.83, P/P₀ ≈ 0.56 for γ=1.2")
print("Most of the expansion happens at supersonic speeds.")
```

**Experiment 2.2 — Full nozzle analysis and plot interpretation**
```python
from nozzle_analysis import run, save_results

params = {
    "name": "study_2_2", "gamma": 1.2, "R": 461.5,
    "T0": 3500, "P0": 7e6, "Pe": 101325,
    "throat_radius": 0.04, "half_angle_deg": 15.0, "n_points": 300,
}
result = run(params)
save_results(result)
print("Open results/study_2_2/nozzle_plots.png")
print("Things to notice:")
print("  1. Temperature drops gradually in converging, rapidly in diverging section")
print("  2. Mach = 1.0 exactly at the throat")
print("  3. Pressure drops from 7 MPa to ~0.1 MPa (70× reduction!)")
print("  4. Velocity goes from ~0 to ~3000+ m/s")
```

**Experiment 2.3 — Effect of expansion ratio on Isp**
```python
from nozzle_analysis import run
import numpy as np, matplotlib.pyplot as plt

base = dict(name="er_test", gamma=1.2, R=461.5, T0=3500, P0=7e6,
            throat_radius=0.04, half_angle_deg=15.0, n_points=200)

exit_pressures = [1e6, 3e5, 1e5, 3e4, 1e4, 3e3, 1e3, 300, 100]
Ae_Ats = []; Isps = []; Ves = []

for Pe in exit_pressures:
    r = run({**base, "Pe": Pe})
    Ae_Ats.append(r[11])   # expansion ratio
    Isps.append(r[18])     # Isp_vac
    Ves.append(r[10])      # exit velocity

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.semilogx(Ae_Ats, Isps, "o-")
ax1.set(xlabel="Expansion ratio Ae/At", ylabel="Isp_vac [s]",
        title="Isp improves with expansion ratio (diminishing returns)")
ax1.grid(True, alpha=0.3)

ax2.semilogx(Ae_Ats, Ves, "s-", color="steelblue")
ax2.set(xlabel="Expansion ratio Ae/At", ylabel="Ve [m/s]",
        title="Exit velocity vs expansion ratio")
ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig("study_2_3.png", dpi=120)
print("Notice: going from Ae/At=10 to 100 gives much less Isp gain than 1→10")
print("This is because you're extracting the 'last bits' of pressure energy.")
```

**Experiment 2.4 — Hand-verify a calculation (DO THIS ON PAPER FIRST)**
```python
import numpy as np

# Given:
gamma = 1.2; T0 = 3500; P0 = 7e6; R = 461.5
Me = 3.0  # exit Mach number

# Step 1: Temperature at exit
T_ratio = (1 + (gamma-1)/2 * Me**2)**(-1)
Te = T0 * T_ratio
print(f"Step 1: T/T₀ = {T_ratio:.4f} → Te = {Te:.0f} K")

# Step 2: Pressure at exit
P_ratio = T_ratio ** (gamma/(gamma-1))
Pe = P0 * P_ratio
print(f"Step 2: P/P₀ = {P_ratio:.6f} → Pe = {Pe:.0f} Pa ({Pe/1e3:.1f} kPa)")

# Step 3: Speed of sound at exit
ae = np.sqrt(gamma * R * Te)
print(f"Step 3: a_exit = √(γRT) = {ae:.1f} m/s")

# Step 4: Exit velocity
Ve = Me * ae
print(f"Step 4: Ve = M×a = {Ve:.0f} m/s")

# Step 5: Area ratio
AR = (1/Me) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * Me**2)) ** ((gamma+1)/(2*(gamma-1)))
print(f"Step 5: Ae/At = {AR:.2f}")

# Step 6: Throat area and mass flow
At = np.pi * 0.04**2
Gamma_func = np.sqrt(gamma) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
mdot = P0 * At * Gamma_func / np.sqrt(R * T0)
print(f"Step 6: At = {At*1e4:.2f} cm², ṁ = {mdot:.2f} kg/s")

# Step 7: Thrust (vacuum)
Ae = AR * At
F_vac = mdot * Ve + Pe * Ae
Isp_vac = F_vac / (mdot * 9.80665)
print(f"Step 7: F_vac = {F_vac:.0f} N, Isp_vac = {Isp_vac:.1f} s")
```

### Questions to answer

1. At what Mach number does the flow have exactly half the stagnation pressure?
   Solve: P/P₀ = 0.5 for M. Do it analytically for γ = 1.4, then check with the code.
2. If you double the throat area (keeping P₀ and T₀ constant), what happens to thrust? To Isp? Why?
3. Why does a sea-level engine (Merlin) have Ae/At ≈ 16, while a vacuum engine (RL-10) has Ae/At ≈ 80?
   What would happen if you used an Ae/At=80 nozzle at sea level?
4. What happens physically when Pe < P_ambient? (Hint: look up "flow separation" and "shock diamonds")
5. In Experiment 2.3, does Isp_vac keep increasing as Ae/At → ∞? Why does it plateau?
6. **Hand calculation:** For γ=1.3, Me=4.0, T₀=3000 K, P₀=10 MPa, R=300 J/kg·K, Rt=0.03 m:
   calculate Te, Pe, Ve, Ae/At, ṁ, and F_vac. Show all steps.

---

## Module 3 — Rao Nozzle Contour (Week 4)

### 3.1 The Problem with Straight Cones

A 15° half-angle cone is the simplest nozzle design. But it wastes thrust
because gas near the wall exits at an angle θ to the axis:

```
              ╱ gas velocity near wall
             ╱  ↗ (angle θ to axis)
            ╱  ╱
           ╱  ╱
  throat  ╱  ╱
  ========╱  ╱
  ========╲  ╲
           ╲  ╲
            ╲  ╲
             ╲  ╲
              ╲

Only the AXIAL component (V·cos θ) produces thrust.
The RADIAL component (V·sin θ) is wasted.
```

**Divergence loss factor λ:**

```
λ = (1 + cos θ) / 2

For a 15° cone:  λ = (1 + cos 15°) / 2 = (1 + 0.9659) / 2 = 0.983  → 1.7% thrust loss
For a 30° cone:  λ = (1 + cos 30°) / 2 = (1 + 0.8660) / 2 = 0.933  → 6.7% thrust loss
```

**Where does λ = (1 + cos θ)/2 come from?** Integration over the exit plane:

```
Consider a circular exit of radius Re. At each radius r, the gas exits at
angle α(r) = θ·r/Re (linear from 0 at center to θ at wall for a cone).

The axial momentum flux at radius r:
  dF = ṁ(r) · Ve · cos α(r)

Integrating over the exit plane:
  F_actual = ∫₀ᴿᵉ ṁ(r) · Ve · cos(α(r)) · 2πr dr

Divide by F_ideal (all flow axial):
  λ = F_actual / F_ideal = (1 + cos θ) / 2

(This assumes uniform mass flux per unit area — a simplification.)
```

### 3.2 The Rao Thrust-Optimized Contour

G.V.R. Rao (1958) solved this problem: find the nozzle wall shape that
maximizes thrust for a given length. The solution uses the **Method of
Characteristics** (MoC) — tracing supersonic wave fronts.

**The key insight:** In supersonic flow, information travels along
**characteristic lines** (Mach waves) at angle μ = arcsin(1/M) to the flow:

```
Characteristic angles:

  At M=1:   μ = arcsin(1/1) = 90°   (waves perpendicular to flow)
  At M=2:   μ = arcsin(1/2) = 30°
  At M=5:   μ = arcsin(1/5) = 11.5°

The nozzle wall is a "characteristic" itself — a boundary condition.
By shaping the wall correctly, we can control how the waves reflect
and ensure all gas exits parallel to the axis.
```

### 3.3 The Rao Contour Geometry

The Rao contour has three sections:

```
  Section 1: Circular arc (near throat)
  ┌─────────────────────────────────────────────────────
  │  R_arc = 0.382 × Rt    (upstream radius of curvature)
  │  Starts at throat, extends to inflection angle θᵢ
  │  Provides smooth transition from M≈1 to initial expansion
  └─────────────────────────────────────────────────────

  Section 2: Hermite cubic (divergent section)
  ┌─────────────────────────────────────────────────────
  │  Connects inflection point (θᵢ) to exit (θₑ)
  │  Boundary conditions:
  │    - Start point: r = r_inflection, slope = tan(θᵢ)
  │    - End point:   r = Re,           slope = tan(θₑ)
  │  Gives C¹ continuity (smooth tangent) everywhere
  └─────────────────────────────────────────────────────

  Section 3: (implicit) — subsonic converging section
  ┌─────────────────────────────────────────────────────
  │  Typically a simple circular arc or spline
  │  Less critical because subsonic flow adapts to shape
  └─────────────────────────────────────────────────────
```

**Why 0.382 × Rt for the throat radius?**

The number 0.382 ≈ (√5 − 1)/2 − 0.236 comes from empirical optimization by Rao
and confirmed by many later studies. A smaller R_arc gives faster expansion
(shorter nozzle) but higher boundary layer losses. 0.382·Rt is the sweet spot.

### 3.4 Hermite Cubic Interpolation — Explained

The divergent section uses Hermite cubic interpolation. This is a way to draw
a smooth curve between two points when you know the slope at each end.

```
Given:  Point A at (x₀, r₀) with slope m₀ = tan(θᵢ)
        Point B at (x₁, r₁) with slope m₁ = tan(θₑ)

Parameter t goes from 0 (at A) to 1 (at B).

The four Hermite basis functions:
  h₀₀(t) = 2t³ - 3t² + 1     (value at A = 1, value at B = 0)
  h₁₀(t) = t³ - 2t² + t      (slope at A = 1, slope at B = 0)
  h₀₁(t) = -2t³ + 3t²        (value at A = 0, value at B = 1)
  h₁₁(t) = t³ - t²            (slope at A = 0, slope at B = 1)

The curve:
  x(t) = h₀₀·x₀ + h₁₀·Δx·1 + h₀₁·x₁ + h₁₁·Δx·1    (linear in x)
  r(t) = h₀₀·r₀ + h₁₀·Δx·m₀ + h₀₁·r₁ + h₁₁·Δx·m₁

This guarantees:
  r(0) = r₀,  r'(0) = m₀    (matches inflection point)
  r(1) = r₁,  r'(1) = m₁    (matches exit conditions)
  Smooth curve between (no corners or kinks)
```

**Analogy:** Imagine driving a car from A to B. Hermite interpolation tells
you the steering wheel angle at every point, given that you start heading
northeast (angle θᵢ) and finish heading almost east (angle θₑ).

### 3.5 Divergence Loss in the Rao Contour

For the Rao contour, the effective divergence loss factor depends on the
velocity profile at the exit plane. Our code computes it by integrating
over the wall contour using MoC:

```
λ_rao > λ_cone (for same length)

Typical results:
  15° cone, L/Rt = 10:    λ = 0.983   (1.7% loss)
  Rao contour, L/Rt = 10: λ = 0.992   (0.8% loss)

The Rao contour recovers about half the divergence loss of a cone at the same length.
```

### 3.6 The Length-Isp Trade-off

There's a fundamental trade-off between nozzle length and performance:

```
Longer nozzle → larger exit area → more expansion → higher Isp
BUT:
  - More mass (structural weight)
  - More friction loss (boundary layer)
  - More cooling surface area needed
  - Physical space constraints
```

**L_factor** (used in moc.py) estimates the length:

```
L_nozzle ≈ L_factor × (Re - Rt)

where L_factor = 0.7 + 0.2 × (θᵢ / 30°)

Larger θᵢ → faster initial expansion → shorter nozzle for same Ae/At
But: too fast expansion → shock waves, boundary layer separation
```

### Experiments

**Experiment 3.1 — Visualize contour shape vs parameters**
```python
import numpy as np
import matplotlib.pyplot as plt
from moc import rao_contour
from nozzle_analysis import area_ratio
from scipy.optimize import brentq

Rt = 0.04; g = 1.2
Me = brentq(lambda M: area_ratio(M, g) - 40.0, 3.0, 15.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vary exit angle θₑ
for te in [2, 5, 10, 15]:
    w = rao_contour(g, Rt, Me, theta_i_deg=30, theta_e_deg=te, n_lines=12)
    axes[0].plot(w[:,0]*100, w[:,1]*100, label=f"θₑ={te}°")
axes[0].set(xlabel="x [cm]", ylabel="r [cm]", title="Effect of exit angle θₑ (θᵢ=30° fixed)")
axes[0].legend(); axes[0].set_aspect("equal")

# Vary inflection angle θᵢ
for ti in [15, 25, 35, 45]:
    w = rao_contour(g, Rt, Me, theta_i_deg=ti, theta_e_deg=5, n_lines=12)
    axes[1].plot(w[:,0]*100, w[:,1]*100, label=f"θᵢ={ti}°")
axes[1].set(xlabel="x [cm]", ylabel="r [cm]", title="Effect of inflection angle θᵢ (θₑ=5° fixed)")
axes[1].legend(); axes[1].set_aspect("equal")

plt.suptitle(f"Rao contour for Ae/At = 40, Rt = {Rt*100:.0f} cm, Me = {Me:.2f}", fontsize=12)
plt.tight_layout(); plt.savefig("study_3_1.png", dpi=120)

# Print observations
print("Left plot: smaller θₑ → straighter exit → less divergence loss → more thrust")
print("Left plot: but θₑ too small → nozzle becomes very long")
print("Right plot: larger θᵢ → wider initial expansion → shorter nozzle")
print("Right plot: but θᵢ too large → risk of flow separation near throat")
```

**Experiment 3.2 — Divergence loss vs exit angle**
```python
import numpy as np
import matplotlib.pyplot as plt

# Simple divergence loss: λ = (1 + cos θₑ)/2
theta_e = np.linspace(0.5, 25, 50)
lam_cone = 0.5 * (1 + np.cos(np.radians(theta_e)))
thrust_loss_pct = (1 - lam_cone) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(theta_e, lam_cone)
ax1.axhline(0.983, color="r", ls="--", alpha=0.5, label="15° cone (λ=0.983)")
ax1.set(xlabel="Exit angle θₑ [deg]", ylabel="λ", title="Divergence loss factor")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(theta_e, thrust_loss_pct)
ax2.axhline(1.7, color="r", ls="--", alpha=0.5, label="15° cone (1.7%)")
ax2.set(xlabel="Exit angle θₑ [deg]", ylabel="Thrust loss [%]",
        title="Thrust lost due to non-axial exit flow")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig("study_3_2.png", dpi=120)
print("At θₑ=5°: loss = 0.19%. At θₑ=15°: loss = 1.7%. At θₑ=25°: loss = 4.7%")
print("Every degree matters at high performance!")
```

**Experiment 3.3 — Cone vs Rao: visual comparison**
```python
import numpy as np
import matplotlib.pyplot as plt
from moc import rao_contour
from nozzle_analysis import area_ratio
from scipy.optimize import brentq

Rt = 0.04; g = 1.2
Me = brentq(lambda M: area_ratio(M, g) - 30.0, 3.0, 15.0)
Re = Rt * np.sqrt(30.0)  # exit radius for Ae/At = 30

# Rao contour
w = rao_contour(g, Rt, Me, theta_i_deg=30, theta_e_deg=5, n_lines=12)

# 15° cone with same exit radius
L_cone = (Re - Rt) / np.tan(np.radians(15))
x_cone = np.linspace(0, L_cone, 100)
r_cone = Rt + x_cone * np.tan(np.radians(15))

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(w[:,0]*100, w[:,1]*100, "b-", lw=2, label="Rao (θᵢ=30°, θₑ=5°)")
ax.plot(x_cone*100, r_cone*100, "r--", lw=2, label="15° cone")
ax.axhline(Rt*100, color="k", ls=":", alpha=0.5)
ax.set(xlabel="x [cm]", ylabel="r [cm]")
ax.set_aspect("equal")
ax.legend(fontsize=11)
ax.set_title("Rao contour vs 15° cone — same expansion ratio")
plt.tight_layout(); plt.savefig("study_3_3.png", dpi=120)

print(f"Cone length: {L_cone*100:.1f} cm")
print(f"Rao length:  {w[-1,0]*100:.1f} cm")
print(f"Rao is {(1 - w[-1,0]/L_cone)*100:.0f}% shorter OR has less divergence loss at same length")
```

**Experiment 3.4 — Manually try to beat the optimizer**
```python
from engine_system import nozzle_perf

eng = {"gamma": 1.2, "R": 461.5, "T0": 3500, "pc": 7e6, "Cstar": 2300}

# Try different combinations — can you find the best Isp?
attempts = [
    (25, 3,  "conservative"),
    (30, 5,  "moderate"),
    (35, 8,  "aggressive"),
    (40, 10, "very aggressive"),
    (30, 2,  "small exit angle"),
    (20, 5,  "small inflection"),
]

print(f"{'θᵢ':>4s}  {'θₑ':>4s}  {'Isp_vac':>8s}  {'Length':>7s}  {'λ':>6s}  {'Label'}")
print("-" * 55)
for ti, te, label in attempts:
    noz = nozzle_perf(eng, Rt=0.04, Pe=101325, theta_i=ti, theta_e=te)
    print(f"{ti:4.0f}  {te:4.0f}  {noz['Isp_vac']:8.2f}  {noz['length']:7.3f}  "
          f"{noz['divergence_lambda']:6.4f}  {label}")

print("\nNow run the full optimizer and compare with your best guess!")
```

### Questions to answer

1. From Experiment 3.2: what θₑ gives λ = 0.999? How long would that nozzle be?
2. Why does making θₑ very small (→0°) not give infinite Isp? (Hint: what limits the nozzle?)
3. Plot nozzle length vs Isp for θᵢ = 30° and θₑ varying from 2° to 20°. Where is the "knee" of the curve?
4. If you increase γ from 1.2 to 1.4, how does the optimal contour change? (Generate contours for both.)
5. Look at the `rao_contour()` function in moc.py. Why is `np.clip(r_wall, Rt, None)` needed?
   What would happen without it?
6. **Hand-draw** a Rao contour for Rt = 4 cm, Re = 20 cm, θᵢ = 30°, θₑ = 5°. Use the
   Hermite formulas from §3.4. Plot at least 5 points on paper, then compare with the code.

---

## Module 4 — Numerical Optimization (Week 5)

### 4.1 The Optimization Problem

We want to find the nozzle shape that maximizes Isp. Formally:

```
MAXIMIZE   Isp_vac(θᵢ, θₑ, n_lines)

Subject to:
  15° ≤ θᵢ ≤ 50°         (inflection angle bounds)
  1°  ≤ θₑ ≤ 15°         (exit angle bounds)
  5   ≤ n_lines ≤ 20      (MoC resolution, integer)
  L(θᵢ, θₑ) ≤ L_max      (length constraint)
```

This is a 3-variable, constrained, mixed-integer (n_lines) optimization problem.

### 4.2 Why Not Gradient Descent?

Gradient descent (and Newton's method) require ∂f/∂x at every point. Our objective has problems:

**Problem 1: No analytical gradient.** Each Isp evaluation requires:
- Solving area-Mach equation (root finding)
- Generating MoC contour (numerical integration)
- Computing thrust integral (numerical quadrature)
- No closed-form Isp(θᵢ, θₑ) exists

**Problem 2: Integer variable.** n_lines is an integer — gradients are undefined for integers.

**Problem 3: Potential local optima.** The Isp landscape may have multiple peaks.

**Analogy:** Imagine searching for the highest hill in a foggy mountain range.
Gradient descent = walking uphill from where you stand (you might find a small hill).
Differential Evolution = sending 20 hikers to random starting points, then
sharing information about who found the best spots (you'll likely find the highest peak).

### 4.3 Differential Evolution (DE) — Step by Step

DE is an evolutionary algorithm that maintains a **population** of candidate
solutions and iteratively improves them.

```
ALGORITHM: Differential Evolution

Input: population size N, mutation factor F ∈ [0.5, 1.0], crossover rate CR ∈ [0.5, 1.0]

1. INITIALIZE: Generate N random vectors x₁, x₂, ..., xₙ within bounds
   Example (N=5):
     x₁ = [θᵢ=22°, θₑ=8°,  n=8 ]  → Isp₁ = 412.3
     x₂ = [θᵢ=38°, θₑ=12°, n=15]  → Isp₂ = 405.7
     x₃ = [θᵢ=30°, θₑ=5°,  n=10]  → Isp₃ = 418.1   ← current best
     x₄ = [θᵢ=45°, θₑ=3°,  n=7 ]  → Isp₄ = 414.5
     x₅ = [θᵢ=17°, θₑ=11°, n=12]  → Isp₅ = 398.2

2. For each vector xᵢ in the population:
   a. Pick 3 random OTHER members: xₐ, xᵦ, xᵧ (all different, ≠ i)
   b. Create MUTANT: v = xₐ + F × (xᵦ - xᵧ)
      Example: v = x₃ + 0.8 × (x₁ - x₄)
             = [30, 5, 10] + 0.8 × ([22, 8, 8] - [45, 3, 7])
             = [30, 5, 10] + 0.8 × [-23, 5, 1]
             = [30-18.4, 5+4.0, 10+0.8]
             = [11.6, 9.0, 10.8]
             (then clip to bounds: [15, 9.0, 11])

   c. CROSSOVER: create trial u by mixing xᵢ and v:
      For each dimension j:
        u[j] = v[j]  if  rand() < CR  (take mutant component)
        u[j] = xᵢ[j] otherwise        (keep original component)
      (At least one component must come from v)

   d. SELECTION: evaluate f(u)
      If f(u) > f(xᵢ):  replace xᵢ with u   (better → keep)
      Else:              keep xᵢ              (worse → discard trial)

3. Repeat step 2 until convergence or n_eval reached.
```

**Why F × (xᵦ - xᵧ)?** The difference vector (xᵦ - xᵧ) captures the "scale"
of the population spread. Early on, when vectors are far apart, mutations are large
(exploration). Later, as the population converges, mutations become small (exploitation).

### 4.4 Nelder-Mead Simplex — The Local Polisher

After DE finds a good region, Nelder-Mead refines the solution. It works with
n+1 = 4 points (a "simplex" in 3D) and uses geometric operations:

```
Given a simplex with vertices sorted by objective value:
  Best = xᵇ,  Second-worst = xˢ,  Worst = xʷ

1. CENTROID: c = mean of all points except worst
2. REFLECT:  xᵣ = c + α(c - xʷ)     (α=1: mirror worst through centroid)
3. If xᵣ is better than best:
     EXPAND:  xₑ = c + β(xᵣ - c)    (β=2: go even further)
     Keep xₑ if better, else keep xᵣ
4. If xᵣ is worse than second-worst:
     CONTRACT: xc = c + ρ(xʷ - c)   (ρ=0.5: halfway to worst)
     Keep xc if better than worst
5. If nothing works:
     SHRINK: move all points toward best  (last resort)

Repeat until the simplex is tiny (converged).
```

**Analogy:** Imagine three friends searching for the highest point on a hill.
They form a triangle. In each step, the person at the lowest point leapfrogs
over the midpoint of the other two. The triangle gradually climbs to the top
and shrinks as they all converge on the peak.

### 4.5 Handling the Length Constraint

The length constraint L ≤ L_max is enforced as a **penalty**:

```
objective(θᵢ, θₑ, n) = Isp_vac(θᵢ, θₑ, n) - penalty

where penalty = {  0                           if L ≤ L_max
                {  big_number × (L - L_max)    if L > L_max
```

This effectively creates a "cliff" in the objective landscape: solutions
that violate the constraint have terrible objective values and are discarded
by DE's selection step.

### Experiments

**Experiment 4.1 — Visualize the objective landscape**
```python
import numpy as np
import matplotlib.pyplot as plt
from engine_system import nozzle_perf

eng = {"gamma": 1.2, "R": 461.5, "T0": 3500, "pc": 7e6, "Cstar": 2300}
Rt = 0.04; Pe = 101325

theta_i_range = np.linspace(15, 50, 35)
theta_e_range = np.linspace(1, 15, 35)
Isp_grid = np.zeros((35, 35))

for i, ti in enumerate(theta_i_range):
    for j, te in enumerate(theta_e_range):
        try:
            noz = nozzle_perf(eng, Rt, Pe, theta_i=ti, theta_e=te, n_lines=10)
            Isp_grid[i, j] = noz["Isp_vac"] if noz["length"] < 0.6 else 0
        except Exception:
            Isp_grid[i, j] = 0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Contour plot
cp = axes[0].contourf(theta_e_range, theta_i_range, Isp_grid, levels=25, cmap="viridis")
plt.colorbar(cp, ax=axes[0], label="Isp_vac [s]")
axes[0].set(xlabel="θₑ [deg]", ylabel="θᵢ [deg]", title="Isp_vac(θᵢ, θₑ) — contour")

# 3D surface
from mpl_toolkits.mplot3d import Axes3D
ax3d = fig.add_subplot(122, projection="3d")
Te, Ti = np.meshgrid(theta_e_range, theta_i_range)
ax3d.plot_surface(Te, Ti, Isp_grid, cmap="viridis", alpha=0.8)
ax3d.set(xlabel="θₑ", ylabel="θᵢ", zlabel="Isp_vac")
ax3d.set_title("3D landscape")

plt.tight_layout(); plt.savefig("study_4_1.png", dpi=120)
print("Look for: is there one clear peak? Multiple peaks? A ridge?")
print("This tells you whether DE is needed (multiple peaks) or gradient descent would suffice (one peak)")
```

**Experiment 4.2 — Watch DE convergence history**
```python
# First, run the optimizer
import subprocess
subprocess.run(["python", "engine_system.py",
    "--mode", "optimize",
    "--fuel", "H2", "--oxidizer", "O2",
    "--throat_radius", "0.04",
    "--p_tank_f", "8e6", "--p_tank_o", "8e6",
    "--A_f", "2e-4", "--A_o", "2e-4",
    "--Pe", "101325", "--n_eval", "100",
    "--name", "study_4_2"])

# Then plot convergence
import pandas as pd
import matplotlib.pyplot as plt
import glob, os

csv_files = glob.glob("results/study_4_2*/optimization_history.csv")
if csv_files:
    df = pd.read_csv(csv_files[0])
    valid = df[df["Isp_vac"] > 0]
    best_so_far = valid["Isp_vac"].cummax()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.scatter(valid.index, valid["Isp_vac"], alpha=0.3, s=5, c="steelblue")
    ax1.plot(best_so_far.index, best_so_far, "r-", lw=2, label="Best so far")
    ax1.set(xlabel="Evaluation #", ylabel="Isp_vac [s]", title="DE convergence")
    ax1.legend()

    # Parameter evolution
    ax2.scatter(valid["theta_i"], valid["theta_e"], c=valid.index, cmap="plasma",
                s=10, alpha=0.5)
    ax2.set(xlabel="θᵢ [deg]", ylabel="θₑ [deg]", title="Parameter space explored")
    plt.colorbar(ax2.collections[0], ax=ax2, label="Evaluation #")

    plt.tight_layout(); plt.savefig("study_4_2.png", dpi=120)
    print(f"Best Isp found: {best_so_far.iloc[-1]:.2f} s after {len(valid)} evaluations")
```

**Experiment 4.3 — Effect of population size (n_eval)**
```python
# Run optimizer with different n_eval and compare results
import subprocess

results = {}
for n_eval in [20, 50, 100, 200]:
    r = subprocess.run(["python", "engine_system.py",
        "--mode", "optimize",
        "--fuel", "H2", "--oxidizer", "O2",
        "--throat_radius", "0.04",
        "--p_tank_f", "8e6", "--p_tank_o", "8e6",
        "--A_f", "2e-4", "--A_o", "2e-4",
        "--Pe", "101325", f"--n_eval", str(n_eval),
        "--name", f"study_4_3_n{n_eval}"],
        capture_output=True, text=True)
    # Parse output for best Isp...
    output = r.stdout
    print(f"n_eval={n_eval}: {output[-200:]}")

print("\nAt what n_eval does the improvement plateau?")
print("This is the 'exploration vs computation time' trade-off.")
```

**Experiment 4.4 — Implement your own mini-optimizer**
```python
# Build a simple random search to understand why DE is better
import numpy as np
from engine_system import nozzle_perf

eng = {"gamma": 1.2, "R": 461.5, "T0": 3500, "pc": 7e6, "Cstar": 2300}

# Random search
best_isp = 0
best_params = None
history = []

for i in range(200):
    ti = np.random.uniform(15, 50)
    te = np.random.uniform(1, 15)
    try:
        noz = nozzle_perf(eng, 0.04, 101325, theta_i=ti, theta_e=te)
        isp = noz["Isp_vac"]
        history.append((ti, te, isp))
        if isp > best_isp:
            best_isp = isp
            best_params = (ti, te)
    except:
        pass

print(f"Random search best: Isp = {best_isp:.2f} s at θᵢ={best_params[0]:.1f}°, θₑ={best_params[1]:.1f}°")
print("Compare with DE result from Experiment 4.2. How much worse is random search?")
print(f"Random search explored {len(history)} points but didn't use any 'intelligence'.")
```

### Questions to answer

1. From Experiment 4.1: is the landscape smooth? Are there multiple local maxima?
   Would gradient descent work here?
2. Why does DE use mutation F·(xᵦ - xᵧ)? What happens if F = 0 (no mutation)?
   What if F = 2 (huge mutation)?
3. In the code, why does Nelder-Mead start from DE's best result rather than a random point?
4. How many function evaluations does your Experiment 4.4 random search need to match DE's
   result from Experiment 4.2? What does this tell you about DE's efficiency?
5. **Design exercise:** If you added a 4th optimization variable (e.g., throat radius Rt),
   how would the problem change? Would DE still work? What about gradient-based methods?

---

## Module 5 — Heat Transfer & Cooling (Week 6–7)

### 5.1 Why Cooling Matters: A Survival Problem

A rocket combustion chamber reaches 3000–3600 K. Common structural metals:
- Copper (CuCrZr): melts at 1070°C (1343 K)
- Stainless steel: melts at 1400°C (1673 K)
- Nickel alloys (Inconel 718): melts at 1336°C (1609 K)

The chamber gas is **2–3× hotter than the melting point of the wall**. Without
active cooling, the engine would melt in less than a second.

**Heat flux at the throat:** typically 10–50 MW/m² — for comparison:
- Sun's surface: 63 MW/m²
- Kitchen stove burner: ~0.01 MW/m²
- A rocket throat can have heat flux comparable to the SUN.

### 5.2 Heat Transfer Modes — Quick Review

Three modes of heat transfer:

```
1. CONDUCTION: heat flows through a solid material
   q = k × (T_hot - T_cold) / thickness     [W/m²]
   k = thermal conductivity [W/m·K]

   Example: copper k ≈ 380 W/m·K (excellent conductor)
            steel  k ≈  16 W/m·K (poor conductor)

2. CONVECTION: heat transfer between a surface and a flowing fluid
   q = h × (T_fluid - T_surface)             [W/m²]
   h = heat transfer coefficient [W/m²·K]

   Example: forced air  h ≈ 50 W/m²·K
            water flow   h ≈ 5000 W/m²·K
            rocket gas   h ≈ 5000–50,000 W/m²·K

3. RADIATION: electromagnetic waves (less important in rockets due to
   dominant convection — we neglect it in our code)
```

### 5.3 The Three-Layer Thermal Resistance Model

Imagine the wall of a rocket engine as a sandwich with three layers of
"thermal resistance" in series:

```
  HOT GAS (T₀ ≈ 3500 K)
  ─────────────────────────  ← gas boundary layer (resistance 1/h_g)
  ████████████████████████   ← wall surface, gas side (T_wg — the critical temp)
  ████████ WALL ██████████   ← wall material (resistance t_wall/k_wall)
  ████████████████████████   ← wall surface, coolant side (T_wc)
  ─────────────────────────  ← coolant boundary layer (resistance 1/h_c)
  COLD COOLANT (T_cool ≈ 100 K)
```

This is exactly like electrical resistance in series: V = I × R_total

```
Thermal analogy:
  ΔT = q × R_total           (like V = I × R)
  R_total = 1/h_g + t_wall/k_wall + 1/h_c

Therefore:
  ┌───────────────────────────────────────────────────────────┐
  │  q = (T_aw - T_cool) / (1/h_g + t_wall/k_wall + 1/h_c)  │
  └───────────────────────────────────────────────────────────┘

And the wall temperatures:
  T_wg = T_aw - q/h_g          (gas-side wall temperature)
  T_wc = T_cool + q/h_c        (coolant-side wall temperature)
```

**Numerical example — throat conditions:**

```
Given:  T_aw = 3200 K,  T_cool = 200 K
        h_g = 15,000 W/m²K,  h_c = 30,000 W/m²K
        wall: CuCrZr, k = 320 W/mK, t = 1.5 mm

R_total = 1/15000 + 0.0015/320 + 1/30000
        = 6.67×10⁻⁵ + 4.69×10⁻⁶ + 3.33×10⁻⁵
        = 1.047×10⁻⁴ K·m²/W

q = (3200 - 200) / 1.047×10⁻⁴ = 28.7 MW/m²  ← intense!

T_wg = 3200 - 28.7×10⁶/15000 = 3200 - 1913 = 1287 K  ← close to CuCrZr limit (1340 K)!
T_wc = 200 + 28.7×10⁶/30000  = 200 + 957 = 1157 K

Observation: the wall barely survives. A 10% increase in h_g (hotter gas) or
10% decrease in h_c (worse cooling) and the wall melts.
```

### 5.4 Adiabatic Wall Temperature — Where T_aw Comes From

T_aw is NOT the stagnation temperature T₀. It's the temperature the wall would
reach if it were perfectly insulated (adiabatic). The difference arises because
the boundary layer doesn't fully recover the kinetic energy:

```
T_aw = T_local × [1 + r × (γ-1)/2 × M²]

where:
  T_local = T₀ / [1 + (γ-1)/2 × M²]     (local static temperature)
  r = recovery factor ≈ Pr^(1/3) ≈ 0.89  (for turbulent flow, Pr ≈ 0.7)

Simplifying:
  T_aw = T₀ × [1 + r·(γ-1)/2·M²] / [1 + (γ-1)/2·M²]

At the throat (M=1, γ=1.2):
  T_local = T₀ / 1.1 = 0.909·T₀
  T_aw = 0.909·T₀ × (1 + 0.89×0.1) / 1 = 0.909·T₀ × 1.089 = 0.990·T₀

  For T₀ = 3500 K: T_aw = 3465 K  (very close to T₀ — the throat is the worst spot!)

At the exit (M=4, γ=1.2):
  T_local = T₀ / (1 + 0.1×16) = T₀/2.6 = 0.385·T₀
  T_aw = 0.385·T₀ × (1 + 0.89×0.1×16) = 0.385·T₀ × 2.424 = 0.933·T₀

  For T₀ = 3500 K: T_aw = 3266 K  (still very hot at exit due to high Mach!)
```

### 5.5 The Bartz Correlation — Gas-Side Heat Transfer

The Bartz (1957) equation estimates h_g from engine parameters:

```
h_g = (0.026 / Dt^0.2) × (μ^0.2 × cₚ / Pr^0.6) × (Pc/C*)^0.8 
      × (Dt/R_curv)^0.1 × (At/A)^0.9 × σ

Where:
  Dt = throat diameter = 2·Rt
  μ = gas viscosity [Pa·s]
  cₚ = gas specific heat [J/kgK]
  Pr = Prandtl number = μ·cₚ/k_gas
  Pc = chamber pressure
  C* = characteristic velocity
  R_curv = throat radius of curvature
  At/A = throat-to-local area ratio (= 1 at throat)
  σ = correction factor for wall temperature effect
```

**Key insight:** h_g ∝ (At/A)^0.9. At the throat, At/A = 1 (maximum).
At the exit where A/At = 30, h_g drops by a factor of 30^0.9 ≈ 24.
**The throat gets 24× more heat than the nozzle exit.**

**Derivation sketch** (why each term appears):

```
Start from the general Nusselt number correlation for turbulent pipe flow:
  Nu = 0.023 × Re^0.8 × Pr^0.4     (Dittus-Boelter)

Where:
  Nu = h·D/k       (Nusselt number: ratio of convective to conductive heat transfer)
  Re = ρVD/μ       (Reynolds number: ratio of inertial to viscous forces)
  Pr = μcₚ/k       (Prandtl number: ratio of momentum to thermal diffusivity)

Bartz adapted this for a convergent-divergent nozzle by:
  1. Using throat diameter Dt as the characteristic length
  2. Replacing ρV with ṁ/A = (Pc/C*)·(At/A) from the choked flow equation
  3. Adding curvature correction (Dt/R_curv)^0.1
  4. Adding the σ factor for property variation across the boundary layer
```

### 5.6 Dittus-Boelter — Coolant-Side Heat Transfer

The coolant flows through narrow channels machined into the wall. The HTC:

```
Nu = 0.023 × Re^0.8 × Pr^0.4     (Dittus-Boelter, heating)

h_c = Nu × k_cool / D_h

Where:
  D_h = hydraulic diameter = 4 × A_channel / P_wetted
      = 4 × w × h / (2w + 2h) = 2wh/(w+h)  for rectangular channel

  Re = ρ_cool × V_cool × D_h / μ_cool
  Pr = μ_cool × cₚ_cool / k_cool
```

**Design trade-off for channel geometry:**

```
More channels (larger n_ch):
  ✓ More surface area → better heat transfer
  ✗ Each channel is narrower → harder to manufacture
  ✗ Less structural support between channels

Deeper channels (larger height h):
  ✓ Larger cross-section → lower velocity → lower ΔP
  ✗ Can weaken the wall structurally

Wider channels (larger width w):
  ✓ Lower velocity → lower ΔP
  ✗ Fewer channels fit around circumference
  ✗ Wider unsupported span → higher thermal stress
```

### 5.7 The Cooling Loop — Marching Solution

The code in `cooling.py` solves the problem by **marching from nozzle exit to
injector** (coolant flows counter-current to gas):

```
For each axial station i (from exit to throat to chamber):
  1. Compute local M, A, T_aw from isentropic relations
  2. Guess T_wg (initial estimate)
  3. Compute h_g (Bartz) using T_wg for σ correction
  4. Compute h_c (Dittus-Boelter) using local coolant properties
  5. Solve three-layer model → q, T_wg_new, T_wc
  6. If T_wg_new ≠ T_wg → iterate (fixed-point) until convergence
  7. Update coolant temperature: T_cool_next = T_cool + q × ΔA / (ṁ_cool × cₚ_cool)
  8. Update coolant pressure: P_cool_next = P_cool - ΔP_friction
  9. Advance to next station
```

### Experiments

**Experiment 5.1 — Run the full cooling loop**
```python
from injector import Injector
from engine_solver import EngineSolver
from engine_system import nozzle_perf
from moc import rao_contour
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
x = result["x"] * 100  # convert to cm

# Temperature profiles
axes[0,0].plot(x, result["T_wg"], "r-", lw=2, label="Gas-side wall T_wg")
axes[0,0].plot(x, result["T_wc"], "b-", lw=2, label="Coolant-side wall T_wc")
axes[0,0].plot(x, result["T_cool"], "g-", lw=2, label="Coolant bulk T_cool")
axes[0,0].axhline(1350, color="k", ls="--", label="CuCrZr melt limit")
axes[0,0].set(xlabel="x [cm]", ylabel="Temperature [K]", title="Temperature profiles along nozzle")
axes[0,0].legend(fontsize=8)

# Heat flux
axes[0,1].plot(x, result["q"]/1e6, "m-", lw=2)
axes[0,1].set(xlabel="x [cm]", ylabel="q [MW/m²]", title="Heat flux (peak = throat)")
axes[0,1].axvline(0, color="k", ls=":", alpha=0.5, label="Throat")

# HTCs
axes[1,0].plot(x, result["h_g"]/1e3, "r-", lw=2, label="Gas-side h_g")
axes[1,0].plot(x, result["h_c"]/1e3, "b-", lw=2, label="Coolant-side h_c")
axes[1,0].set(xlabel="x [cm]", ylabel="h [kW/m²K]", title="Heat transfer coefficients")
axes[1,0].legend()

# Structural margin
axes[1,1].plot(x, result["margin"], "k-", lw=2)
axes[1,1].axhline(1.0, color="r", ls="--", label="Failure threshold")
axes[1,1].axhline(1.5, color="orange", ls="--", label="Design target (1.5×)")
axes[1,1].set(xlabel="x [cm]", ylabel="UTS/σ_applied", title="Structural margin")
axes[1,1].legend()

for ax in axes.flat: ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("study_5_1.png", dpi=120)

print(f"Max wall temp:   {result['max_T_wg']:.0f} K  (limit: 1340 K for CuCrZr)")
print(f"Total heat load: {result['total_heat']/1e6:.1f} MW")
print(f"Coolant ΔP:      {result['dP_total']/1e6:.2f} MPa")
print(f"Coolant exit T:  {result['T_cool_exit']:.0f} K")
print(f"Min margin:      {min(result['margin']):.2f}×")
```

**Experiment 5.2 — Channel geometry trade-off study**
```python
from cooling import CoolingChannelGeometry, CoolantProperties, solve_cooling_loop
import matplotlib.pyplot as plt

# Sweep number of channels
n_channels_range = [30, 50, 70, 90, 110, 140, 180]
max_Twg = []; dP_cool = []; total_Q = []

for n_ch in n_channels_range:
    geom = CoolingChannelGeometry(n_channels=n_ch, width=1.5e-3, height=3e-3,
                                   wall_thickness=1e-3)
    r = solve_cooling_loop(eng, noz, wall, geom, CoolantProperties("H2"),
                           "CuCrZr", mdot_coolant=eng["mdot_f"], n_stations=60)
    max_Twg.append(r["max_T_wg"])
    dP_cool.append(r["dP_total"]/1e6)
    total_Q.append(r["total_heat"]/1e6)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(n_channels_range, max_Twg, "ro-")
axes[0].axhline(1350, color="k", ls="--", label="CuCrZr melt limit")
axes[0].set(xlabel="Number of channels", ylabel="Max T_wg [K]",
            title="Wall temperature vs channel count")
axes[0].legend()

axes[1].plot(n_channels_range, dP_cool, "bs-")
axes[1].set(xlabel="Number of channels", ylabel="ΔP_coolant [MPa]",
            title="Coolant pressure drop")

axes[2].plot(n_channels_range, total_Q, "gD-")
axes[2].set(xlabel="Number of channels", ylabel="Total heat [MW]",
            title="Total heat absorbed by coolant")

for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("study_5_2.png", dpi=120)

print("Design challenge: find the MINIMUM n_channels where max T_wg < 1200 K")
print("That gives you maximum margin AND minimum pressure drop.")
```

**Experiment 5.3 — Material comparison**
```python
from cooling import CoolingChannelGeometry, CoolantProperties, solve_cooling_loop
import matplotlib.pyplot as plt

materials = ["CuCrZr", "OFHC-Cu", "SS304", "Inconel718"]
geom = CoolingChannelGeometry(n_channels=80, width=1.5e-3, height=3e-3,
                               wall_thickness=1e-3)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for mat in materials:
    r = solve_cooling_loop(eng, noz, wall, geom, CoolantProperties("H2"),
                           mat, mdot_coolant=eng["mdot_f"], n_stations=80)
    x = r["x"] * 100
    axes[0].plot(x, r["T_wg"], label=f"{mat} (max={r['max_T_wg']:.0f} K)")
    axes[1].plot(x, r["q"]/1e6, label=mat)

axes[0].axhline(1350, color="k", ls="--", label="Cu melt limit")
axes[0].set(xlabel="x [cm]", ylabel="T_wg [K]", title="Gas-side wall temperature")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="x [cm]", ylabel="q [MW/m²]", title="Heat flux")
axes[1].legend(fontsize=8)
for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("study_5_3.png", dpi=120)

print("\nWhy copper wins: k_CuCrZr ≈ 320 W/mK vs k_SS304 ≈ 16 W/mK")
print("Wall resistance t/k: CuCrZr = 0.001/320 = 3.1μ, SS304 = 0.001/16 = 62μ")
print("SS304 has 20× higher thermal resistance → much hotter gas-side wall")
```

**Experiment 5.4 — Hand-calculate a single station (DO ON PAPER FIRST)**
```python
import numpy as np

# Given conditions at throat:
T_aw = 3200     # K (adiabatic wall temperature)
T_cool = 250    # K (coolant bulk temperature)
h_g = 15000     # W/m²K (from Bartz)
h_c = 30000     # W/m²K (from Dittus-Boelter)
k_wall = 320    # W/mK (CuCrZr)
t_wall = 0.0015 # m (1.5 mm)

# Step 1: total thermal resistance
R_gas = 1.0 / h_g
R_wall = t_wall / k_wall
R_cool = 1.0 / h_c
R_total = R_gas + R_wall + R_cool
print(f"Thermal resistances:")
print(f"  Gas film:  {R_gas*1e6:.1f} μK·m²/W  ({R_gas/R_total*100:.1f}%)")
print(f"  Wall:      {R_wall*1e6:.1f} μK·m²/W  ({R_wall/R_total*100:.1f}%)")
print(f"  Coolant:   {R_cool*1e6:.1f} μK·m²/W  ({R_cool/R_total*100:.1f}%)")
print(f"  Total:     {R_total*1e6:.1f} μK·m²/W")

# Step 2: heat flux
q = (T_aw - T_cool) / R_total
print(f"\nHeat flux: q = {q/1e6:.1f} MW/m²")

# Step 3: wall temperatures
T_wg = T_aw - q * R_gas
T_wc = T_cool + q * R_cool
print(f"T_wg = {T_wg:.0f} K  (gas-side wall)")
print(f"T_wc = {T_wc:.0f} K  (coolant-side wall)")

# Step 4: check survival
print(f"\nCuCrZr melts at ~1340 K. T_wg = {T_wg:.0f} K → ", end="")
print("SURVIVES ✓" if T_wg < 1340 else "MELTS ✗")

# Step 5: which resistance dominates?
print(f"\nThe gas-side film ({R_gas/R_total*100:.0f}%) is the bottleneck.")
print("To reduce T_wg, the most effective approach is to increase h_g")
print("(e.g., higher Pc, smaller throat) or use film cooling.")
```

### Questions to answer

1. Where along the nozzle axis is the peak heat flux? Why there specifically?
   (Think about what h_g ∝ (At/A)^0.9 means.)
2. From Experiment 5.2: is there an optimal number of channels? What determines
   the minimum (cooling) and maximum (pressure drop) limits?
3. If you double the wall thickness from 1.0 mm to 2.0 mm, by how much does T_wg
   increase? Calculate it by hand first, then verify with the code.
4. Why does Inconel 718 (high-strength alloy) perform WORSE as a chamber wall
   material than CuCrZr (which is actually weaker)? When would you prefer Inconel?
5. In Experiment 5.4, which thermal resistance dominates? If you could magically
   double only ONE of h_g, k_wall, or h_c, which would reduce T_wg the most?
6. **Design challenge:** For a 500 N thruster with Pc = 2 MPa and LOX/CH₄ at O/F = 3.4,
   design a cooling system that keeps max T_wg < 1000 K. Specify: number of channels,
   channel dimensions, wall material, wall thickness, and coolant flow rate.

---

## Module 6 — Feed Systems & Engine Cycles (Week 8–9)

### 6.1 The Pressure Budget

Every component in the feed system either **adds** or **drops** pressure:

```
  P_tank (or pump exit)          ← what you have to work with
  − ΔP_feedline (pipe friction)  ← lost to friction
  − ΔP_valves (fittings, turns)  ← lost to turbulence
  − ΔP_injector (pressure drop)  ← intentional! (for stability)
  = P_chamber                    ← what you need to achieve
```

The fundamental equation:

```
  ┌──────────────────────────────────────────────────────┐
  │  P_tank + ΔP_pump ≥ P_chamber + ΔP_line + ΔP_inj   │
  └──────────────────────────────────────────────────────┘
```

**Why the injector drops pressure on purpose:** If the injector had zero ΔP,
a slight increase in chamber pressure would reduce propellant flow, which
lowers Pc, which increases flow again — creating **chugging** instability.
The injector ΔP (typically 15–25% of Pc) provides **stiffness** against
pressure fluctuations.

### 6.2 Pipe Friction — Darcy-Weisbach

Pressure drop in a straight pipe:

```
ΔP = f × (L/D) × (ρV²/2)

Where:
  f = Darcy friction factor (dimensionless)
  L = pipe length [m]
  D = pipe inner diameter [m]
  V = flow velocity [m/s]
  ρ = fluid density [kg/m³]
```

The friction factor f depends on Reynolds number Re = ρVD/μ:

```
Laminar flow (Re < 2300):
  f = 64/Re        (exact, analytical solution — Hagen-Poiseuille)

Turbulent flow (Re > 4000):
  1/√f = -2 log₁₀(ε/(3.7D) + 5.74/Re^0.9)    (Swamee-Jain, explicit)

  This approximates the implicit Colebrook-White equation within 1%.
```

**Numerical example — LOX feed line:**

```
Given: ṁ = 2.0 kg/s, ρ_LOX = 1141 kg/m³, μ_LOX = 1.9×10⁻⁴ Pa·s
       D = 25 mm, L = 2.0 m, ε/D = 0.001 (commercial pipe)

Step 1: velocity
  A = π/4 × 0.025² = 4.91×10⁻⁴ m²
  V = ṁ/(ρA) = 2.0/(1141×4.91×10⁻⁴) = 3.57 m/s

Step 2: Reynolds number
  Re = 1141 × 3.57 × 0.025 / 1.9×10⁻⁴ = 535,600  (highly turbulent!)

Step 3: friction factor (Swamee-Jain)
  f = 0.25 / [log₁₀(0.001/3.7 + 5.74/535600^0.9)]²
  f ≈ 0.020

Step 4: pressure drop
  ΔP = 0.020 × (2.0/0.025) × (1141×3.57²/2)
     = 0.020 × 80 × 7270
     = 11,632 Pa ≈ 11.6 kPa   (very small compared to Pc = 7 MPa)
```

### 6.3 Valve and Fitting Losses — K-factor Method

Fittings (elbows, tees, valves) create extra pressure drops:

```
ΔP_fitting = K × ρV²/2

Typical K values:
  90° elbow:        K ≈ 0.3–0.7
  45° elbow:        K ≈ 0.2
  Gate valve (open): K ≈ 0.1–0.3
  Ball valve:        K ≈ 0.05
  Tee (branch):      K ≈ 1.0–1.5
  Check valve:       K ≈ 2.0–5.0
```

### 6.4 Blowdown Tanks — Polytropic Process

In a pressure-fed engine, a high-pressure gas (He or N₂) pushes propellant
out of the tank. As propellant is expelled, the gas expands and pressure drops.

The process is **polytropic** (between isothermal and adiabatic):

```
P × V^n = constant

Where n = polytropic exponent:
  n = 1.0  → isothermal (infinitely slow, perfect heat exchange)
  n = γ    → adiabatic (fast, no heat exchange)
  Real blowdown: n ≈ 1.2–1.5 (depends on insulation and burn time)
```

**Derivation of the blowdown curve:**

```
Initial state: gas volume V_ull_0 at pressure P₀
After expelling mass m_prop (volume m_prop/ρ_prop):
  V_ull = V_ull_0 + m_prop/ρ_prop

From polytropic relation:
  P₀ × V_ull_0^n = P × V_ull^n

  ┌────────────────────────────────────────────────────────┐
  │  P(m_prop) = P₀ × (V_ull_0 / (V_ull_0 + m/ρ))^n     │
  └────────────────────────────────────────────────────────┘

With constant mass flow ṁ and m_prop = ṁ·t:
  P(t) = P₀ × (V_ull_0 / (V_ull_0 + ṁ·t/ρ))^n
```

**Numerical example:**

```
Tank: V_total = 50 liters, 5% ullage → V_ull_0 = 2.5 L
      P₀ = 8 MPa, propellant = LOX (ρ = 1141 kg/m³)
      n = 1.4 (He pressurant, fast blowdown)
      ṁ = 0.5 kg/s

After 20 seconds (10 kg expelled):
  V_ull = 0.0025 + 10/1141 = 0.0025 + 0.00877 = 0.01127 m³
  P = 8 × (0.0025/0.01127)^1.4 = 8 × 0.222^1.4 = 8 × 0.117 = 0.94 MPa

Pressure dropped from 8 to 0.94 MPa — an 88% drop!
This is why blowdown systems work poorly for long burns.
```

### 6.5 Why Helium is the Best Pressurant

```
Pressurant comparison at same tank conditions:

Gas    γ      MW [g/mol]   Density at 8 MPa, 300K   Mass for V=50L
He     1.667   4.003       12.9 kg/m³                0.64 kg
N₂     1.400  28.014       90.5 kg/m³                4.53 kg
Ar     1.667  39.948      128.8 kg/m³                6.44 kg

Helium advantages:
  ✓ 7× lighter than N₂ → saves tank mass
  ✓ Higher γ → "stiffer" blowdown curve → more uniform pressure
  ✓ Chemically inert (won't react with hot propellant)
  ✓ Won't condense at LOX temperatures (He bp = 4 K vs N₂ bp = 77 K)

Helium disadvantage:
  ✗ Expensive (~$5-15/kg vs ~$0.10/kg for N₂)
  ✗ Harder to seal (small atoms leak through O-rings)
```

### 6.6 Gas Generator (GG) Cycle

The GG cycle bleeds a small fraction of propellant to drive the turbopump:

```
  ┌─────────┐  main flow     ┌───────────┐
  │  Tank   │───────────────→│  Chamber   │───→ Main nozzle (high Isp)
  │ (P_tank)│                │   (Pc)     │
  └────┬────┘                └───────────┘
       │
       │ bleed flow (~2-5%)
       ↓
  ┌─────────┐     exhaust    ┌───────────┐
  │   GG    │───────────────→│  Turbine   │───→ Dump/small nozzle (low Isp)
  │(T~800K) │                │           │
  └─────────┘                └─────┬─────┘
                                   │ shaft
                                   ↓
                              ┌─────────┐
                              │  Pump    │ (compresses main propellant)
                              └─────────┘
```

**Isp penalty from GG bleed:**

```
Total Isp = Isp_main × (1 - f_bleed) + Isp_dump × f_bleed

Example: Isp_main = 430 s, Isp_dump = 150 s, f_bleed = 3%
  Isp_total = 430 × 0.97 + 150 × 0.03 = 417.1 + 4.5 = 421.6 s

Isp loss: (430 - 421.6)/430 = 2.0%
```

**Why limit GG temperature to 700–1100 K?** The turbine blades must survive
without cooling (unlike the chamber walls). Nickel superalloys survive up
to ~1100 K. Running the GG fuel-rich keeps T low but wastes more propellant.

### 6.7 Expander Cycle

The expander cycle uses the heat from regenerative cooling to drive the turbine:

```
  ┌─────────┐              ┌───────────┐
  │  Tank   │─→ Pump ─→ ───│ Cooling   │ (fuel heated by chamber wall)
  │         │              │ Channels  │
  └─────────┘              └─────┬─────┘
                                 │ hot fuel (500-800 K)
                                 ↓
                            ┌─────────┐
                            │ Turbine  │ (extracts energy from hot fuel)
                            └─────┬───┘
                                  │ shaft → drives pump
                                  ↓
                            ┌───────────┐
                            │  Injector  │ → Chamber → Nozzle
                            └───────────┘

No propellant is wasted — ALL fuel goes through the chamber!
100% of propellant produces thrust → no Isp penalty.
```

**Limitation:** The turbine power is limited by the heat absorbed in cooling:

```
P_turbine ≤ ṁ_fuel × cₚ × (T_cool_exit - T_cool_inlet) × η_turbine

For LOX/H₂: H₂ has cₚ ≈ 14,000 J/kgK → lots of energy absorbed → expander works
For LOX/RP-1: RP-1 has cₚ ≈ 2,000 J/kgK → limited energy → expander barely works
```

This is why the RL-10 (LOX/H₂) uses an expander cycle, but the Merlin (LOX/RP-1)
uses a GG cycle.

### 6.8 Pump Sizing — Head and Power

A centrifugal pump is characterized by:

```
Head H:  how much pressure it adds (in meters of fluid column)
  ΔP = ρ × g × H
  H_design typically 500–3000 m for rocket pumps

Power:   P = ρ × g × Q × H / η_pump
  η_pump ≈ 0.60–0.75 for rocket pumps

Flow rate Q = ṁ/ρ [m³/s]

Off-design: H(Q) = H_design × [1 - k_H × (Q/Q_design - 1)²]
```

**Numerical example — LOX pump sizing:**

```
ṁ_ox = 1.5 kg/s, ρ_LOX = 1141 kg/m³, P_required = 7 MPa + 2 MPa (margin) = 9 MPa
P_tank = 0.3 MPa

ΔP_pump = 9 - 0.3 = 8.7 MPa
H = ΔP/(ρg) = 8.7×10⁶ / (1141 × 9.81) = 777 m   (that's a tall water column!)
Q = 1.5/1141 = 1.31×10⁻³ m³/s = 1.31 L/s

Power = 1141 × 9.81 × 1.31×10⁻³ × 777 / 0.70
      = 14,900 / 0.70 = 21,300 W ≈ 21 kW

For comparison: 21 kW ≈ 28 horsepower — like a small car engine,
but spinning at 30,000–60,000 RPM and weighing only 2–5 kg.
```

### 6.9 NPSH — Cavitation Prevention

**Net Positive Suction Head** ensures the pump inlet pressure stays above the
propellant's vapor pressure. If it drops below, the liquid boils (cavitates)
and the pump destroys itself:

```
NPSH_available = (P_inlet - P_vapor)/(ρg) + z_head - h_loss

NPSH_required: given by pump manufacturer (typically 1–10 m)

Must have: NPSH_available > NPSH_required

For LOX at 90 K: P_vapor ≈ 101 kPa
  P_inlet = P_tank - ΔP_feedline = 300 kPa - 12 kPa = 288 kPa
  NPSH_avail = (288,000 - 101,000)/(1141×9.81) = 187,000/11,193 = 16.7 m  ✓

If P_tank drops to 150 kPa (late blowdown):
  NPSH_avail = (138,000 - 101,000)/11,193 = 3.3 m  ← dangerously low!
```

### Experiments

**Experiment 6.1 — Blowdown curve comparison**
```python
from tank import Tank
import numpy as np, matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Compare pressurant gases
for pressurant, color in [("He", "steelblue"), ("N2", "orange")]:
    tank = Tank(V_tank=0.05, P_0=8e6, propellant="LOX",
                pressurant=pressurant, ullage_fraction=0.05)
    t_burn = tank.m_prop_0 / 0.5  # burn time at mdot=0.5 kg/s
    bd = tank.blowdown_curve(mdot=0.5, t_burn=t_burn * 0.95)
    axes[0].plot(bd["t"], bd["P"]/1e6, label=pressurant, color=color)
    axes[1].plot(bd["t"], bd["m_prop"], label=pressurant, color=color)

axes[0].set(xlabel="Time [s]", ylabel="Pressure [MPa]", title="Tank pressure")
axes[0].axhline(2.0, color="r", ls="--", alpha=0.5, label="Min Pc + margin")
axes[0].legend()
axes[1].set(xlabel="Time [s]", ylabel="m_prop [kg]", title="Propellant remaining")
axes[1].legend()

# Compare polytropic exponents
for n, label in [(1.0, "Isothermal n=1"), (1.3, "n=1.3"), (1.667, "Adiabatic n=γ(He)")]:
    tank = Tank(V_tank=0.05, P_0=8e6, propellant="LOX",
                pressurant="He", ullage_fraction=0.05)
    tank.n = n  # override polytropic exponent
    bd = tank.blowdown_curve(mdot=0.5, t_burn=60)
    axes[2].plot(bd["t"], bd["P"]/1e6, label=label)

axes[2].set(xlabel="Time [s]", ylabel="Pressure [MPa]",
            title="Effect of polytropic exponent")
axes[2].legend()

for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("study_6_1.png", dpi=120)

print("Higher n → steeper drop → less usable propellant before P drops too low.")
print("He (γ=1.667) gives a steeper drop than N₂ (γ=1.4), but He is lighter and won't condense.")
```

**Experiment 6.2 — Pipe sizing**
```python
from feed_system import line_pressure_drop

mdot = 0.5    # kg/s
rho  = 1141.0 # LOX density
mu   = 1.9e-4 # LOX viscosity

print(f"{'D [mm]':>7s}  {'V [m/s]':>8s}  {'Re':>8s}  {'f':>7s}  {'ΔP [kPa]':>9s}  {'% of 8MPa':>10s}")
print("-" * 60)
for D_mm in [8, 10, 15, 20, 25, 30, 40, 50]:
    D = D_mm * 1e-3
    r = line_pressure_drop(mdot, rho, mu, D=D, L=2.0)
    frac = r["dP_total"] / 8e6 * 100
    print(f"{D_mm:7.0f}  {r['velocity']:8.1f}  {r['Re']:8.0f}  {r['f']:7.4f}  "
          f"{r['dP_total']/1e3:9.1f}  {frac:10.3f}%")

print("\nDesign rule: keep ΔP < 1-2% of tank pressure")
print("But also keep V < 10 m/s to avoid erosion and water hammer")
```

**Experiment 6.3 — GG cycle balance**
```python
from turbopump import Pump, Turbine
from feed_system import FeedLine, GasGeneratorCycle
import matplotlib.pyplot as plt

feed_ox   = FeedLine(D=0.025, L=2.0, propellant="LOX")
feed_fuel = FeedLine(D=0.020, L=2.5, propellant="H2")
pump_o = Pump(eta_design=0.70)
pump_f = Pump(eta_design=0.65)
turb   = Turbine(eta_turbine=0.60)

gg = GasGeneratorCycle(pump_o, pump_f, turb, feed_ox, feed_fuel,
                        T_gg=900, gamma_gg=1.3, cp_gg=2500)

# Sweep GG temperature
T_ggs = [600, 700, 800, 900, 1000, 1100, 1200]
bleeds = []
for T in T_ggs:
    gg.T_gg = T
    r = gg.solve(0.3e6, 0.3e6, 7e6, mdot_total=1.0, OF=6.0)
    bleeds.append(r["bleed_fraction"] * 100)
    print(f"T_gg = {T:5d} K  →  bleed = {r['bleed_fraction']*100:.2f}%  "
          f"  Isp loss ≈ {r['bleed_fraction']*(430-150)/430*100:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(T_ggs, bleeds, "o-", color="firebrick", lw=2)
plt.xlabel("GG temperature [K]"); plt.ylabel("Bleed fraction [%]")
plt.title("Propellant bleed vs GG temperature")
plt.axvline(1100, color="k", ls="--", alpha=0.5, label="Nickel alloy limit ~1100 K")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("study_6_3.png", dpi=120)
```

**Experiment 6.4 — Hand-calculate pump power (DO ON PAPER FIRST)**
```python
import numpy as np

# LOX pump sizing for 500 N thruster
F_thrust = 500       # N
Isp = 340            # s (LOX/CH4 estimate)
OF = 3.4
Pc = 2e6             # Pa (2 MPa)
P_tank = 0.3e6       # Pa (low tank pressure → need pump)
rho_LOX = 1141       # kg/m³
eta_pump = 0.65

# Step 1: mass flow rates
mdot = F_thrust / (Isp * 9.80665)
mdot_ox = mdot * OF / (1 + OF)
mdot_f = mdot / (1 + OF)
print(f"Total ṁ = {mdot:.3f} kg/s")
print(f"ṁ_ox = {mdot_ox:.3f} kg/s, ṁ_fuel = {mdot_f:.3f} kg/s")

# Step 2: pump head
dP = Pc * 1.25 - P_tank   # 25% margin over Pc
H = dP / (rho_LOX * 9.81)
print(f"\nΔP_pump = {dP/1e6:.2f} MPa → Head = {H:.1f} m")

# Step 3: pump power
Q = mdot_ox / rho_LOX
P_pump = rho_LOX * 9.81 * Q * H / eta_pump
print(f"Q = {Q*1e6:.1f} mL/s")
print(f"Pump power = {P_pump:.1f} W = {P_pump/746:.2f} hp")
print(f"\nThat's only {P_pump:.0f} W — tiny! For a 500 N thruster, a pressure-fed")
print(f"system (higher tank pressure) might be simpler than a turbopump.")
```

### Questions to answer

1. In Experiment 6.1: why does N₂ cause a steeper pressure drop than He at the same conditions?
   (Hint: think about what happens physically as the pressurant gas expands.)
2. From Experiment 6.2: what pipe diameter keeps ΔP < 1% of tank pressure? What's
   the flow velocity at that diameter? Is it reasonable?
3. Why is an injector pressure drop of 15–25% of Pc considered "good"? What happens
   if ΔP_inj < 5% of Pc? What if ΔP_inj > 50% of Pc?
4. In Experiment 6.3: what limits T_gg? What would happen if you ran the GG at
   T_gg = 3000 K (same as the main chamber)?
5. For a 60-second burn with a 50-liter tank, at what time does the tank pressure
   drop below the minimum needed for Pc = 2 MPa? (Use the blowdown equation from §6.4.)
6. **Design decision:** For a 500 N thruster with 60 s burn time, would you use:
   (a) pressure-fed with high-pressure tank, or (b) turbopump with low-pressure tank?
   Calculate the total system mass for each option and compare.
   Assumptions: tank wall thickness ∝ P_tank, pump mass ≈ 2 kg.

---

## Module 7 — Full System Integration (Week 10)

### Capstone Project: Design Your Own Engine

Design a small vacuum thruster for a spacecraft upper stage.

**Constraints:**
- Propellants: LOX/CH₄ (modern, relatively storable)
- Target thrust: 500 N vacuum
- Burn time: 60 s
- Max nozzle length: 15 cm
- Feed system: pressure-fed (simplicity over performance)

### Task 1 — Estimate the Throat Area

```
Step 1: estimate Isp
  LOX/CH₄ at O/F=3.4, Pc=2 MPa: Isp_vac ≈ 340 s (from Module 1)

Step 2: mass flow rate
  ṁ = F / (Isp × g₀) = 500 / (340 × 9.81) = 0.150 kg/s

Step 3: C* (from Module 1 experiments or estimate ≈ 1800 m/s)

Step 4: throat area
  At = ṁ × C* / Pc = 0.150 × 1800 / 2×10⁶ = 1.35×10⁻⁴ m²
  Rt = √(At/π) = √(1.35×10⁻⁴/π) = 0.00656 m ≈ 6.6 mm

Alternatively using Cf:
  At = F / (Pc × Cf) ≈ 500 / (2×10⁶ × 1.8) = 1.39×10⁻⁴ m²
  Rt ≈ 6.65 mm
```

Verify with the code:
```python
from chamber import chamber_conditions
import numpy as np

c = chamber_conditions("O2", "CH4", OF=3.4, P0=2e6)
print(f"T₀={c['T0']:.0f} K, γ={c['gamma']:.3f}, R={c['R']:.1f}, C*={c['Cstar']:.0f} m/s")

mdot = 500 / (340 * 9.80665)
At = mdot * c["Cstar"] / 2e6
Rt = np.sqrt(At / np.pi)
print(f"ṁ = {mdot:.4f} kg/s, At = {At*1e4:.3f} cm², Rt = {Rt*1000:.2f} mm")
```

### Task 2 — Injector Sizing

```
Mass flow split:
  ṁ_ox  = ṁ × OF/(1+OF) = 0.150 × 3.4/4.4 = 0.116 kg/s
  ṁ_fuel = ṁ / (1+OF)    = 0.150 / 4.4      = 0.034 kg/s

Injector orifice areas (Bernoulli):
  A = ṁ / (Cd × √(2ρΔP))

  P_tank = 3.0 MPa (50% above Pc)
  ΔP_inj = P_tank - Pc = 1.0 MPa
  Cd = 0.65 (sharp-edged orifice)

  A_ox  = 0.116 / (0.65 × √(2×1141×1e6)) = 0.116 / (0.65×47,780) = 3.74×10⁻⁶ m²
  A_fuel = 0.034 / (0.65 × √(2×423×1e6))  = 0.034 / (0.65×29,086) = 1.80×10⁻⁶ m²
```

### Task 3 — Run the Simulation

```bash
python engine_system.py --mode simulate \
    --fuel CH4 --oxidizer O2 \
    --throat_radius 0.0066 \
    --p_tank_f 3e6 --p_tank_o 3e6 \
    --A_f 1.80e-6 --A_o 3.74e-6 \
    --Pe 100  \
    --name my_500N_thruster
```

### Task 4 — Optimize the Nozzle

```bash
python engine_system.py --mode optimize \
    --fuel CH4 --oxidizer O2 \
    --throat_radius 0.0066 \
    --p_tank_f 3e6 --p_tank_o 3e6 \
    --A_f 1.80e-6 --A_o 3.74e-6 \
    --Pe 100 --max_length_m 0.15 \
    --n_eval 100 --name my_500N_opt
```

### Task 5 — Propellant Budget and Mission Analysis

```python
from tank import Tank
import numpy as np

# From your simulation:
Isp = 340     # s (replace with actual result)
F   = 500     # N
mdot = F / (Isp * 9.80665)
OF = 3.4

mdot_ox = mdot * OF / (1 + OF)
mdot_f  = mdot / (1 + OF)
t_burn  = 60  # seconds

m_ox_needed  = mdot_ox * t_burn
m_f_needed   = mdot_f * t_burn

print(f"Mass flow: ṁ_total = {mdot:.4f} kg/s")
print(f"  ṁ_ox  = {mdot_ox:.4f} kg/s → {m_ox_needed:.2f} kg for {t_burn}s burn")
print(f"  ṁ_fuel = {mdot_f:.4f} kg/s → {m_f_needed:.2f} kg for {t_burn}s burn")

# Tank sizing
# LOX density = 1141 kg/m³, LCH4 density ≈ 423 kg/m³
V_ox = m_ox_needed / 1141 * 1.15   # 15% ullage margin
V_f  = m_f_needed / 423 * 1.15

print(f"\nTank volumes (with 15% ullage):")
print(f"  LOX tank:  {V_ox*1000:.1f} liters ({m_ox_needed:.2f} kg)")
print(f"  CH₄ tank:  {V_f*1000:.1f} liters ({m_f_needed:.2f} kg)")

# Blowdown check
tank_ox = Tank(V_tank=V_ox, P_0=3e6, propellant="LOX", pressurant="He")
bd = tank_ox.blowdown_curve(mdot=mdot_ox, t_burn=t_burn)
P_min = min(bd["P"])
print(f"\nLOX tank: P_0 = 3.0 MPa → P_final = {P_min/1e6:.2f} MPa")
print(f"Need P > Pc + margin = {2e6*1.15/1e6:.2f} MPa → ", end="")
print("OK ✓" if P_min > 2e6*1.15 else "INSUFFICIENT ✗ — increase P₀ or tank size")

# Delta-v budget
m_prop = m_ox_needed + m_f_needed
m_dry  = 5.0  # kg (estimate: engine + tanks + structure)
dv = Isp * 9.80665 * np.log((m_dry + m_prop) / m_dry)
print(f"\nTotal propellant: {m_prop:.2f} kg")
print(f"Assumed dry mass: {m_dry:.1f} kg")
print(f"Δv = {dv:.0f} m/s")
print(f"(For reference: ~200 m/s is typical for satellite orbit raising)")
```

### Task 6 — Write a Design Report

Write a 1–2 page report answering:

1. What Isp did your engine achieve? How does it compare to:
   - RL-10 (LOX/H₂, Isp ≈ 465 s)
   - Raptor Vacuum (LOX/CH₄, Isp ≈ 380 s)
   - Your result

2. What limited your performance?
   - Geometry constraint (15 cm max length)?
   - Low chamber pressure (2 MPa)?
   - Propellant choice?
   - Something else?

3. What would you change to improve Isp by 5%?
   - Higher Pc? (quantify: how much does Isp change with Pc?)
   - Longer nozzle? (quantify: what length gives 5% more?)
   - Different propellant? (quantify: LOX/H₂ vs LOX/CH₄ gain)

4. Would this engine be useful for a real mission? What would you use it for?

---

## Module 8 — Going Deeper (Optional, Week 11–12)

### 8.1 Eckert Reference Temperature Method

The Bartz correlation assumes uniform gas properties. In reality, viscosity
and conductivity vary significantly across the boundary layer.

**The idea:** evaluate gas properties not at T₀ or T_wall, but at a
"reference temperature" T_ref that better represents the boundary layer:

```
T_ref = 0.5·T_local + 0.22·r·(γ-1)/2·M²·T₀ + 0.28·T_wall

This gives a temperature "between" the hot gas and the cold wall.
```

**Exercise:** Implement this in `cooling.py` by adding a `T_ref` calculation
before the Bartz call. Compare the corrected h_g profile with the uncorrected one.
Does it increase or decrease predicted wall temperature? By how much?

### 8.2 Real-Gas Equation of State

At high pressures (>10 MPa), the ideal gas law `P = ρRT` has errors of 1–3%.
The **Peng-Robinson** equation is a common correction:

```
P = RT/(V-b) - a·α/(V²+2bV-b²)

Where:
  a = 0.45724·R²·Tc²/Pc    (attraction parameter)
  b = 0.07780·R·Tc/Pc       (repulsion parameter)
  α(T) = [1 + κ(1-√(T/Tc))]²
  κ = 0.37464 + 1.54226ω - 0.26992ω²  (ω = acentric factor)
```

**Exercise:** Implement Peng-Robinson for the combustion products and compare
Isp predictions with the ideal gas model at Pc = 5, 10, 20, 30 MPa.
At what pressure does the difference exceed 1%?

### 8.3 Combustion Efficiency η_c*

Real engines don't achieve 100% theoretical C*:

```
C*_actual = η_c* × C*_theoretical

Typical values:
  Small engines (F < 1 kN):  η_c* = 0.92–0.95
  Medium engines (1–100 kN): η_c* = 0.95–0.97
  Large engines (>100 kN):   η_c* = 0.97–0.99
  (Merlin 1D: η_c* ≈ 0.985, SSME: η_c* ≈ 0.995)
```

**Exercise:** Add an `eta_cstar` parameter to `engine_system.py`. Run your 500 N
thruster at η_c* = 0.93, 0.95, 0.97, 0.99. Plot Isp vs η_c*. How much Isp
do you lose at 93% vs 99% combustion efficiency?

### 8.4 Specific Speed and Turbopump Scaling

Turbopump design uses the **specific speed** Ns to classify pump types:

```
Ns = ω·√Q / (g·H)^(3/4)     [dimensionless, ω in rad/s]

  Ns < 0.3:  positive displacement pump (gear, piston)
  0.3–1.5:   centrifugal pump (most rocket pumps)
  1.5–4.0:   mixed-flow pump
  > 4.0:     axial pump (high flow, low head)
```

**Exercise:** For the Aquila VAC engine (10.3 kN, Pc = 9 MPa), calculate the
required specific speed and determine the impeller diameter assuming Ns = 0.5
and RPM = 30,000. Would a single-stage centrifugal pump work?

### 8.5 Parametric Mass Estimation

Estimate engine mass from empirical correlations (Huzel & Huang):

```
Chamber mass (thin-walled cylinder):
  m_chamber = ρ_wall × π × D_c × L_c × t_wall
  t_wall = Pc × D_c / (2 × σ_allow)    (hoop stress)

Nozzle mass (thin shell):
  m_nozzle ≈ ρ_wall × π × ∫[x_throat to x_exit] 2·r(x)·t(x)·dx
  (approximate as cone: m ≈ ρ × π × (Rt+Re) × L_nozzle × t_avg)

Total: m_engine ≈ m_chamber + m_nozzle + m_injector + m_valves
```

**Exercise:** Estimate the dry mass of your 500 N thruster. Then compute
the thrust-to-weight ratio F/(m_engine × g₀). Compare with:
- Merlin 1D: T/W ≈ 180
- Raptor: T/W ≈ 200
- RL-10: T/W ≈ 40 (vacuum optimized, not T/W optimized)

### 8.6 Multi-Objective Optimization

The current optimizer maximizes only Isp. Real engine design balances
multiple objectives:

```
Maximize:  Isp_vac
Minimize:  Nozzle mass
Minimize:  Max heat flux (easier cooling)
Minimize:  Nozzle length (packaging)
```

**Exercise:** Modify `nozzle_optimizer.py` to use a weighted objective:

```python
objective = w1 * Isp_vac - w2 * length - w3 * max_heat_flux
```

Plot the **Pareto front** for Isp vs length by sweeping w1/w2. A Pareto
front shows all designs where you can't improve one metric without
worsening another.

---

## Recommended Reading

| Resource | Covers | Where to find |
|----------|--------|---------------|
| Sutton & Biblarz — *Rocket Propulsion Elements* (9th ed.) | Everything: thermodynamics, nozzles, cooling, cycles | Library / O'Reilly |
| Huzel & Huang — *Modern Engineering for Design of Liquid-Propellant Rocket Engines* | Practical design methodology, sizing | NASA NTRS (free PDF) |
| Anderson — *Modern Compressible Flow* (3rd ed.) | Isentropic flow, Method of Characteristics, shocks | Library |
| Bartz (1957) — *A Simple Equation for Rapid Estimation of Rocket Nozzle Convective Heat Transfer Coefficients* | Bartz correlation original paper | NASA NTRS (free) |
| Rao (1958) — *Exhaust Nozzle Contour for Optimum Thrust* | Rao contour theory | NASA NTRS (free) |
| Cantera documentation — cantera.org | Chemical kinetics, HP equilibrium | Online (free) |
| Storn & Price (1997) — *Differential Evolution* | The DE algorithm | Journal of Global Optimization |
| MIT OCW 16.512 — Rocket Propulsion (lecture notes) | Graduate-level theory, free online | ocw.mit.edu |

---

## Progress Checklist

- [ ] Module 1: Ran O/F sweep, understand C* and T₀/MW trade-off, hand-calculated C*
- [ ] Module 2: Can derive isentropic relations from energy conservation, hand-verified M/T/P/V
- [ ] Module 3: Generated 4+ contours with different θᵢ/θₑ, understand λ, drew contour by hand
- [ ] Module 4: Plotted objective landscape, ran optimizer, compared DE vs random search
- [ ] Module 5: Full cooling loop, found minimum channel count, hand-calculated wall temp at one station
- [ ] Module 6: Plotted blowdown curves, sized feed pipes, understood GG vs expander trade-off
- [ ] Module 7: Designed and simulated your own 500 N thruster, wrote design report
- [ ] Module 8: ≥1 extension implemented and tested

**You understand the project when you can answer ALL of these:**

> 1. *"Why does LOX/H₂ have higher Isp than LOX/CH₄ despite similar flame temperature?"*
>    (MW effect on exhaust velocity)
>
> 2. *"Why does a rocket nozzle converge then diverge?"*
>    (Subsonic vs supersonic area-velocity relation from Module 2)
>
> 3. *"Why is the throat the hardest point to cool?"*
>    (h_g ∝ (At/A)^0.9 from Bartz, plus T_aw ≈ T₀ at M=1)
>
> 4. *"Why does the Raptor achieve higher Isp than the Merlin?"*
>    (Chamber pressure 300 vs 100 bar suppresses dissociation → higher C*,
>     full-flow staged combustion vs GG cycle → no bleed loss,
>     LOX/CH₄ vs LOX/RP-1 → lower MW)

---

*Last updated: 2026-04-10*
