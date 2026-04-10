"""
runner.py — Quick start examples for the engine system

Run any of the examples below directly or use engine_system.py for full CLI.
"""

import subprocess

EXAMPLES = {
    "1": dict(
        desc="LOX/H2 simulate",
        cmd=["python", "engine_system.py", "--mode", "simulate",
             "--fuel", "H2", "--oxidizer", "O2",
             "--throat_radius", "0.04",
             "--p_tank_f", "8e6", "--p_tank_o", "8e6",
             "--A_f", "2e-4", "--A_o", "2e-4",
             "--name", "lox_h2_demo"]
    ),
    "2": dict(
        desc="LOX/CH4 simulate",
        cmd=["python", "engine_system.py", "--mode", "simulate",
             "--fuel", "CH4", "--oxidizer", "O2",
             "--throat_radius", "0.05",
             "--p_tank_f", "8e6", "--p_tank_o", "8e6",
             "--A_f", "2.5e-4", "--A_o", "2.5e-4",
             "--name", "lox_ch4_demo"]
    ),
    "3": dict(
        desc="LOX/H2 optimize nozzle shape",
        cmd=["python", "engine_system.py", "--mode", "optimize",
             "--fuel", "H2", "--oxidizer", "O2",
             "--throat_radius", "0.04",
             "--p_tank_f", "8e6", "--p_tank_o", "8e6",
             "--A_f", "2e-4", "--A_o", "2e-4",
             "--n_eval", "80", "--name", "lox_h2_opt_demo"]
    ),
    "4": dict(
        desc="Aquila VAC (LOX/C3H8) simulate — 94 kN vacuum engine",
        cmd=["python", "aquila_vac.py"]
    ),
}

if __name__ == "__main__":
    print("Available examples:")
    for k, v in EXAMPLES.items():
        print(f"  {k}: {v['desc']}")
    choice = input("\nRun example [1/2/3/4] or press Enter for 1: ").strip() or "1"
    ex = EXAMPLES.get(choice, EXAMPLES["1"])
    print(f"\nRunning: {' '.join(ex['cmd'])}\n")
    subprocess.run(ex["cmd"])
