"""
post_processor/process_nozzle.py
Copies rocket_nozzle results into LLM_based_DB/raw/ so wiki_maintainer picks them up.

Usage:
  python process_nozzle.py --run_dir ../results/rao_opt_01
  python process_nozzle.py --watch          # auto-process new runs
  python process_nozzle.py                  # process all existing results
"""

import os, sys, time, argparse, shutil

NOZZLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(NOZZLE_ROOT, "results")

# LLM_based_DB lives alongside rocket_nozzle inside the project root
PROJECT_ROOT = os.path.dirname(NOZZLE_ROOT)
LLM_DB_ROOT  = os.path.join(PROJECT_ROOT, "LLM_based_DB")
RAW_DIR      = os.path.join(LLM_DB_ROOT, "raw")


def find_log(run_dir):
    for f in os.listdir(run_dir):
        if f.startswith("log."):
            return os.path.join(run_dir, f)
    return None


def copy_run_to_db(run_dir):
    run_name = os.path.basename(run_dir)
    os.makedirs(RAW_DIR, exist_ok=True)

    log_src = find_log(run_dir)
    if not log_src:
        print(f"[SKIP] No log.* in {run_dir}")
        return

    # wiki_maintainer watches raw/*_report.md
    report_dst = os.path.join(RAW_DIR, f"{run_name}_report.md")
    shutil.copy2(log_src, report_dst)
    print(f"[COPY] {run_name} -> {report_dst}")

    # Copy supporting files into raw/<run_name>/
    ref_dir = os.path.join(RAW_DIR, run_name)
    os.makedirs(ref_dir, exist_ok=True)
    extras = ["results.json", "contour.csv", "optimization_history.csv",
              "optimizer_plots.png", "nozzle_plots.png"]
    for fname in extras:
        src = os.path.join(run_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, ref_dir)
            print(f"  [+] {fname}")


def process_single(run_dir):
    if not os.path.isdir(run_dir):
        print(f"[ERR] Not a directory: {run_dir}")
        return
    copy_run_to_db(run_dir)


def watch_mode(interval=5):
    print(f"[WATCH] {RESULTS_DIR}  (interval={interval}s)")
    seen = set(os.listdir(RESULTS_DIR)) if os.path.isdir(RESULTS_DIR) else set()
    while True:
        time.sleep(interval)
        if not os.path.isdir(RESULTS_DIR):
            continue
        current = set(os.listdir(RESULTS_DIR))
        for run in current - seen:
            run_dir = os.path.join(RESULTS_DIR, run)
            if os.path.isdir(run_dir):
                time.sleep(2)
                copy_run_to_db(run_dir)
        seen = current


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--watch",   action="store_true")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    if args.watch:
        watch_mode(args.interval)
    elif args.run_dir:
        process_single(os.path.abspath(args.run_dir))
    else:
        if os.path.isdir(RESULTS_DIR):
            for run in sorted(os.listdir(RESULTS_DIR)):
                process_single(os.path.join(RESULTS_DIR, run))
        else:
            print(f"[ERR] {RESULTS_DIR} not found")
