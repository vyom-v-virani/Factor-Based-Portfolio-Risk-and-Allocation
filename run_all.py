import argparse
import os
import re
import subprocess
import sys

import config

parser = argparse.ArgumentParser(description="Run Phases 1 to 7 in order.")
parser.add_argument("--refresh", action="store_true", help="delete the cached downloads so the data is downloaded again")
args = parser.parse_args()

project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)

if args.refresh and os.path.isdir(config.CACHE_DIR):
    for name in os.listdir(config.CACHE_DIR):
        if name.endswith(".parquet"):
            os.remove(os.path.join(config.CACHE_DIR, name))
            print(f"Removed cached file {name}")

results = []
for phase in range(1, 8):
    script = f"run_phase{phase}.py"
    print(f"Running {script} ...", flush=True)
    done = subprocess.run([sys.executable, script], capture_output=True, text=True)
    match = re.search(r"(\d+)/(\d+) checks passed", done.stdout)
    if done.returncode != 0 or match is None:
        print(done.stdout[-3000:])
        print(done.stderr[-3000:])
        print(f"FAILED: {script} (exit code {done.returncode}). Stopping.")
        sys.exit(1)
    results.append((phase, int(match.group(1)), int(match.group(2))))

print("\n=== SUMMARY ===")
any_failed = False
for phase, passed, total in results:
    if passed == total:
        status = "OK"
    else:
        status = "CHECKS FAILED"
        any_failed = True
    print(f"Phase {phase}: {passed}/{total} checks passed  [{status}]")
if any_failed:
    sys.exit(1)
