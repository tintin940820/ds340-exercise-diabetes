# runs the whole project in order (works on Windows, Mac and Linux)
#   python run_all.py
# step 1 makes the train / validation / test split, every other step reads it.
# the last step (check_results.py) compares the new results with the numbers in our paper.
# each step's printout is also saved to results/log_<step>.txt

import os
import subprocess
import sys
import time

STEPS = ["make_splits", "q1_part1_which_exercise", "q1_part2_exercise_vs_factors",
         "q1_checks", "q2_generations", "parent_paper_1", "novelty", "check_results"]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

start = time.time()
for s in STEPS:
    print(f"\n=== {s}", flush=True)
    t = time.time()
    p = subprocess.Popen([sys.executable, "-W", "ignore", f"{s}.py"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         env=dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg"))
    with open(f"results/log_{s}.txt", "w", encoding="utf-8") as log:
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
    if p.wait() != 0:
        sys.exit(f"\n{s}.py stopped with an error")
    print(f"({s} took {(time.time() - t) / 60:.1f} min)")

print(f"\nall done in {(time.time() - start) / 60:.1f} min")
