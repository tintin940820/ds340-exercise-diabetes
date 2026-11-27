#!/bin/bash
# runs the whole project from the two clean tables in data/
# step 1 makes the train / validation / test split, every other step reads it
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p results figures
for s in make_splits q1_part1_which_exercise q1_part2_exercise_vs_factors q1_checks q2_generations parent_paper_1 novelty check_results; do
  echo "=== $s"
  python3 -W ignore $s.py | tee results/log_$s.txt
done
