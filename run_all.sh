#!/bin/bash
# Runs the three questions in order (about 1 minute). Tables go to results/, charts to figures/.
cd "$(dirname "$0")"
for q in q1_part1_which_exercise q1_part2_exercise_vs_factors q2_generations; do
  python3 -W ignore $q.py | tee results/log_$q.txt
done
