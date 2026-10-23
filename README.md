# Which Exercise Protects Against Diabetes?

DS 340 · Group 27 · Tim Chen, Ying Jiang

**The data and the code are in this same repository.** Data is in `data/`; code is everything else.

## Research questions

- **Q1 part 1.** Which exercise helps the most against diabetes?
- **Q1 part 2.** How does exercise compare with other factors (weight, smoking, drinking, diet)?
- **Q2.** How do Gen Z (2023) and Millennial (2015) young adults aged 18 to 29 compare, and do habits explain the difference?

## Dataset

Source: CDC Behavioral Risk Factor Surveillance System (BRFSS), public and de-identified.

| File | What it is | Rows | Used by |
|---|---|---|---|
| `data/analysis_core.csv.gz` | BRFSS 2015, all adults, cleaned | 253,680 | Q1 |
| `data/generation_core.csv.gz` | BRFSS 2015, 2021 and 2023, ages 18 to 29, cleaned | 139,113 | Q2 |

The raw CDC files are too large for GitHub (over 1 GB in total), so they are not included. Download them from the CDC:

- 2015: https://www.cdc.gov/brfss/annual_data/annual_2015.html (`LLCP2015XPT.zip`)
- 2021: https://www.cdc.gov/brfss/annual_data/annual_2021.html (`LLCP2021XPT.zip`)
- 2023: https://www.cdc.gov/brfss/annual_data/annual_2023.html (`LLCP2023XPT.zip`)

The outcome is `Diabetes` / `diabetes`, where 1 means the person was ever told they have diabetes or prediabetes. Every row has a CDC survey weight (`weight`), stratum (`strata`) and sampling unit (`psu`).

## Code

```
prepare/                          builds the two clean tables from the raw CDC files (already run)
q1_part1_which_exercise.py        Q1 part 1
q1_part2_exercise_vs_factors.py   Q1 part 2
q2_generations.py                 Q2
run_all.sh                        runs the three questions (about 1 minute)
DESIGN.md                         how the code is designed and why
results/                          output tables (.csv) and printouts
figures/                          one chart per question
```

Every analysis script follows the same five steps: load the data with survey weights, make 0/1 columns, take a simple look at percentages, fit a weighted logistic regression, and save the table and chart.

## How to run

```bash
pip install pandas numpy statsmodels matplotlib pyreadstat
bash run_all.sh
```

To rebuild the clean tables from the raw CDC files first:

```bash
cd prepare
python3 01_load_raw.py /path/to/LLCP2015.XPT
python3 02_clean.py
python3 07_load_generations.py /path/to/LLCP2015.XPT /path/to/LLCP2021.XPT /path/to/LLCP2023.XPT
python3 08_clean_generations.py
```
