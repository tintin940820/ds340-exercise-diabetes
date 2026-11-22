# last step: compare the results you just made with the numbers written in our paper.
# reads the files in results/ and data/splits/ and prints each number next to the paper's.
# stops with an error if any number is off by more than 0.001.

import pandas as pd
import tools

bad = []


def check(name, ours, paper):
    same = abs(ours - paper) <= 0.001 if isinstance(paper, float) else ours == paper
    if not same:
        bad.append(name)
    o = f"{ours:.3f}" if isinstance(paper, float) else f"{ours:,}"
    p = f"{paper:.3f}" if isinstance(paper, float) else f"{paper:,}"
    print(f"  {name:52s} ours {o:>8s}   paper {p:>8s}   {'same' if same else 'DIFFERENT'}")


def auc(file, model, part, col="model"):
    d = pd.read_csv(f"results/{file}")
    return d[(d[col] == model) & (d.set == part)].auc.iloc[0]


print("Table III: size of each set")
for q, sizes in {"q1": [177576, 25368, 50736], "q2": [53016, 7574, 15148]}.items():
    for part, n in zip(tools.PARTS, sizes):
        check(f"{q} {part} rows", len(pd.read_csv(f"{tools.SPLIT_DIR}/{q}_{part}.csv.gz")), n)

print("\nTable IV: validation and test AUC of the picked model")
check("RQ1a activity + all factors, validation", auc("q1_part1_model_check.csv", "full", "valid"), 0.759)
check("RQ1a activity + all factors, test", auc("q1_part1_model_check.csv", "full", "test"), 0.752)
check("RQ1b factors + demographics, validation",
      auc("q1_part2_model_check.csv", "factors + demographics", "valid"), 0.778)
check("RQ1b factors + demographics, test",
      auc("q1_part2_model_check.csv", "factors + demographics", "test"), 0.782)
check("RQ2 age, sex + obesity, validation", auc("q2_model_check.csv", "+ obesity", "valid"), 0.645)
check("RQ2 age, sex + obesity, test", auc("q2_model_check.csv", "+ obesity", "test"), 0.692)

print("\nTable II: test AUC of each feature set (and protective features kept)")
nov = pd.read_csv("results/novelty_comparison.csv")
table2 = {"All 21 (paper)": ("All 21", 6, 0.817, 0.825, 0.822),
          "Gumbel top 10 (paper)": ("Gumbel top 10", 0, 0.813, 0.821, 0.817),
          "Two-sided Gumbel top 10 (ours)": ("Two-sided top 10", 2, 0.813, 0.821, 0.817),
          "Mutual information top 10": ("Mutual information", 1, 0.814, 0.822, 0.818),
          "L1 logistic top 10": ("L1 logistic", 2, 0.816, 0.824, 0.821),
          "Two-sided top 10, yes/no exercise -> exercise detail (ours)":
              ("Two-sided + exercise detail", 4, 0.813, 0.821, 0.817)}
for features, (label, kept, *aucs) in table2.items():
    rows = nov[nov.features == features]
    check(f"{label}, protective kept", int(rows.protective_kept.iloc[0]), kept)
    for model, paper in zip(["LR", "GB", "XGB"], aucs):
        check(f"{label}, {model}", rows[rows.model == model].auc.iloc[0], paper)

print("\nSection III.E: F1 of the paper's top 10 and ours")
for features, label, f1s in [("Gumbel top 10 (paper)", "Gumbel top 10", (0.803, 0.811, 0.812)),
                             ("Two-sided Gumbel top 10 (ours)", "Two-sided top 10", (0.798, 0.816, 0.812))]:
    rows = nov[nov.features == features]
    for model, paper in zip(["LR", "GB", "XGB"], f1s):
        check(f"{label} F1, {model}", rows[rows.model == model].f1.iloc[0], paper)

print("\nSection IV.D: parent paper 1 on the same sets (test AUC)")
rep = pd.read_csv("results/parent_paper_1_reproduction.csv")
for model, paper in [("LR", 0.817), ("RF", 0.806), ("GB", 0.825), ("XGB", 0.822)]:
    check(f"all 21 features, {model}", rep[(rep.features == "All") & (rep.model == model)].auc.iloc[0], paper)

# the paper's top 10 (Aich et al.), in order
paper_top10 = ["GenHlth", "HighBP", "DiffWalk", "HighChol", "BMI",
               "HeartDiseaseorAttack", "Age", "PhysHlth", "Stroke", "CholCheck"]
ours_top10 = pd.read_csv("results/parent_paper_1_gumbel_scores.csv").feature.head(10).tolist()
print("\nGumbel top 10, ours:  ", ", ".join(ours_top10))
print("Gumbel top 10, paper: ", ", ".join(paper_top10))
if ours_top10 != paper_top10:
    bad.append("Gumbel top 10 list")
print("  same list, same order" if ours_top10 == paper_top10 else "  DIFFERENT")

print()
if bad:
    raise SystemExit("these do not match the paper: " + ", ".join(bad))
print("every number above matches the paper")
