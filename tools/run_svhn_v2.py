"""Driver for E0 v2: tune on the validation hold-out (seed 0, Mean Teacher), then run the finals on the test set.

Stage 1: labeled images per batch in {1, 10, 50}, lambda_c = 100, alpha_late = 0.999.
Stage 2: with the best split, lambda_c in {30, 300} and alpha_late = 0.99.
Selection: lowest validation error of the evaluated model (EMA teacher). The test set is not touched here.
Finals: sup, pi and mt at seeds 0, 1, 2 with the chosen settings; pi uses the same split and lambda_c.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import svhn_mt_v2 as s

tried = []
for nl in [1, 10, 50]:
    tried.append(s.run("tune", "mt", 0, 100.0, nl, 0.999))
best = min(tried, key=lambda r: r["error_pct"])
nl = best["n_lab_per_batch"]
for lam in [30.0, 300.0]:
    tried.append(s.run("tune", "mt", 0, lam, nl, 0.999))
tried.append(s.run("tune", "mt", 0, 100.0, nl, 0.99))
best = min(tried, key=lambda r: r["error_pct"])
choice = dict(n_lab=best["n_lab_per_batch"], lam=best["lam_c"], alpha_late=best["alpha_late"],
              val_error=best["error_pct"],
              tried=[(r["n_lab_per_batch"], r["lam_c"], r["alpha_late"], round(r["error_pct"], 2),
                      round(r["student_error_pct"], 2)) for r in tried])
json.dump(choice, open(s.OUT / "choice.json", "w"), indent=1)
print("CHOSEN on validation:", choice, flush=True)

for seed in [0, 1, 2]:
    for mode in ["sup", "mt", "pi"]:
        s.run("final", mode, seed, choice["lam"], choice["n_lab"], choice["alpha_late"])
print("ALL DONE", flush=True)
