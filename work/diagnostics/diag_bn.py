# Diagnostic: is the supervised-only collapse a BatchNorm running-statistics problem?
import json, os, time
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
nb = json.load(open("mean_teacher_cefa_experiments.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]
g = {"__name__": "__main__"}
for i in [2, 4, 6, 8, 9, 11, 12, 14, 15, 17]:
    exec(compile(cells[i], f"cell{i}", "exec"), g)
exec(compile(cells[24], "cell24", "exec"), g)
g["RESULTS"] = Path(os.environ["DIAG_OUT"]); g["RESULTS"].mkdir(exist_ok=True)
import torch, numpy as np, copy
cap = {}
orig = g["score_split"]
def grab(model, ids):
    cap["m"] = model; return orig(model, ids)
g["score_split"] = grab
for sp, budget in [("2_1", "all"), ("2_2", 400)]:
    seed = 0 if budget == "all" else 1
    rec = g["run_experiment"]("sup", sp, budget, seed, tag="diag")
    m = cap["m"]
    d = g["make_splits"](g["FM"], sp, budget, seed)
    ev = lambda mm: g["fas_metrics"](orig(mm, d["val_idx"]), orig(mm, d["test_idx"]))
    a = ev(m)
    # recompute BN stats on clean (un-augmented) labeled frames
    r = copy.deepcopy(m)
    for mod in r.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats(); mod.momentum = None
    r.train()
    with torch.no_grad():
        lab = d["lab_idx"]; rng = np.random.default_rng(0)
        for _ in range(100):
            ti = torch.as_tensor(rng.choice(lab, 64))
            r(g["prep_eval"](g["Xt"][ti].to(g["dev"])))
    b = ev(r)
    # recompute BN stats on augmented labeled frames (what training saw)
    r2 = copy.deepcopy(m)
    for mod in r2.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats(); mod.momentum = None
    r2.train()
    with torch.no_grad():
        for _ in range(100):
            ti = torch.as_tensor(rng.choice(lab, 64))
            r2(g["augment"](g["Xt"][ti].to(g["dev"]), g["CFG"]))
    c = ev(r2)
    vs = orig(m, d["val_idx"]).score.values
    print(f"{sp}/{budget}/s{seed}: eval-mode ACER {100*a['acer']:.2f} valEER {100*a['val_eer']:.2f} AUC {a['auc']:.3f} | "
          f"BN recal clean ACER {100*b['acer']:.2f} valEER {100*b['val_eer']:.2f} AUC {b['auc']:.3f} | "
          f"BN recal aug ACER {100*c['acer']:.2f} valEER {100*c['val_eer']:.2f} AUC {c['auc']:.3f} | "
          f"val score quantiles {np.quantile(vs,[0,.25,.5,.75,1]).round(4).tolist()}", flush=True)
