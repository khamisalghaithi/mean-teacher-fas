"""Extra CeFA runs after the first grid, using the notebook's own code (cells are executed, not copied).

A  E3 with the EMA teacher in training mode while it makes its targets (dropout on, batch statistics),
   as in the paper and in the SVHN runs. Tag "ttrain". 2_1 and 2_2, 100 and 400 videos, seeds 0-2.
B  E5 for E4: frozen ResNet-50 teacher with the other instrument in the unlabeled pool, 400 videos, seeds 0-2.
C  Seeds 3 and 4 for E1, E3 and E4 on 2_1 at 100 and 400 videos.
D  Consistency weight above the old grid: lambda_c 300 and 1000, E3, 2_1, 400 videos, seed 0 (validation EER only).
E  KL consistency (tau 2) instead of MSE, E3, 2_1, 400 videos, seeds 0-2.
F  E2, the Pi model, on 2_1 and 2_2 at 100 and 400 videos, seeds 0-2 (run with argument F).
Finished runs are skipped. Results go to work/results/ next to the first grid.
"""
import json, os, sys
os.environ["MPLBACKEND"] = "Agg"
nb = json.load(open("mean_teacher_cefa_experiments.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]
g = {"__name__": "__main__"}
for i in [2, 4, 6, 8, 9, 11, 14, 15, 17]:
    exec(compile(cells[i], f"cell{i}", "exec"), g)

old = "                with torch.no_grad(): logits_t = teacher(xt)"
new = ("                with torch.no_grad():\n"
       "                    if cfg.get(\"teacher_train\"): teacher.train()   # dropout and batch statistics, as in the paper\n"
       "                    logits_t = teacher(xt)\n"
       "                    teacher.eval()")
assert cells[24].count(old) == 1
exec(compile(cells[24].replace(old, new), "cell24", "exec"), g)

CFG, run = g["CFG"], g["run_experiment"]
which = sys.argv[1:] or ["A", "B", "C", "D", "E"]

if "A" in which:
    cfg = dict(CFG, teacher_train=True)
    for sp in ["2_1", "2_2"]:
        for b in [100, 400]:
            for s in [0, 1, 2]:
                run("mt", sp, b, s, cfg=cfg, tag="ttrain")
if "B" in which:
    for sp in ["2_1", "2_2"]:
        for s in [0, 1, 2]:
            run("kd", sp, 400, s, variant="visible")
if "C" in which:
    for b in [100, 400]:
        for s in [3, 4]:
            for m in ["sup", "mt", "kd"]:
                run(m, "2_1", b, s)
if "D" in which:
    for lam in [300.0, 1000.0]:
        run("mt", "2_1", 400, 0, cfg=dict(CFG, lam_c=lam), tag=f"lam{int(lam)}_recal")
if "E" in which:
    for s in [0, 1, 2]:
        run("mt", "2_1", 400, s, cfg=dict(CFG, cons_loss="kl"), tag="kl")
if "F" in which:
    for sp in ["2_1", "2_2"]:
        for b in [100, 400]:
            for s in [0, 1, 2]:
                run("pi", sp, b, s)
print("ALL DONE", flush=True)
