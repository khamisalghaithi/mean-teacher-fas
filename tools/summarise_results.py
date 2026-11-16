"""Print every number the Part 2 report quotes from work/results/ (CeFA) and work/results_svhn_v2/ (SVHN)."""
import json, glob, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1] / "work"
rows = [json.load(open(p)) for p in sorted(glob.glob(str(R / "results" / "*.json")))]
rows = [r for r in rows if r.get("dataset") == "cefa"]

def pick(mode, sp, b, variant="hidden", tag="", seeds=(0, 1, 2)):
    out = {r["seed"]: r for r in rows if r["mode"] == mode and r["sub_protocol"] == sp and str(r["budget"]) == str(b)
           and r["variant"] == variant and (r.get("tag") or "") == tag and r["seed"] in seeds}
    return [out[s] for s in seeds if s in out]

def ms(v):
    v = 100 * np.asarray(v, float)
    return f"{v.mean():.2f} +- {v.std(ddof=1):.2f}" if len(v) > 1 else (f"{v[0]:.2f}" if len(v) else "--")

def per(rs, key="acer"):
    return ", ".join(f"{100 * r[key]:.2f}" for r in rs)

print("== ACER, U-hidden, seeds 0-2 (per seed | mean +- sd)")
for mode, tag in [("sup", ""), ("pi", ""), ("mt", ""), ("mt", "ttrain"), ("kd", "")]:
    for sp in ["2_1", "2_2"]:
        for b in [100, 400, "all"]:
            rs = pick(mode, sp, b, tag=tag, seeds=(0,) if b == "all" else (0, 1, 2))
            if rs:
                print(f"{mode:3s} {tag:7s} {sp} {str(b):4s} | {per(rs):28s} | {ms([r['acer'] for r in rs]):16s} "
                      f"| AUC {ms([r['auc'] for r in rs])} | BPCER {ms([r['bpcer'] for r in rs])}")
    for b in [100, 400]:
        a, c = pick(mode, "2_1", b, tag=tag), pick(mode, "2_2", b, tag=tag)
        if len(a) == len(c) == 3:
            print(f"{mode:3s} {tag:7s} mean {b}: {ms([(x['acer'] + y['acer']) / 2 for x, y in zip(a, c)])}")

print("\n== 2_1, seeds 0-4")
for mode in ["sup", "mt", "kd"]:
    for b in [100, 400]:
        rs = pick(mode, "2_1", b, seeds=(0, 1, 2, 3, 4))
        print(f"{mode} {b}: n={len(rs)} | {per(rs)} | {ms([r['acer'] for r in rs])}")
for b in [100, 400]:
    s, m, k = (pick(x, "2_1", b, seeds=(0, 1, 2, 3, 4)) for x in ["sup", "mt", "kd"])
    if len(s) == len(m) == len(k) == 5:
        d1 = [100 * (y["acer"] - x["acer"]) for x, y in zip(s, m)]
        d2 = [100 * (y["acer"] - x["acer"]) for x, y in zip(m, k)]
        print(f"  {b}: MT-sup per seed {[round(d, 2) for d in d1]}; KD-MT {[round(d, 2) for d in d2]}")

print("\n== E3 teacher mode: eval (old) vs train, per seed")
for sp in ["2_1", "2_2"]:
    for b in [100, 400]:
        o, t = pick("mt", sp, b), pick("mt", sp, b, tag="ttrain")
        print(f"{sp} {b}: eval {per(o)} | train {per(t)} | student(train) {per(t, 'acer_student')}")

print("\n== E5 (400 videos): APCER on the missing instrument")
for mode in ["mt", "kd"]:
    for sp, key in [("2_1", "apcer_replay"), ("2_2", "apcer_print")]:
        h, v = pick(mode, sp, 400), pick(mode, sp, 400, variant="visible")
        if h and key not in h[0]:
            key = "apcer_print" if "apcer_print" in h[0] else key
        print(f"{mode} {sp} {key}: hidden {per(h, key)} ({ms([r.get(key, np.nan) for r in h])}) | "
              f"visible {per(v, key)} ({ms([r.get(key, np.nan) for r in v]) if v else '--'}) | "
              f"ACER {ms([r['acer'] for r in h])} -> {ms([r['acer'] for r in v]) if v else '--'}")

print("\n== lambda_c (E3, 2_1, 400, seed 0): validation EER after recalibration")
for tag in ["", "lam300_recal", "lam1000_recal"]:
    for r in pick("mt", "2_1", 400, tag=tag, seeds=(0,)):
        print(f"lam {r['lam_c']}: val EER {100 * r['val_eer']:.2f} | test ACER {100 * r['acer']:.2f}")

print("\n== KL vs MSE (E3, 2_1, 400)")
for tag in ["", "kl"]:
    rs = pick("mt", "2_1", 400, tag=tag)
    print(f"{tag or 'mse'}: {per(rs)} | {ms([r['acer'] for r in rs])} | val EER {per(rs, 'val_eer')}")

print("\n== SVHN")
for p in sorted(glob.glob(str(R / "results_svhn_v2" / "svhn2_final_*.json"))):
    r = json.load(open(p))
    print(Path(p).stem, f"{r['error_pct']:.2f}", f"student {r['student_error_pct']:.2f}", r["steps"])
