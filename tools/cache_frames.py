"""Cache K evenly spaced RGB frames per video for one CeFA protocol (uint8, pad-to-square, resized).

CeFA frames are already face-cropped with a black background, so no detector is used. Output is a .npz inside
the CeFA folder (derived data stays under the licence): per split, frames (N_video, K, S, S, 3) plus per-video
metadata (path, CeFA label 0 live / 1 fake, group, subject, PAI).

The P5 lists are already one file per test group, so they give the keys train, dev, test_AF, test_CA, test_EA
and test_masks. Protocols 1 to 4 hold one test list that mixes the test ethnicities and the mask videos; such
a split is written out per group under the same names, so every cache has the same layout.

Usage: python cache_frames.py <CeFA root> --protocol p5 --frames 8 --size 112 [--workers 12]
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

PAI = {"2": "print", "3": "photo", "4": "replay"}


def meta(path):
    parts = path.split("/")
    if parts[1] == "Ethnicity":
        return parts[2], parts[3], PAI.get(parts[4].split("_")[-1], "live")
    return "mask", f"{parts[2]}-{parts[3]}", "mask"


def load(args):
    folder, k, size = args
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
    pick = np.linspace(0, len(files) - 1, k).round().astype(int)
    out = np.zeros((k, size, size, 3), np.uint8)
    for j, i in enumerate(pick):
        im = Image.open(os.path.join(folder, files[i])).convert("RGB")
        w, h = im.size
        sq = Image.new("RGB", (max(w, h),) * 2)
        sq.paste(im, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
        out[j] = np.asarray(sq.resize((size, size), Image.BILINEAR))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--protocol", default="p5")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    pdir = a.root / ("protocol_p5" if a.protocol == "p5" else "protocol")
    files = sorted(pdir.glob(f"CASIA_CeFA@{a.protocol}_video_*.txt"))
    if not files:
        raise SystemExit(f"no video lists for protocol {a.protocol!r} in {pdir}")
    data = {}
    with Pool(a.workers) as pool:
        for f in files:
            split = f.stem.split("_video_")[1]
            rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
            folders = [str(a.root / p.replace("CASIA_CeFA/", "", 1)) for p, _ in rows]
            frames = np.stack(pool.map(load, [(d, a.frames, a.size) for d in folders], chunksize=8))
            m = [meta(p) for p, _ in rows]
            cols = {"x": frames,
                    "y": np.array([int(y) for _, y in rows], np.int8),
                    "path": np.array([p for p, _ in rows]),
                    "group": np.array([g for g, _, _ in m]),
                    "subject": np.array([s for _, s, _ in m]),
                    "pai": np.array([p for _, _, p in m])}
            groups = sorted(set(cols["group"]))
            if split == "test" and len(groups) > 1:  # one test list holding several ethnicities and the masks
                for g in groups:
                    sel = cols["group"] == g
                    name = "test_masks" if g == "mask" else f"test_{g}"
                    for k, v in cols.items():
                        data[f"{name}/{k}"] = v[sel]
                    print(name, cols["x"][sel].shape, flush=True)
            else:
                for k, v in cols.items():
                    data[f"{split}/{k}"] = v
                print(split, frames.shape, flush=True)
    out = a.root / f"cache_{a.protocol}_rgb{a.size}_f{a.frames}.npz"
    np.savez(out, **data)
    print("wrote", out)


if __name__ == "__main__":
    main()
