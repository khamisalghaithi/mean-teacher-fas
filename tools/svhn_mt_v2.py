"""E0 v2: SVHN check of Mean Teacher with the paper's training recipe (Appendix B.1.2), 10,000 steps.

Changes from the v1 run in the notebook (Step 8):
  * the teacher (and the Pi model's second pass) runs in training mode, so it gets dropout and uses
    its own batch mean, as in the paper; v1 ran the teacher in eval mode
  * learning rate and consistency weight both follow the paper's sigmoid ramp-up
  * Adam beta2 is 0.99 during the ramp-up and 0.999 after (paper Table 7)
  * a sigmoid ramp-down of the learning rate (to 0) and of Adam beta1 (to 0.5) over the last part of
    training, taken from the paper's CIFAR-10 setting, because the schedule is 18x shorter
  * the labeled/unlabeled split of each batch and the consistency weight are chosen on a validation
    hold-out (10% of the training set), never on the test set
  * both teacher and student error are recorded

Usage:
  python tools/svhn_mt_v2.py tune  --mode mt --seed 0 --lam 100 --n_lab 1 --alpha_late 0.999
  python tools/svhn_mt_v2.py final --mode mt --seed 0 --lam 100 --n_lab 1 --alpha_late 0.999
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1] / "work"
OUT = ROOT / "results_svhn_v2"
OUT.mkdir(exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class MeanOnlyBN(nn.Module):
    """Batch normalization that subtracts the batch mean only (no variance scaling), plus a learned bias."""
    def __init__(self, c, momentum=0.1):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(c))
        self.bias = nn.Parameter(torch.zeros(c)); self.momentum = momentum
        self.track = True
    def forward(self, x):
        if self.training:
            m = x.mean(dim=(0, 2, 3))
            if self.track:
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(m.detach() * self.momentum)
        else:
            m = self.running_mean
        return x - m[None, :, None, None] + self.bias[None, :, None, None]


def _conv(cin, cout, k, pad):
    conv = nn.utils.parametrizations.weight_norm(nn.Conv2d(cin, cout, k, padding=pad))
    return nn.Sequential(conv, MeanOnlyBN(cout), nn.LeakyReLU(0.1))


class ConvNet13(nn.Module):
    """The paper's 13-layer ConvNet (Table 6); same definition as the notebook."""
    def __init__(self, n_classes=10):
        super().__init__()
        L = [_conv(3, 128, 3, 1), _conv(128, 128, 3, 1), _conv(128, 128, 3, 1), nn.MaxPool2d(2), nn.Dropout(0.5),
             _conv(128, 256, 3, 1), _conv(256, 256, 3, 1), _conv(256, 256, 3, 1), nn.MaxPool2d(2), nn.Dropout(0.5),
             _conv(256, 512, 3, 0), _conv(512, 256, 1, 0), _conv(256, 128, 1, 0),
             nn.AdaptiveAvgPool2d(1), nn.Flatten(),
             nn.utils.parametrizations.weight_norm(nn.Linear(128, n_classes))]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        return self.net(x)


def load_svhn():
    from torchvision.datasets import SVHN
    tr = SVHN(str(ROOT / "svhn"), split="train", download=False)
    te = SVHN(str(ROOT / "svhn"), split="test", download=False)
    return (np.transpose(tr.data, (0, 2, 3, 1)).copy(), np.asarray(tr.labels),
            np.transpose(te.data, (0, 2, 3, 1)).copy(), np.asarray(te.labels))


def prep(x, mean, std):
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1); s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x.permute(0, 3, 1, 2).float() / 255.0 - m) / s


def augment(x, mean, std, px=2, sigma=0.15):
    x = prep(x, mean, std)
    B, _, H, W = x.shape
    t = px * 2.0 / W
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1.0; theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = (torch.rand(B, device=x.device) * 2 - 1) * t
    theta[:, 1, 2] = (torch.rand(B, device=x.device) * 2 - 1) * t
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return x + sigma * torch.randn_like(x)


def rampup(step, n):
    x = min(step / max(n, 1), 1.0)
    return math.exp(-5.0 * (1.0 - x) ** 2)


def rampdown(step, total, n):
    """Paper's ramp-down 1 - exp(-12.5 x^2), x going 1 -> 0 over the last n steps."""
    if step < total - n:
        return 1.0
    x = (total - step) / n
    return 1.0 - math.exp(-12.5 * x ** 2)


def set_track(model, on):
    for m in model.modules():
        if isinstance(m, MeanOnlyBN):
            m.track = on


@torch.no_grad()
def ema_update(teacher, student, alpha, step):
    a = min(1 - 1 / (step + 1), alpha)
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(a).add_(ps.detach(), alpha=1 - a)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)          # running means are used at evaluation time only


@torch.no_grad()
def error(model, X, y, mean, std, bs=500):
    model.eval(); wrong = 0
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        wrong += (model(prep(xb, mean, std)).argmax(1).cpu().numpy() != y[i:i + bs]).sum()
    return 100.0 * wrong / len(X)


def run(phase, mode, seed, lam, n_lab, alpha_late, steps=10000, n_labels=1000, batch=100,
        lr=0.003, ramp_frac=40000 / 180000, down_frac=0.25):
    tag = f"svhn2_{phase}_{mode}_s{seed}" + ("" if mode == "sup" else f"_lam{lam:g}_nl{n_lab}_a{alpha_late:g}") \
          + ("" if steps == 10000 else f"_steps{steps}")
    out = OUT / f"{tag}.json"
    if out.exists():
        print("skip (done):", tag); return json.load(open(out))
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    Xtr, ytr, Xte, yte = load_svhn()
    rng = np.random.default_rng(seed)
    if phase == "tune":
        # 10% of the training set is held out with labels for choosing settings (paper Appendix B.3).
        perm = np.random.default_rng(1234).permutation(len(Xtr))
        n_val = len(Xtr) // 10
        val_idx, pool = perm[:n_val], np.sort(perm[n_val:])
        Xev, yev = Xtr[val_idx], ytr[val_idx]
    else:
        pool = np.arange(len(Xtr)); Xev, yev = Xte, yte
    mean = (Xtr[pool].reshape(-1, 3).mean(0) / 255).tolist(); std = (Xtr[pool].reshape(-1, 3).std(0) / 255).tolist()
    lab = np.concatenate([rng.choice(pool[ytr[pool] == c], n_labels // 10, replace=False) for c in range(10)])
    unl = pool                                       # the paper uses all training images as unlabeled

    Xt, yt = torch.from_numpy(Xtr), torch.as_tensor(ytr).long()
    student = ConvNet13().to(dev)
    teacher = None
    if mode == "mt":
        import copy
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad_(False)
        set_track(teacher, False)
    opt = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)
    ramp_steps = int(ramp_frac * steps); down_steps = int(down_frac * steps)
    t0 = time.time()
    lab_order = rng.permutation(lab); lp = 0
    unl_order = rng.permutation(unl); up = 0
    for step in range(steps):
        r_up, r_down = rampup(step, ramp_steps), rampdown(step, steps, down_steps)
        for g in opt.param_groups:
            g["lr"] = lr * r_up * r_down
            g["betas"] = (0.5 + 0.4 * r_down, 0.99 if step < ramp_steps else 0.999)
        # epoch-style sampling: labeled and unlabeled images are each shuffled and reused, as in the paper
        nl = batch if mode == "sup" else n_lab
        nu = batch - nl
        il = []
        while len(il) < nl:
            take = min(nl - len(il), len(lab_order) - lp); il.extend(lab_order[lp:lp + take]); lp += take
            if lp == len(lab_order): lab_order = rng.permutation(lab); lp = 0
        iu = []
        while len(iu) < nu:
            take = min(nu - len(iu), len(unl_order) - up); iu.extend(unl_order[up:up + take]); up += take
            if up == len(unl_order): unl_order = rng.permutation(unl); up = 0
        idx = torch.as_tensor(np.array(il + iu, dtype=np.int64))
        xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
        student.train()
        logits_s = student(augment(xb, mean, std))
        loss = F.cross_entropy(logits_s[:nl], yb[:nl])
        cons = torch.zeros((), device=dev)
        if mode != "sup":
            xt = augment(xb, mean, std)
            if mode == "pi":
                logits_t = student(xt)                      # gradients through both passes, as the paper's Pi model
            else:
                teacher.train()                             # dropout and batch mean, as in the paper
                with torch.no_grad():
                    logits_t = teacher(xt)
            cons = F.mse_loss(F.softmax(logits_s, 1), F.softmax(logits_t, 1))
            loss = loss + lam * r_up * cons
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if mode == "mt":
            ema_update(teacher, student, 0.99 if step < ramp_steps else alpha_late, step)
        if step % 1000 == 0 or step == steps - 1:
            print(f"{tag} step {step} loss {loss.item():.4f} cons {cons.item():.5f} lr {opt.param_groups[0]['lr']:.5f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    rec = dict(exp="E0v2", phase=phase, mode=mode, seed=seed, steps=steps, n_labels=n_labels, batch=batch,
               n_lab_per_batch=(batch if mode == "sup" else n_lab), lam_c=(None if mode == "sup" else lam),
               alpha_late=(alpha_late if mode == "mt" else None), lr=lr, ramp_frac=ramp_frac, down_frac=down_frac,
               eval_split=("validation (10% of train)" if phase == "tune" else "test"),
               student_error_pct=float(error(student, Xev, yev, mean, std)),
               seconds=round(time.time() - t0, 1), device=str(dev), torch=torch.__version__)
    rec["teacher_error_pct"] = float(error(teacher, Xev, yev, mean, std)) if mode == "mt" else None
    rec["error_pct"] = rec["teacher_error_pct"] if mode == "mt" else rec["student_error_pct"]
    json.dump(rec, open(out, "w"), indent=1)
    print(tag, "error %.2f%%" % rec["error_pct"], "| student %.2f%%" % rec["student_error_pct"], "in", rec["seconds"], "s")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "final"])
    ap.add_argument("--mode", choices=["sup", "pi", "mt"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--n_lab", type=int, default=1)
    ap.add_argument("--alpha_late", type=float, default=0.999)
    ap.add_argument("--steps", type=int, default=10000)
    a = ap.parse_args()
    run(a.phase, a.mode, a.seed, a.lam, a.n_lab, a.alpha_late, steps=a.steps)
