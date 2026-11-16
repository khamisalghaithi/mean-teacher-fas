# Mean Teacher for face anti-spoofing on CeFA Protocol 2

Course project for COMP6603 at Sultan Qaboos University. It tests whether
Mean Teacher (Tarvainen and Valpola, NeurIPS 2017) helps a small face anti-spoofing model when only a few
training videos are labeled. The student is MobileNetV3-Small (1.52M parameters) on 112x112 RGB face crops.

## What is in the repository

| Path | Content |
|---|---|
| `mean_teacher_cefa_experiments.ipynb` | All code: data checks, splits, models, training, evaluation, tables and figures. |
| `mean_teacher_cefa_experiments_executed.ipynb` | The same notebook after the final run, with its printed output. |
| `tools/cache_frames.py` | Builds the frame caches the notebook reads (5 frames per video, 112 px). |
| `work/results/` | One JSON per run: settings, validation curve and test metrics. These are the numbers in the report. |
| `work/results_v1_no_bn_recal_2026-11-15/` | The first run of the grid, before the BatchNorm fix (see below). Kept for the record. |
| `work/diagnostics/` | The script and output that found the BatchNorm problem. |
| `tools/svhn_mt_v2.py`, `tools/run_svhn_v2.py` | The corrected SVHN check (E0 v2), 10,000 steps (see below). |
| `work/results_svhn_v2/`, `work/svhn_v2.log` | E0 v2 results: tuning runs on a validation hold-out, `choice.json`, and the final test runs. |
| `tools/cefa_extra.py`, `work/cefa_extra*.log` | Runs added after the first grid (see below). |
| `tools/summarise_results.py` | Prints every number the report quotes, from the JSON files. |

## Experiments

- E0: SVHN with 1,000 labels, to check the implementation against the paper (supervised, Pi model, Mean Teacher).
- E1: supervised-only. E3: Mean Teacher. E4: a frozen ResNet-50 teacher fine-tuned on the labeled videos.
- CeFA Protocol 2 (cross-instrument): 2_1 trains on print and tests on replay plus masks, 2_2 the reverse.
  100, 400 or all labeled training videos; 3 seeds (one seed for "all").
- E2: the Pi model on CeFA.
- E5: E3 and E4 with the unseen instrument added to the unlabeled pool. This is my own change to the protocol.

Runs added after the first grid (`tools/cefa_extra.py`, tags in the file names): E3 with the teacher in training
mode as in the paper (`_ttrain`), E5 for E4, seeds 3 and 4 on 2_1, lambda_c 300 and 1000 (`_lam*_recal`), KL
consistency (`_kl`) and E2 (`python tools/cefa_extra.py F`). One SVHN Mean Teacher run of 30,000 steps
(`--steps 30000`) checks how much of the gap to the paper is the schedule.

Metrics are APCER, BPCER and ACER (ISO/IEC 30107-3) at the equal-error threshold of the validation videos.
The test subjects are never used to choose anything.

## The BatchNorm fix

The first run of the grid gave a broken supervised baseline: with all labels it scored 48.11% ACER. Training
adds Gaussian noise to every input, so the BatchNorm running statistics describe noisy frames, while testing
uses clean ones. Every model is now scored after its BatchNorm statistics are recomputed on clean labeled
training frames, without changing any weights. Each JSON keeps both numbers (`acer` after, `acer_raw` before).

## The SVHN check (E0 v2)

The first SVHN check in the notebook (3,000 steps) did not reproduce the paper. The second version
in `tools/svhn_mt_v2.py` runs the teacher in training mode, follows the paper's learning-rate and
consistency ramp-up and its Adam beta2 schedule, adds a learning-rate ramp-down at the end, and runs
10,000 steps. The batch split and the consistency weight were chosen on 10% of the training set held out
for validation. Test error over 3 seeds: supervised-only 12.44 +- 0.41%, Pi model 7.13 +- 0.05%,
Mean Teacher 6.45 +- 0.23% (paper: 12.32, 4.95, 3.95). Rerun with `python tools/run_svhn_v2.py`,
about 6 hours on an Apple M3 Max.

## Data

The CASIA-SURF CeFA dataset is **not** included. Its licence forbids redistribution, so this repository has no
frames, caches or protocol lists. To rerun, request CeFA from its authors, then:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python tools/cache_frames.py "<CeFA root>" --protocol p2.1 --frames 5 --size 112
python tools/cache_frames.py "<CeFA root>" --protocol p2.2 --frames 5 --size 112
CEFA_ROOT="<CeFA root>" jupyter nbconvert --to notebook --execute mean_teacher_cefa_experiments.ipynb
```

The caches are written inside the CeFA folder. SVHN is downloaded by torchvision on first use. Finished runs are
skipped, so delete `work/results/` to start again. The full grid took about eight hours on an Apple M3 Max (MPS backend).
