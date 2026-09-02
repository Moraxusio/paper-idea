# MPFADet Work Log

All timestamps are local time on the development machine.

## 2026-09-02 — repository bootstrap and Step 1

### Goal
Start implementation from IDEA section 9, step 1: lock PNG canvas / coordinate mapping by drawing DR boxes back onto spectrograms. Do not generate PEM or train yet.

### Environment
- Host python `/usr/bin/python3` is 3.12.3 and has no pip.
- Used `/home/finnwe/miniconda3/bin/python` (3.14.7).
- Installed: numpy 2.5.2, pillow 12.3.0, matplotlib 3.11.1.

### Dataset facts re-checked
- 4000 samples, each with `.wav` / `.mat` / `_spectrogram.png` / `.DR.txt`.
- PNG is 875x656, 8-bit RGB but R=G=B (grayscale spectrogram).
- Border analysis: no white margin, no dark axis line, no isolated colorbar. Content fills the whole 875x656 canvas (unique grayscale values ~70–78).
- Therefore Step 1 treats the full image as the TF canvas. Crop remains identity unless overlays later prove otherwise.

### Coordinate hypotheses to test
Time: `DateTime ∈ [0, 5e7]` maps to x ∈ [0, W).
Frequency: `Freq ∈ [0, 5e4]` maps to y. Two hypotheses:
- `y_up`: y=0 is 50 kHz (frequency increases upward, common spectrogram)
- `y_down`: y=0 is 0 Hz (frequency increases downward)

Decision rule: mean pixel intensity inside mapped boxes vs outside. Signals in these spectrograms are bright on a darker background, so the correct mapping should have higher inside-box mean.

### Commands / artifacts
- Code: `scripts/step01_align_boxes.py`
- Overlays: `outputs/step01_alignment/`
- Numeric report: `logs/step01_alignment_report.json`
- Run log: `logs/step01_run.log`

### Step 1 result (2026-09-02 01:59:57+08:00)
Command:
```
/home/finnwe/miniconda3/bin/python scripts/step01_align_boxes.py
```
- 4000 PNGs, all 875x656.
- Stats on 400 images / overlays on 24 images.
- Inside-minus-outside contrast: **y_up = 21.821**, y_down = 1.234.
- y_up better on **399/400** images.
- **Decision: y_up**. `x = DateTime/5e7*W`, `y = (1-Freq/5e4)*H`, crop = identity.
- Locked for all later steps. Overlays in `outputs/step01_alignment/`.

## 2026-09-02 — git init and Step 2 (I/Q + PEM)

### Git
- `git init` in `/home/finnwe/project/paper/MPFADet`, branch `main`.
- Root commit `2f9e6eb`: repo skeleton, geometry helpers, Step-1 script and alignment report.

### Step 2a I/Q convention probe (02:08:29+08:00)
Command:
```
/home/finnwe/miniconda3/bin/python scripts/step02_probe_iq.py
```
Swept `swap × conj × fftshift` on 12 wavs. Metric: STFT log-mag vs PNG correlation, plus inside-box contrast.

**Locked:** `swap=False, conj=False, fftshift=False`
- mean corr to PNG = 0.597
- mean box contrast = 57.66
- All other conventions are near zero or negative.
- WAV is 2ch int16, fs=50000, n=250000, I/Q uncorrelated (~0). Channel 0 = I, channel 1 = Q.
- Frequency axis of `np.fft.fft` without fftshift already matches PNG `y_up` after flipping the frequency dimension for image display (`FLIP_FREQ=True`).

Report: `logs/step02_iq_probe.json`

### Step 2b PEM preview (02:11:45+08:00)
Command:
```
/home/finnwe/miniconda3/bin/python scripts/step02_preview_pem.py
```
- 18 strips (2 per class): MAG | PEM | R=IF | G=Coh | B=Residual, boxes overlaid.
- Inside-box channel contrast is consistently positive (IF/Residual ~30–60, Coh ~6–11).
- Settings: n_fft=1024, hop=256, energy gate percentile=55, alpha=6.
- Previews: `outputs/step02_preview/`
- Report: `logs/step02_preview_report.json`

### Step 2c batch PEM
Command:
```
/home/finnwe/miniconda3/bin/python scripts/step02_generate_pem.py --skip-existing
```
Output: `data/processed/phase/{id}.png` (875x656 RGB, same geometry as magnitude PNG).

Result (02:14:26–02:37:40, 1347 s ≈ 22.5 min):
- 4000/4000 PEM files written, 0 failures.
- `--skip-existing` made the JSON `n_ok=3995` because 5 files already existed from a 5-sample dry run; directory count is 4000.
- Mean RGB ≈ (58.18, 10.80, 58.19): Coh channel is sparse (expected), IF/Residual occupy signal regions.

## 2026-09-02 — Step 3 YOLO labels and split (02:37:40)

Command:
```
/home/finnwe/miniconda3/bin/python scripts/step03_make_yolo.py
```
- Split seed=42, stratified by `background_2` / `background_5` (2:1).
- train 3200 (2132/1068), val 400 (267/133), test 400 (267/133).
- Boxes: train 16229, val 2009, test 2006. No missing phase.
- Magnitude: `data/processed/images/{split}/{id}.png` → symlink to raw spectrogram PNG.
- Phase: `data/processed/image/{split}/{id}.png` → symlink to PEM.
- Labels: `data/processed/labels/{split}/{id}.txt` YOLO `cls x y w h` with y_up mapping.
- YAML: `data/processed/data_MP_Multimodel.yaml` (also copied to `configs/`).

Class ids: 0=2FSK 1=4FSK 2=8-Tone 3=16-Tone 4=GMSK 5=FM 6=AM-DSB 7=Morse 8=PSK.

### Next
Install torch and train baseline A (magnitude-only detector) before dual-stream fusion. RTX 4060 Laptop GPU is present; conda python currently has no torch.

## 2026-09-02 — Step 4 env + smoke (02:45–02:53)

### Python / torch
- Base conda python is 3.14.7: **no torch wheels**. Do not use it for training.
- `conda create` first failed on TOS; accepted:
  - `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
  - `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r`
- Created env `mpfadet` at `/home/finnwe/miniconda3/envs/mpfadet` (Python 3.12.14).
- Installed `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`.
- GPU: RTX 4060 Laptop, 8.59 GB, CUDA available.

### Smoke (02:52:55)
```
/home/finnwe/miniconda3/envs/mpfadet/bin/python  # MagPhaseDataset + MPFADet dual=False
```
- batch mag `[2,3,512,512]`, labels 4 and 5 boxes
- params **2.131 M**
- preds P3/P4/P5: `[2,14,128,128] / [2,14,64,64] / [2,14,32,32]`
- loss 12.76 (box 1.29, obj 0.70, cls 2.26, n_pos=27), backward OK, grad_norm 21.18
- **SMOKE_OK** — train loop can start.

Training command (baseline A, mag-only):
```
/home/finnwe/miniconda3/envs/mpfadet/bin/python scripts/train.py --epochs 30 --batch 8 --imgsz 512 --out outputs/train_A
```

### 3-epoch smoke train (02:53:27–02:57)
```
/home/finnwe/miniconda3/envs/mpfadet/bin/python scripts/train.py --epochs 3 --batch 8 --imgsz 512 --out outputs/train_A_smoke
```
| epoch | train box/obj/cls | val box/obj/cls | vloss | time |
|---|---|---|---|---|
| 1 | 0.870 / 0.101 / 2.063 | 0.786 / 0.023 / 2.006 | 8.447 | 95.9s |
| 2 | 0.697 / 0.014 / 1.914 | 0.706 / 0.012 / 1.951 | 8.006 | 87.4s |
| 3 | 0.648 / 0.011 / 1.798 | 0.688 / 0.011 / 1.958 | 7.973 | 82.5s |

Loss is decreasing. Loop is healthy. Full 30-epoch A / dual F not started yet (~40 min/30 ep on this GPU).

## 2026-09-02 — fix previous-stage bugs, lock conda, continue A

User request: (1) all python via conda env (2) inspect/fix last-stage errors (3) continue.

### Bugs found in Step 4 smoke detector
1. **Wrong FPN strides.** MagNet is stem/s2/s3/s4 all stride-2 → feature maps are **4/8/16**, but `self.strides` and the loss used **8/16/32**. Grid assignment was off by 2x.
2. **Every GT assigned to all 3 levels.** Inflated `n_pos` (27 vs 9 boxes) and made box/cls inconsistent across scales.
3. **Naive BCE on obj.** Positives are ~5 cells vs 128²+64²+32² negatives; obj collapsed (0.101 → 0.011) while cls barely moved.
4. **Stretch resize 875x656 → 512x512.** Frequency axis compressed; Morse/GMSK heights became ~1–2 px.
5. **No detection metric.** Only val_loss, cannot tell if boxes are useful.
6. Scripts used `#!/usr/bin/env python3` and could silently run outside `mpfadet`.

### Fixes
- `mpfadet.env.ensure_conda_env()` re-execs every script onto `/home/finnwe/miniconda3/envs/mpfadet/bin/python`. Wrapper: `scripts/run_conda.sh`.
- Strides locked to `(4, 8, 16)`.
- Size-based FPN assign: min(w,h) <24 → P3, <64 → P4, else P5. `n_pos` now equals number of GT boxes.
- Obj BCE `pos_weight = clamp(neg/pos, max=50)`. Box loss weight 5.
- Dataset uses **letterbox** (keep aspect, pad), labels shifted accordingly.
- Val reports **mAP50** (NMS, VOC-style AP). Decode caps 1000 pre-NMS / 300 post-NMS.
- Rechecked with conda python: `LETTERBOX_SMOKE_OK`, n_pos=19 for 19 boxes.

### Continue
30-epoch baseline A (mag-only) with the fixed loss/geometry.

### Baseline A result (14:16:47–~14:51, conda `mpfadet`)
```
/home/finnwe/miniconda3/envs/mpfadet/bin/python scripts/train.py --epochs 30 --batch 8 --imgsz 512 --out outputs/train_A
```
- python confirmed: `/home/finnwe/miniconda3/envs/mpfadet/bin/python`, strides=(4, 8, 16), 2.13M params.
- npos=40.6 / batch of 8 ≈ 5.1 boxes/image (matches dataset).
- **best mAP50 = 0.633** (epoch 29). Epoch 30: 0.630 / vloss 3.814.
- ~68 s/epoch, 30 ep ≈ 34 min. Log: `logs/step04_train_A.log`.
- Train cls collapsed to 0.001 while val cls ~0.20 — mild overfit on classification; boxes still improving slowly.
- This is the **magnitude-only anchor** for later mag+PEM comparison.

Next: dual F (`--dual`), same 30 epochs, compare mAP50 vs 0.633 without dropping box quality.

## 2026-09-02 — dual F (mag+PEM occupancy-gated fusion)

### Smoke (14:53:05)
```
/home/finnwe/miniconda3/envs/mpfadet/bin/python  # MagPhaseDataset dual=True + MPFADet dual=True
```
- mag/phase both `[2,3,512,512]`, dual params **2.877 M**, gates at P3/P4/P5.
- n_pos=9 for 9 boxes. **DUAL_SMOKE_OK**.

### 30-epoch train F (14:53:30–~15:44, conda `mpfadet`)
```
/home/finnwe/miniconda3/envs/mpfadet/bin/python scripts/train.py --dual --epochs 30 --batch 8 --imgsz 512 --out outputs/train_F
```
- python confirmed, dual=True, strides=(4, 8, 16), ~100 s/epoch ≈ 50 min.
- **best mAP50 = 0.713** (epoch 24). Epoch 30: 0.707 / vloss 3.293.
- vs A: **+0.080 mAP50** (0.633 → 0.713), val box 0.209 → 0.190, val cls 0.200 → 0.125.
- Box quality did not drop; classification val cls is clearly better than A.
- Train cls still →0 (overfit on class logits); fusion still helps val mAP.
- Log: `logs/step04_train_F.log`.

### Per-class eval (conda `mpfadet`, `scripts/eval_ckpt.py`, best.pt)

Val (400 images):

| class | A AP50 | F AP50 | Δ | A AP75 | F AP75 |
|---|---|---|---|---|---|
| 2FSK | 0.456 | 0.534 | +0.078 | 0.022 | 0.075 |
| 4FSK | 0.516 | 0.659 | +0.143 | 0.024 | 0.090 |
| 8-Tone | 0.697 | 0.706 | +0.009 | 0.235 | 0.275 |
| 16-Tone | 0.892 | 0.964 | +0.072 | 0.293 | 0.415 |
| GMSK | 0.444 | 0.599 | +0.155 | 0.024 | 0.076 |
| FM | 0.895 | 0.944 | +0.049 | 0.543 | 0.643 |
| AM-DSB | 0.788 | 0.853 | +0.065 | 0.220 | 0.298 |
| Morse | 0.108 | 0.253 | +0.145 | 0.001 | 0.010 |
| PSK | 0.898 | 0.903 | +0.005 | 0.334 | 0.456 |
| **mAP** | **0.633** | **0.713** | **+0.080** | **0.188** | **0.260** |

Test (400 images): A mAP50=0.646 / mAP75=0.205; F mAP50=0.720 / mAP75=0.279. Same pattern: GMSK/4FSK/Morse gain most; PSK already high from mag.

Reports: `logs/step04_eval_{A,F}_{val,test}.json`.

### Reading
- PEM fusion helps **FSK stairs / GMSK / Morse on-off**, which is the intended role of IF+Coh.
- PSK AP50 was already ~0.90 on mag-only; Residual channel is not the main val gain at IoU=0.5.
- **mAP75 is the next bottleneck** (thin Morse/GMSK/FSK boxes). Do not add DGCL until a localization-focused fix is isolated (center-sampling for 1-px-tall boxes, extra P2, or coarse+fine offsets).

## 2026-09-02 — F_loc: in-box multi-positive + DIoU (wrap-up)

User request: wrap previous work, then continue the plan.

### Diagnosis (letterbox 512, first 400 train images)
- Morse height ~1.31 px, 2FSK ~3.1 px, 4FSK ~6.9 px, GMSK p10~1.9.
- At assigned P3 (stride 4), **236/1097 boxes had no cell center inside the GT** (21%). P4/P5 were fine.
- Center-only assignment therefore starved Morse/2FSK of loc/obj supervision.

### Fix (weights-compatible, no architecture change)
- Expand tiny w/h to at least one stride so a cell center can land in-box; fallback to nearest center if still empty.
- Multi-positive along the time axis for box/obj; **cls still only at GT center** (avoid Morse stripes flooding CE).
- GIoU → DIoU (center-distance term for 1-px-tall boxes).
- Train logs mAP75; `--weights` loads F for finetune. Best checkpoint by mAP50+mAP75.

### Train (18:08:33–19:06, conda `mpfadet`, timeout after epoch 29)
```
/home/finnwe/miniconda3/envs/mpfadet/bin/python scripts/train.py --dual --epochs 30 --batch 8 --imgsz 512 --lr 5e-4 --weights outputs/train_F/best.pt --out outputs/train_F_loc
```
- loaded F best, missing=0. npos ~8489.8 / batch (≈1060/image vs ~5 before).
- No epoch 30 / `history.json` (tool timeout). `best.pt` = epoch **21**, `last.pt` = epoch 29.
- Log: `logs/step05_train_F_loc.log`.

### Eval vs F (conda, `scripts/eval_ckpt.py`)

Val:

| class | F AP50 | F_loc AP50 | Δ | F AP75 | F_loc AP75 |
|---|---|---|---|---|---|
| 2FSK | 0.534 | 0.652 | +0.118 | 0.075 | 0.069 |
| 4FSK | 0.659 | 0.752 | +0.094 | 0.090 | 0.102 |
| 8-Tone | 0.706 | 0.789 | +0.082 | 0.275 | 0.315 |
| 16-Tone | 0.964 | 0.840 | **-0.124** | 0.415 | 0.445 |
| GMSK | 0.599 | 0.679 | +0.080 | 0.076 | 0.107 |
| FM | 0.944 | 0.944 | 0.000 | 0.643 | 0.785 |
| AM-DSB | 0.853 | 0.866 | +0.013 | 0.298 | 0.447 |
| Morse | 0.253 | 0.495 | **+0.242** | 0.010 | 0.035 |
| PSK | 0.903 | 0.959 | +0.057 | 0.456 | 0.479 |
| **mAP** | **0.713** | **0.775** | **+0.062** | **0.260** | **0.309** |

Test: F 0.720 / 0.279 → F_loc **0.782 / 0.315**. Epoch 29 last is slightly worse than epoch 21 best (val 0.771 / 0.302).

### Reading
- In-box positives fixed Morse **recall** (AP50). mAP75 on Morse/2FSK/GMSK is still tiny: IoU=0.75 on a 1–3 px height is a frequency-edge problem, not more positives.
- 16-Tone AP50 dropped (fragmented dets / class-agnostic NMS with denser positives). Next isolated fix: **class-wise NMS** + **height-weighted box loss**. Still no P2/DGCL until those are measured.

## 2026-09-02 — isolated loc follow-ups after F_loc wrap-up

### Class-wise NMS (eval only, no retrain)
`batched_nms` on F_loc `best.pt` val. Report: `logs/step06_eval_Floc_classnms_val.json`.

| | class-agnostic | class-wise |
|---|---|---|
| mAP50 | **0.775** | 0.724 |
| mAP75 | **0.309** | 0.283 |
| 16-Tone AP50 | 0.840 | **0.993** |
| FM AP50 | **0.944** | 0.709 |
| AM-DSB AP50 | **0.853** | 0.650 |
| Morse AP50 | 0.495 | 0.494 |

**Do not ship class-wise NMS.** It recovers 16-Tone (wide FM no longer kills overlapping 16-Tone) but leaves extra FPs on FM/AM-DSB. Official F_loc stays class-agnostic. 16-Tone drop vs F is a **training** issue (multi-pos mean DIoU), not eval NMS.

### Height-weighted DIoU (training)
`wgt = clamp(8/bh, 0.5, 8)` on box loss: Morse ~6×, 16-Tone clamped to 0.5 so large boxes are not drowned. Eval NMS remains class-agnostic.

Train F_hw from F_loc best:
```
/home/finnwe/miniconda3/envs/mpfadet/bin/python scripts/train.py --dual --epochs 20 --batch 8 --imgsz 512 --lr 5e-4 --weights outputs/train_F_loc/best.pt --out outputs/train_F_hw
```

