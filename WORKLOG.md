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
