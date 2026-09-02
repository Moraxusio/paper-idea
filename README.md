# MPFADet

Magnitude–Phase Feature Adaptive Detector for wideband time-frequency object detection.

See `../IDEA_MagPhase_Fusion.md` for the design. Work progress is recorded in `WORKLOG.md` and `logs/`.

All Python scripts must run in conda env `mpfadet`:

```
/home/finnwe/miniconda3/envs/mpfadet/bin/python scripts/train.py
```

Scripts re-exec into that interpreter via `mpfadet.env.ensure_conda_env()`.
