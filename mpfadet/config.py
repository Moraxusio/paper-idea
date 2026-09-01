"""Locked conventions from Step 1 / Step 2 probes."""

from pathlib import Path

RAW_DIR = Path("/home/finnwe/project/paper/few_shot_data")
ROOT = Path("/home/finnwe/project/paper/MPFADet")

IQ_SWAP = False
IQ_CONJ = False
FFTSHIFT = False
FLIP_FREQ = True  # y_up: 0 Hz at bottom of the image

N_FFT = 1024
HOP = 256
GATE_ALPHA = 6.0
GATE_PERCENTILE = 55.0
