# AI Model Checkpoints

These files are too large for GitHub and must be downloaded separately.

## Required Models

### 1. SAM2 (Segment Anything Model 2)
```
sam2_hiera_large.pt (~857MB)
```
Download: https://github.com/facebookresearch/segment-anything-2

### 2. GroundingDINO
```
groundingdino_swint_ogc.pth (~662MB)
```
Download: https://github.com/IDEA-Research/GroundingDINO

### 3. RAM++ (Recognize Anything Model)
```
ram_plus_swin_large_14m.pth (~2.9GB)
```
Download: https://github.com/xinyu1205/recognize-anything

## Quick Setup

The rigger will attempt to download models automatically on first run.
If that fails, manually place the files in this `checkpoints/` folder.
