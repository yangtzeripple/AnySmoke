# AnySmoke

**AnySmoke: Segment Any Wildfire Smoke Dataset and Benchmark**

---

## Overview

**AnySmoke** is the official implement for our paper _“AnySmoke: Segment Any Wildfire Smoke Dataset and Benchmark”_. It provides:

- A new wildfire smoke segmentation dataset (**AnySmoke**).
- Evaluation scripts for state-of-the-art segmentation models on this dataset.
- Baseline results and benchmark metrics.

---

## Supported Models

We evaluate the following SOTA segmentation architectures:

- **U-Net**
- **DeepLabV3+**
- **SegFormer**
- **Mask2Former**
- **FoSp** 
- **Trans-BVM**

> The FoSp and Trans-BVM implementations follow their respective official repositories (see References below)
   * **FoSp**: follow instructions at [LujianYao/FoSp](https://github.com/LujianYao/FoSp)
   * **Trans-BVM**: follow instructions at [SiyuanYan1/Transmission-BVM](https://github.com/SiyuanYan1/Transmission-BVM)

---

## Usage

1. **Prepare the AnySmoke dataset**

   * Place images and masks in the prescribed `data/` folder structure.

2. **Train a model**

   ```bash
   python model_name/train.py --model unet --data-root data/AnySmoke --epochs 50
   ```

3. **Evaluate**

   ```bash
   python evaluate.py --model segformer --checkpoint checkpoints/segformer.pth
   ```

4. **Benchmark all models**

   ```bash
   bash scripts/run_benchmarks.sh
   ```

Benchmark results (IoU, F1, etc.) will be saved in `results/`.

---

## Roadmap

* **Post-acceptance refactor**
  Once the paper is accepted, we will refactor this codebase to improve usability, add tutorials, and include Docker support.

---

## References

* [FoSp (official)](https://github.com/LujianYao/FoSp)
* [Trans-BVM (official)](https://github.com/SiyuanYan1/Transmission-BVM)

---

