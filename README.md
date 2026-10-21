# MedVAE Transfer for Spinal Geometry

Code and saved results for the MSc individual project report *MedVAE Transfer for Spinal Geometry*. The report source is available as a read-only Overleaf project: [MedVAE Transfer for Spinal Geometry](https://www.overleaf.com/read/nhmhbtcygvvq#ceaef9).

The project tests whether the pretrained 2D encoders of [MedVAE](https://github.com/StanfordMIMI/MedVAE) retain enough information to predict spinal geometry: three Cobb angles and 68 vertebral-corner landmarks on the 609 radiographs of the BoostNet/AASCE dataset. The encoders stay frozen except in the finetuning experiments of Chapter 4. The experiments cover:

- how to present tall radiographs to the encoder (Chapter 4);
- where reconstruction error occurs, and how prediction error changes across MedVAE configurations (Chapter 6);
- MedVAE's stage-2 projection and channel averaging (Chapter 5);
- a comparison with DINOv2, contrast enhancement and augmentation (Sections 7.1, 7.2 and 8.1);
- diffusion models that generate radiographs as extra training data (Section 7.3).

The report presents and interprets the results. This repository holds the code and the saved outputs behind its tables and figures.

MedVAE has four 2D configurations, written `4_3`, `4_1`, `8_4` and `8_1` in the report. The first number is the downsampling factor along each image axis and the second is the number of latent channels. The code uses MedVAE's model names: `medvae_4_3_2d`, `medvae_4_1_2d`, `medvae_8_4_2d` and `medvae_8_1_2d`.

## Repository contents

| Path | Contents |
|---|---|
| `*.py` (top level) | Scripts for data preparation, latent extraction, training, analysis and generation. Most scripts describe their inputs, outputs and usage at the top of the file. |
| `projection_experiments/` | Projection and channel-averaging study (Chapter 5, Appendix B.6) and its tests. |
| `medvae_followup/` | Follow-up checks in Appendix C. |
| `medvae_patches/` | Files this project adds to MedVAE: the extended image loader, the patch dataset and the finetuning configurations. |
| `MedVAE_Report_Completion/` | Scripts and data used to draw the report figures. |
| `report_figures/` | The report figures and the values plotted in them. |
| `report/figures/` | Earlier figures and LaTeX tables written by `make_report_figures.py`. |
| `results/` | Saved results of the main experiments, one folder per run. |
| `outputs/` | Saved results and run records of the projection study and the Appendix C checks. |
| `data/boostnet_labeldata/` | Not included. Place the BoostNet/AASCE radiographs and labels here after downloading them (see [Data preparation](#data-preparation)). |

Cached latents and trained models are not included because of their size; the scripts regenerate them.

## Where to find each result

| Report section | Code | Saved results |
|---|---|---|
| 3.1.2, 3.7 Dataset checks: angle columns, landmark order, image-label matching | `verify_angle_semantics.py`, `verify_angle_columns.py`, `verify_landmarks.py`, `verify_landmark_alignment.py`, `verify_corner_ordering.py` | `results/angle_semantics/` |
| 3.5 Landmark error in pixels | `landmark_pixel_audit.py` | `results/landmark_audit/` |
| 4.1 to 4.3 Input strategies: padding, resizing, tiling | `extract_latents.py`, `extract_latents_whole.py`, `extract_latents_tiled.py`, `extract_latents_tiled_pretrained.py`, `merge_split_latents.py`, `run_multiseed_matched.py`, `train_cobb_regressor_matched.py`, `phase_a_bootstrap.py`, `phase_a_breakdowns.py` | `results/multiseed/` |
| 4.2, 4.3.1 Encoder finetuning | `medvae_patches/`, `prepare_spine_bboxes.py`, `evaluate_reconstruction.py`, `evaluate_reconstruction_tiled.py`, `evaluate_whole_image.py` | |
| 4.4 Diffusion features | `extract_diffusion_features.py` | |
| 6.1, 6.2 Reconstruction error by spatial frequency and anatomical region | `freq_loss_spectrum.py`, `freq_loss_endplate.py`, `freq_loss_edge_interior.py` | `results/freq_loss/` |
| 6.4 Cobb and landmark error | `extract_latents_aspect.py`, `train_frontier.py` | `results/frontier/` |
| 6.5, 6.6 Position, orientation and per-curve error | `cobb_from_landmarks.py` | `results/mechanism/` |
| 6.7 Saliency | `xai_verify.py` | `results/xai/` |
| 6.3 Reconstruction fidelity | `recon_fidelity.py` | `results/fidelity/` |
| 6.9 Computational cost | `cost_benchmark.py` | `results/cost/` |
| 5.1 to 5.5, B.6 Projection and channel averaging | `projection_experiments/` | `outputs/projection_study_fresh_v1/` |
| 5.4, C.1 Removing the two corrupt annotations | `medvae_followup/` | `outputs/followup_annotation_v1/` |
| 8.1 DINOv2 comparison | `extract_latents_dino.py` | `results/dino/`, `results/dino_pca/` (equal storage), `results/dino_aug/` (augmented) |
| 7.1 Contrast enhancement | `extract_latents_multichannel.py`, `compare_runs.py` | `results/clahe/`, `results/clahe_rep/` |
| 7.2 Augmentation presets | `augment_and_extract.py` | `results/aug_cons/` (conservative), `results/aug_mod/` (moderate), `results/aug/` (aggressive) |
| 7.2.4, C.3 Augmentation after projection | `medvae_followup/` | `outputs/followup_augmentation_v1/` |
| 7.3.1, B.1 Cobb-conditioned generation | `extract_latents_vindr.py`, `extract_latents_vindr_cropped.py`, `extract_latents_boostnet_512.py`, `extract_latents_boostnet_cropped.py`, `train_diffusion.py`, `sample_diffusion.py`, `connector_synth_to_medvae.py`, `investigate_dataset_shift.py`, `investigate_dataset_shift_content_only.py`, `investigate_joint_dataset_shift.py`, `verify_padding_fractions.py` | `results/figures/` |
| 7.3.2 to 7.3.6, B.3 Landmark-conditioned generation | `diffusion_latent.py`, `latent_quality.py` | `results/diffusion5/`, `results/scaling_v3/` |
| B.4 Six-task comparison | `derive_task_labels.py`, `train_frontier.py`, `task_baselines.py` | `results/ladder/`, `results/baselines/` |
| B.4.1 Alternative landmark heads | `landmark_heatmap.py`, `landmark_fullres.py` | `results/landmark_audit_hm/` |
| C.2 Reconstruction at 384 and 768 pixels | `medvae_followup/reconstruction_audit.py` | `outputs/followup_reconstruction_v1/` |

Notes on the saved results:

- The report cites some files by their original location, `outputs/<name>/`. In this repository those folders are under `results/<name>/`; for example, `outputs/ladder/` in the report is `results/ladder/` here.
- `results/frontier/` holds the first run of Chapter 6, with ten seeds per configuration and task. The saliency results use three trained heads instead, and the pixel landmark errors come from an earlier run.
- Folders in `results/` and top-level scripts that are not listed above belong to earlier runs or supporting analyses.

## Running the code

### Setup

Each experiment ran on one NVIDIA GPU, either an A10 (24 GB) or an A100 (40 GB). The recorded software versions are Python 3.10.21, PyTorch 2.14.0 with CUDA 13.0, torchvision 0.29.0, MONAI 1.4.0 and NumPy 1.26.4 (`runtime` in `outputs/projection_study_fresh_v1/manifest.json`).

The scripts run from the root of this repository and expect MedVAE's package folder at `./medvae`, as on the experiment machine. Install MedVAE from its GitHub repository and link its package folder into this one:

```bash
git clone https://github.com/StanfordMIMI/MedVAE.git
git clone https://github.com/tienmintan/MedVAE_Transfer.git
pip install -e ./MedVAE
cd MedVAE_Transfer
ln -s ../MedVAE/medvae medvae
```

`medvae_patches/` holds the files this project adds to MedVAE. Copy them into the MedVAE folder:

```bash
cp -r medvae_patches/. ../MedVAE/
```

This replaces MedVAE's `medvae/utils/loaders.py` with the version used in the experiments, which is MedVAE's original file with two loaders added. `load_2d_finetune_aspect`, used by most extraction and reconstruction scripts, pads each radiograph to a square without changing its proportions, then resizes it to 384 × 384 (report Section 3.3). `load_2d_finetune_patch` returns random 384 × 384 crops at native resolution for patch finetuning. The copy also adds the patch dataset and the finetuning configurations described under [Encoder finetuning](#encoder-finetuning-chapter-4).

MedVAE's model code itself is unchanged: the projection-study records include a hash of MedVAE's model file, which matches the original repository.

Three further packages are needed by particular scripts: `transformers` for DINOv2 (`extract_latents_dino.py`), `lpips` for perceptual distance (`recon_fidelity.py`) and `diffusers` for the Stable Diffusion autoencoder used in Cobb-conditioned generation.

```bash
pip install transformers lpips diffusers
```

Pretrained weights are downloaded from Hugging Face on first use: MedVAE (`stanfordmimi/MedVAE`), DINOv2 (`facebook/dinov2-base`) and the Stable Diffusion autoencoder (`stabilityai/sd-vae-ft-mse`).

These checks run on a CPU without the data or model weights:

```bash
python -m unittest discover -s projection_experiments -p "test_*.py"
python medvae_followup/selftest.py
```

Many top-level scripts also accept `--smoke`, which runs them on small synthetic inputs.

### Data preparation

The BoostNet/AASCE radiographs and annotations are not included in this repository. The dataset has 481 training and 128 test radiographs with their landmark and angle files. After downloading, place it in `data/boostnet_labeldata/` with this layout:

- `data/training/` and `data/test/`: the 481 training and 128 test radiographs, as JPEG files named `sunhl-<...>.jpg`.
- `labels/training/` and `labels/test/`: one landmark file per radiograph, named after the image with `.mat` appended (`<image name>.jpg.mat`), and the three files `angles.csv`, `filenames.csv` and `landmarks.csv`.

The commands below copy the images into MedVAE's data folder, create the training, validation and test split, attach the Cobb angles, and match each landmark row to its image, since the landmark files list rows without image names.

```bash
python prepare_boostnet.py --src ./data/boostnet_labeldata --dst ./medvae/data
python prepare_cobb_labels.py --boostnet-src ./data/boostnet_labeldata \
    --csv-in ./medvae/data/boostnet.csv --csv-out ./medvae/data/boostnet_with_angles.csv
python fix_train_lookup.py --csv ./medvae/data/boostnet_with_angles.csv \
    --landmarks-dir ./data/boostnet_labeldata/labels --out ./landmark_row_lookup_train_fixed.csv
python fix_landmark_lookup.py --split test
```

`prepare_boostnet.py` keeps the 128 test images and splits the 481 training images into 409 training and 72 validation images (seed 42). Before running `fix_landmark_lookup.py`, set the two paths at the top of the file, `LABELDATA` and `PIPELINE_CSV`, to `data/boostnet_labeldata` and `medvae/data/boostnet.csv`.

BoostNet's angle file has no column names. The code calls its three columns `angle_proximal_thoracic`, `angle_main_thoracic` and `angle_thoracolumbar` (abbreviated PT, MT and TL), in file order. Report Section 3.1.2 shows that column 0 is the major curve and that columns 1 and 2 usually lie above and below it, so the report calls them the major, upper and lower curves.

VinDr-SpineXR (version 1.0.0, [PhysioNet](https://physionet.org/content/vindr-spinexr/1.0.0/)) is used only for the Cobb-conditioned generator (Section 7.3.1, Appendix B.1): to pretrain it without labels and to compare image padding between the two datasets. It is not included, because access requires a PhysioNet account and a signed data use agreement. Scripts that read it take the dataset folder as an argument (`--vindr-dir`, `--dicom-dir` or `--vindr-dicom-dir`) and expect PhysioNet's folder structure, with `train_images/` and `test_images/`.

### Encoder finetuning (Chapter 4)

MedVAE's finetuning command reads the location of its data folder from `PROJECT_DIR` in `MedVAE/.env`. Create that file once, then finetune the 4_1 encoder on resized or padded images:

```bash
echo "PROJECT_DIR=$(cd ../MedVAE && pwd)" > ../MedVAE/.env
medvae_finetune experiment=medvae_4x_1c_2d_boostnet          # images resized to 384 x 384
medvae_finetune experiment=medvae_4x_1c_2d_boostnet_aspect   # images padded to a square first
```

Patch finetuning also needs spine bounding boxes in the split file and the patch dataset registered with MedVAE:

```bash
python prepare_spine_bboxes.py --boostnet-src ./data/boostnet_labeldata \
    --csv-in ./medvae/data/boostnet.csv --csv-out ./medvae/data/boostnet.csv
echo "from medvae.dataloaders.patch_dataset import BoostNetPatchDataset" >> ../MedVAE/medvae/dataloaders/__init__.py
medvae_finetune experiment=medvae_4x_1c_2d_boostnet_patch
```

Each run saves its checkpoints under `../MedVAE/medvae/logs/<task name>/runs/`. Scripts that use a finetuned encoder take its checkpoint as an argument: `--ckpt` in `extract_latents_tiled.py` and `extract_latents_whole.py`, and `--finetuned-ckpt` in `extract_latents.py`.

### Main comparison (Chapter 6)

These commands extract padded latents for the four configurations, train the fixed head (the same task-head design for every configuration) with ten seeds per configuration and task, and calculate the position, orientation and per-curve errors of Sections 6.5 and 6.6:

```bash
python extract_latents_aspect.py \
    --csv ./medvae/data/boostnet_with_angles.csv --img-dir ./medvae/data/boostnet \
    --landmarks-dir ./data/boostnet_labeldata/labels \
    --train-lookup ./landmark_row_lookup_train_fixed.csv \
    --test-lookup ./landmark_row_lookup_test_fixed.csv \
    --configs medvae_4_3_2d medvae_4_1_2d medvae_8_4_2d medvae_8_1_2d \
    --out-dir ./latents_aspect
python train_frontier.py --latents-dir ./latents_aspect --out-dir ./outputs/frontier \
    --seeds 0 1 2 3 4 5 6 7 8 9 --save-preds
python cobb_from_landmarks.py --preds-dir ./outputs/frontier --img-dir ./medvae/data/boostnet
```

Adding `--verify` to the first command overlays the landmarks on a few images without extracting anything, to check that labels match images. `train_frontier.py` writes per-image errors, summary tables and the paired comparisons for the equal-storage pair (4_1 and 8_4) and the spatial comparison (4_1 and 8_1). `compare_runs.py` compares two runs image by image, for example augmented against unaugmented training.

The saliency analysis (Section 6.7) uses three trained Cobb heads, and the saved pixel landmark errors (Section 3.5) also come from an earlier run. To repeat both, train with three seeds and save the models:

```bash
python train_frontier.py --latents-dir ./latents_aspect --out-dir ./outputs/frontier_3seed \
    --seeds 0 1 2 --save-preds --save-models
python xai_verify.py --models-dir ./outputs/frontier_3seed --latents-dir ./latents_aspect --out-dir ./outputs/xai
python landmark_pixel_audit.py --preds ./outputs/frontier_3seed/preds_medvae_4_1_2d.npz \
    --img-dir ./medvae/data/boostnet --out-dir ./outputs/landmark_audit
```

### Projection study (Chapter 5)

The study extracts raw and projected latents together from the images, using the earlier latents only for image names and labels. It then trains 240 heads (12 latent sets, two tasks, seeds 0 to 9) and computes the 38 planned comparisons. Training uses the copy of `train_frontier.py` kept in `projection_experiments/references/`, and the analysis first checks that the latents and scripts match those recorded when the study was run.

```bash
python projection_experiments/prepare_fresh_projection_study.py \
    --label-cache-dir ./latents_aspect --split-csv ./medvae/data/boostnet_with_angles.csv \
    --img-dir ./medvae/data/boostnet --out-dir ./outputs/projection_study_fresh
python projection_experiments/run_projection_study.py --study-dir ./outputs/projection_study_fresh --phase all
python projection_experiments/analyse_projection_study.py --study-dir ./outputs/projection_study_fresh
python projection_experiments/analyse_projection_geometry.py --study-dir ./outputs/projection_study_fresh \
    --img-dir ./medvae/data/boostnet
```

The output folder must be new and empty. The reported study is in `outputs/projection_study_fresh_v1/`.

### Follow-up checks (Appendix C)

These checks build on the projection study. Each first runs a short pilot to estimate the run time, and the scripts refuse the full run until the pilot has finished.

```bash
# Removing the two corrupt annotations (Section 5.4, Appendix C.1)
python medvae_followup/prepare_annotation_study.py --study-dir ./outputs/projection_study_fresh \
    --out-dir ./outputs/followup_annotation --img-dir ./medvae/data/boostnet
python medvae_followup/run_heads.py --out-dir ./outputs/followup_annotation --pilot
python medvae_followup/run_heads.py --out-dir ./outputs/followup_annotation
python medvae_followup/analyse_heads.py --out-dir ./outputs/followup_annotation

# Augmentation of projected, channel-averaged 4_3 latents (Section 7.2.4, Appendix C.3)
python medvae_followup/prepare_augmentation.py --annotation-dir ./outputs/followup_annotation \
    --out-dir ./outputs/followup_augmentation --img-dir ./medvae/data/boostnet --phase pilot
python medvae_followup/prepare_augmentation.py --annotation-dir ./outputs/followup_annotation \
    --out-dir ./outputs/followup_augmentation --img-dir ./medvae/data/boostnet --phase run
python medvae_followup/run_heads.py --out-dir ./outputs/followup_augmentation --pilot
python medvae_followup/run_heads.py --out-dir ./outputs/followup_augmentation
python medvae_followup/analyse_heads.py --out-dir ./outputs/followup_augmentation

# Reconstruction at 384 and 768 pixels (Appendix C.2)
python medvae_followup/reconstruction_audit.py --phase prepare --out-dir ./outputs/followup_reconstruction \
    --study-dir ./outputs/projection_study_fresh --img-dir ./medvae/data/boostnet
python medvae_followup/reconstruction_audit.py --phase pilot --out-dir ./outputs/followup_reconstruction
python medvae_followup/reconstruction_audit.py --phase run --out-dir ./outputs/followup_reconstruction
python medvae_followup/reconstruction_audit.py --phase analyse --out-dir ./outputs/followup_reconstruction
```

The reported runs are in `outputs/followup_annotation_v1/`, `outputs/followup_augmentation_v1/` and `outputs/followup_reconstruction_v1/`.

### Rerunning analyses from the saved results

The projection, annotation-removal and augmentation analyses need the cached latents and per-seed predictions, which are not included, so they cannot be rerun from this repository alone. Their results are in each study's `analysis/` folder, with the projection geometry in `geometry/`. The reconstruction check (Appendix C.2) needs only its saved per-image records, so its tables can be rebuilt on a CPU:

```bash
python medvae_followup/reconstruction_audit.py --phase analyse --out-dir ./outputs/followup_reconstruction_v1
```

`projection_experiments/references/` and `medvae_followup/references/` keep exact copies of the training, geometry and augmentation scripts used in those studies, and the study scripts check them before use. The top-level versions of these scripts run the same code but have revised comments and messages, so they are not exact copies.

### Report figures

```bash
python MedVAE_Report_Completion/scripts/plot_report_csv.py \
    --root MedVAE_Report_Completion/data/project_csv --out-dir ./figures_check
```

This redraws Figures 3.1, 6.2, 7.1, 8.1 and A.2 from saved CSV files, without a GPU. Figures 2.1, 6.1, 7.2, A.1 and A.3 also need the radiographs, saved predictions or generated latents (`report_radiographs.py`, `report_endplate.py` and `report_generated.py` in the same folder).

## Checkpoints

The two trained generators from Section 7.3 are attached to the [v1.0 release](https://github.com/tienmintan/MedVAE_Transfer/releases/tag/v1.0) of this repository. Other trained models are not included; the commands above retrain them.

| File | Generator | How it is loaded |
|---|---|---|
| `landmark_generator.pt` | Landmark-conditioned generator (Sections 7.3.2 to 7.3.6), the run saved in `results/diffusion5/` | `diffusion_latent.py` reads `diffusion.pt` from its `--out-dir` |
| `cobb_generator_cropped.pt` | Cobb-conditioned generator, cropped pipeline (Section 7.3.1, Appendix B.1) | `sample_diffusion.py --checkpoint` |

```bash
mkdir -p outputs/diffusion5
curl -L -o outputs/diffusion5/diffusion.pt \
    https://github.com/tienmintan/MedVAE_Transfer/releases/download/v1.0/landmark_generator.pt
curl -L -o cobb_generator_cropped.pt \
    https://github.com/tienmintan/MedVAE_Transfer/releases/download/v1.0/cobb_generator_cropped.pt
```

## Acknowledgements

This project uses the following code, models and data:

- MedVAE: Varma M, Kumar A, van der Sluijs R, et al. MedVAE: efficient automated interpretation of medical images with large-scale generalizable autoencoders. arXiv:2502.14753, 2025. Code: <https://github.com/StanfordMIMI/MedVAE> (MIT licence). Weights: <https://huggingface.co/stanfordmimi/MedVAE>.
- BoostNet/AASCE: Wu H, Bailey C, Rasoulinejad P, Li S. Automatic landmark estimation for adolescent idiopathic scoliosis assessment using BoostNet. MICCAI 2017, LNCS 10433, pp. 127-135. Challenge page: <https://aasce19.grand-challenge.org/>.
- VinDr-SpineXR: Pham HH, Nguyen Trung H, Nguyen HQ. VinDr-SpineXR: a large annotated medical image dataset for spinal lesions detection and classification from radiographs (version 1.0.0). PhysioNet, 2021. doi:10.13026/q45h-5h59.
- DINOv2: Oquab M, et al. DINOv2: learning robust visual features without supervision. TMLR, 2024.
- Stable Diffusion autoencoder: Rombach R, et al. High-resolution image synthesis with latent diffusion models. CVPR 2022. Weights: `stabilityai/sd-vae-ft-mse`.
