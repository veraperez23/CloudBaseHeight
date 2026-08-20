# Cloud Base Height Estimation

This repository contains a PyTorch regression pipeline for estimating cloud-base height from sky images. It supports training and inference with configurable convolutional and transformer-based backbones, validation, checkpointing, and experiment tracking with Weights & Biases (W&B).

## Repository Structure

```text
.
├── archs/                  # Available model architectures
├── config/                 # Additional configuration files
├── dataset/                # Dataset implementation
├── datos/                  # Dataset split files (.txt)
├── imagenes_day/           # Daytime images
├── imagenes_night/         # Night-time images
├── imagenes_todoeldia/     # Images from the full day
├── results/                # Trained models and inference outputs
├── scripts/                # Training, validation, and testing logic
├── utils/                  # Augmentations, losses, and utilities
├── baseline.yml            # Full-day training configuration
├── baseline_day.yml        # Daytime training configuration
├── baseline_night.yml      # Night-time training configuration
├── requirements.txt        # Python dependencies
└── run.py                  # Main training and inference entry point
```

## Requirements

- Windows, Linux, or macOS
- Python 3.10 or newer recommended
- An NVIDIA GPU with CUDA is recommended for training
- A Weights & Biases account and API key when W&B logging is enabled

## Installation

Create and activate a virtual environment from the repository root.

### Windows Command Prompt

```bat
python -m venv venv_ALTURA
venv_ALTURA\Scripts\activate
```

### Windows PowerShell

```powershell
python -m venv venv_ALTURA
.\venv_ALTURA\Scripts\Activate.ps1
```

Install PyTorch with the CUDA 11.8 wheels used by the original setup:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining packages:

```bash
pip install timm wandb fvcore PyYAML tqdm opencv-python matplotlib imageio pillow numpy ipykernel
```

The pinned dependency file is also available for reproducing the recorded environment:

```bash
pip install -r requirements.txt
```

Use either the explicit installation commands above or the requirements file according to your local CUDA/PyTorch setup. If CUDA is not available, install a CPU-compatible PyTorch build from the official PyTorch installation selector and install the remaining packages afterward.

## Weights & Biases Setup

The baseline configurations enable W&B with `wandb.use: True`. Authenticate with your own API key before training:

```bash
wandb login
```

When prompted, paste the API key from your W&B account. Alternatively, set it as an environment variable.

### Windows Command Prompt

```bat
set WANDB_API_KEY=YOUR_WANDB_API_KEY
```

### Windows PowerShell

```powershell
$env:WANDB_API_KEY = "YOUR_WANDB_API_KEY"
```

Do not commit API keys to the repository. If W&B is not required, set `wandb.use: False` in the selected YAML configuration.

## Dataset Preparation

The project does not include a universal dataset. Add your own image dataset and split files before running the code.

The repository contains three baseline configurations, one for each dataset split:

| Configuration | Dataset | Split files |
|---|---|---|
| `baseline_day.yml` | Daytime images | `train_day.txt`, `val_day.txt`, `test_day.txt` |
| `baseline_night.yml` | Night-time images | `train_night.txt`, `val_night.txt`, `test_night.txt` |
| `baseline.yml` | Full-day images | `train.txt`, `val.txt`, `test.txt` |

The expected image layout is:

```text
imagenes_day/
├── train_day/
├── val_day/
└── test_day/

imagenes_night/
├── train_night/
├── val_night/
└── test_night/

imagenes_todoeldia/
├── imagenes train/
├── imagenes_val/
└── imagenes_test/
```

Each split file in `datos/` must contain one sample per line using this format:

```text
image_filename.jpg;cloud_base_height
```

For example:

```text
IMG_0001.jpg;1250.5
IMG_0002.png;980.0
```

The image filename must match a file in the corresponding image directory, and the height must be a numeric value in metres. The dataset loader normalizes this value by dividing it by `10000` during training.

Create or replace the following files for your dataset:

```text
datos/train_day.txt
datos/val_day.txt
datos/test_day.txt
datos/train_night.txt
datos/val_night.txt
datos/test_night.txt
datos/train.txt
datos/val.txt
datos/test.txt
```

Update the selected YAML file so that `train.train_dir`, `validation.val_dir`, and `test.test_dir` point to your image directories. Keep the `.txt` files and image directories consistent.

The current `run.py` uses the daytime split filenames (`train_day.txt`, `val_day.txt`, and `test_day.txt`) internally for all configurations. Therefore, before using `baseline_night.yml` or `baseline.yml` with your own data, update those three paths in `run.py` to the corresponding night-time or full-day files, or adapt the entry point to select the split names from the YAML configuration.

## Training

From the repository root, run:

```bash
python run.py --mode train --config baseline_day.yml
```

For the night-time configuration:

```bash
python run.py --mode train --config baseline_night.yml
```

For the full-day configuration:

```bash
python run.py --mode train --config baseline.yml
```

Training writes model weights and related outputs to `results/`. Checkpoints used to resume training are stored in `checkpoints/`.

You can select the CUDA device with `--device` and provide an experiment name with `--name`:

```bash
python run.py --mode train --config baseline_day.yml --device 0 --name my-day-model
```

## Inference

Inference expects a trained model at `results/<model-name>/<model-name>.pt`.

```bash
python run.py --mode inference --config baseline_day.yml --name test-model_082337
```

For a night-time model:

```bash
python run.py --mode inference --config baseline_night.yml --name test-model_062021
```

For a full-day model:

```bash
python run.py --mode inference --config baseline.yml --name my-full-day-model
```

Inference results are written to the relevant results directory, including prediction and error files when enabled in the YAML configuration.

## Configuration

The YAML files control dataset directories, batch sizes, training epochs, augmentations, optimizer settings, model selection, W&B logging, and output behavior. The `model.pick` value selects one of the models listed under `model.models`.

## Notes

- Run commands from the repository root so relative paths resolve correctly.
- Ensure image names in every `.txt` file exactly match the corresponding image files.
- Keep W&B credentials outside version control.
- Large datasets, model weights, W&B runs, and generated results are normally better stored outside the Git repository or managed with Git LFS.
