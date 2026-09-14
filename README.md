# Cloud Base Height Estimation

This repository contains a PyTorch regression pipeline for estimating cloud-base height from sky images. It supports training and inference with configurable convolutional and transformer-based backbones, validation, checkpointing, and experiment tracking with Weights & Biases (W&B).

## Repository Structure

The repository includes both the main training/inference pipeline and additional analysis folders used to inspect prediction quality over time.

```text
.
├── archs/                          # Model architectures
├── baseline.yml                   # Main training configuration
├── checkpoints/                   # Training checkpoint metadata and resume files
├── conjunto_datos.py               # Dataset builder from raw source data
├── dataset/                       # Dataset implementation
├── datos/                         # Image-to-label split files (.txt)
├── ejecutar.txt                   # Local execution notes / command shortcuts
├── evolucion_predicciones/        # Temporal analysis of predictions over time
├── graphs_results/                # Saved plots and comparison visualizations
├── imagenes_train/                # Training images
├── imagenes_val/                  # Validation images
├── imagenes_test/                 # Test images
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── results/                       # Trained models and inference outputs
├── run.py                         # Main training and inference entry point
├── scripts/                       # Training, validation, and testing scripts
├── tea_debug.log                  # Local debug log
├── tratamiento_datos.ipynb        # Data-processing notebook
├── utils/                         # Augmentations, losses, and utilities
├── venv_ALTURA/                   # Local virtual environment
├── wandb/                         # Local W&B artifacts and run metadata
└── .git/                          # Git repository metadata
```

## Prediction Evolution and Analysis

The folder `evolucion_predicciones/` is not part of the main training loop. It is a post-processing and analysis workflow used to study how model predictions evolve over time for a given week of data. The notebooks `evolucion_lindenberg.ipynb` and `evolucion_valladolid.ipynb` load the detailed prediction outputs, compare them against reference measurements, and visualize temporal trends, error evolution, and event-by-event behaviour.

This workflow typically uses:

- `conjunto_datos.py` to build weekly image-to-ceilometer match files from the raw observation data.
- `archivos_nc/` for the source NetCDF files used during the analysis.
- `semana_lindenberg/` and `semana_valladolid/` for the generated results for each site.
- `graphs_results/` to store final plots and comparison figures produced from the notebooks.

In short, `evolucion_predicciones/` complements the main model pipeline by helping assess whether the model is stable and accurate over extended periods, not just on isolated samples.

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

The baseline configuration enables W&B with `wandb.use: True`. Authenticate with your own API key before training:

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

Do not commit API keys to the repository. If W&B is not required, set `wandb.use: False` in `baseline.yml`.

## Image Folders and Dataset Layout

This project currently uses a single training configuration, `baseline.yml`, and expects the following image folders at the repository root:

```text
imagenes_train/
├── ...

imagenes_val/
├── ...

imagenes_test/
├── ...
```

Each folder contains the images for its corresponding split. The files in `datos/` must match those images and follow this format:

```text
image_filename.jpg;cloud_base_height
```

Example:

```text
C009_20240224_2034.jpg;7957
C009_20240601_1025.jpg;1495
```

The `CloudDataset` loader expects one sample per line, and the cloud-base height is normalized internally by dividing it by `10000` during training.

The current repository uses these split files:

```text
datos/train.txt
datos/train_day.txt
datos/train_night.txt
datos/val.txt
datos/val_day.txt
datos/val_night.txt
datos/test.txt
datos/test_day.txt
datos/test_night.txt
```

Create or replace them with your own data so that every image name listed there exists in the corresponding folder and the numerical height is in metres.

## Dataset Generation with `conjunto_datos.py`

`conjunto_datos.py` is the Python script used to generate the dataset pairs from the raw source data. It reads the image directory and the ceilometer text files, matches each image timestamp to the nearest ceilometer measurement, filters invalid rows, and writes the final `.txt` file used by the model.

This is the script to run when you want to build a dataset from raw camera images and ceilometer readings. It is not a training script; it is a data-preparation utility.

The script is structured around these steps:

1. Read all `.txt` files from the ceilometer source folder.
2. Merge them into a single time-indexed dataframe.
3. Walk the image folder recursively and read each photo filename.
4. Extract the timestamp from the image name.
5. Match each image to the nearest valid ceilometer sample.
6. Keep only valid rows according to the project's filters.
7. Write the resulting `filename;cloud_base_height` entries to the output dataset file.

The default script settings in `conjunto_datos.py` show the idea of the workflow, but you should adapt the source folders and output filename to your own dataset before running it.

## Training

From the repository root, run:

```bash
python run.py --mode train --config baseline.yml
```

Training writes model weights and related outputs to `results/`. Checkpoints used to resume training are stored in `checkpoints/`.

You can select the CUDA device with `--device` and provide an experiment name with `--name`:

```bash
python run.py --mode train --config baseline.yml --device 0 --name my-model
```

## Inference

Inference expects a trained model at `results/<model-name>/<model-name>.pt`.

```bash
python run.py --mode inference --config baseline.yml --name test-model_082337
```

Inference results are written to the relevant results directory, including prediction and error files when enabled in the YAML configuration.

## Configuration

`baseline.yml` controls dataset directories, batch sizes, training epochs, augmentations, optimizer settings, model selection, W&B logging, and output behavior. The `model.pick` value selects one of the models listed under `model.models`.

You should keep the YAML values consistent with your dataset folders and split files, especially:

```yaml
train:
  train_dir: '.\imagenes_train'

validation:
  val_dir: '.\imagenes_val'

test:
  test_dir: '.\imagenes_test'
```

## Notes

- Run commands from the repository root so relative paths resolve correctly.
- Ensure image names in every `.txt` file exactly match the corresponding image files.
- Keep W&B credentials outside version control.
- Large datasets, model weights, W&B runs, and generated results are normally better stored outside the Git repository or managed with Git LFS.
