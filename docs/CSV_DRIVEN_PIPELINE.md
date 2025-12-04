# CSV-Driven Training and Incremental Lot-Based Inference

## Overview

This pipeline supports two key features:
1. **CSV-driven training data**: Training is controlled entirely by `data/annotations/mapped_train.csv`
2. **Incremental lot-based inference**: Batch inference processes lots incrementally without reprocessing existing predictions

## Directory Structure

```
data/
├── raw/
│   ├── s1/           # Lot 1 images
│   │   ├── s1_00000.png
│   │   ├── s1_00001.png
│   │   └── ...
│   ├── s2/           # Lot 2 images
│   │   ├── s2_00000.png
│   │   └── ...
│   └── s3/           # Lot 3 images
│       └── ...
├── annotations/
│   └── mapped_train.csv  # Training annotations
└── processed/
    └── train_data_s1.pt  # Processed training data

outputs/
├── predictions_s1.csv    # Predictions for lot s1
├── predictions_s2.csv    # Predictions for lot s2
└── predictions_s3.csv    # Predictions for lot s3
```

## CSV-Driven Training

### How It Works

Training data is determined **only** by what's listed in `data/annotations/mapped_train.csv`. The CSV should have these columns:

- `filename`: Relative path from `data/raw/` (e.g., `s1/img_001.png`)
- `beard`: Binary label (0 or 1)
- `mustache`: Binary label (0 or 1)
- `glasses_binary`: Binary label (0 or 1)
- `hair_color_label`: Multi-class label (0-4)
- `hair_length`: Multi-class label (0-2)

### Example CSV

```csv
filename,beard,mustache,glasses_binary,hair_color_label,hair_length
s1/s1_00000.png,1,1,1,0,1
s1/s1_00001.png,0,0,1,0,2
s2/s2_00005.png,0,0,0,4,2
```

### Adding New Training Data

To add new labeled images to training:

1. Ensure the images are in `data/raw/` under their lot subdirectory (e.g., `data/raw/s3/`)
2. Add new rows to `data/annotations/mapped_train.csv` with the image paths and labels
3. Run: `dvc repro prepare_train`
4. (Optional) Run: `dvc repro train` to retrain the model
5. (Optional) Run: `dvc repro evaluate` to evaluate the updated model

**No manual copying or moving of images is required!**

### Running Training Data Preparation

```bash
# Prepare training data from CSV
dvc repro prepare_train

# Or run directly
python src/data/make_dataset.py
```

The script will:
- Read `data/annotations/mapped_train.csv`
- Load only the images listed in the CSV
- Apply preprocessing (crop, resize, normalize)
- Save to `data/processed/train_data_s1.pt`

## Incremental Lot-Based Inference

### How It Works

The batch inference script processes images by lot and **only generates predictions for lots that don't already have output files**. This enables true incremental processing.

### Process Flow

1. **Lot Detection**: Finds all directories under `data/raw/` matching pattern `sX` (where X is a number)
2. **Check Existing**: Looks for existing `outputs/predictions_sX.csv` files
3. **Incremental Processing**: Only processes lots without existing predictions
4. **Skip Training Images** (optional): Can skip images already in `mapped_train.csv` to avoid duplicate work

### Running Batch Inference

```bash
# Run incremental inference
dvc repro inference_batches

# Or run directly
python src/inference/batch_inference.py
```

### Example Workflow

**Initial state:**
```
data/raw/
├── s1/  (100 images)
├── s2/  (100 images)
└── s3/  (100 images)

outputs/
└── (empty)
```

**First run:**
```bash
python src/inference/batch_inference.py
```
Creates:
- `outputs/predictions_s1.csv`
- `outputs/predictions_s2.csv`
- `outputs/predictions_s3.csv`

**Add new lot:**
```
data/raw/
├── s1/  (100 images)
├── s2/  (100 images)
├── s3/  (100 images)
└── s4/  (100 images) ← New lot added
```

**Second run:**
```bash
python src/inference/batch_inference.py
```
- ✅ Processes s4 and creates `outputs/predictions_s4.csv`
- ⏭️ Skips s1, s2, s3 (already have predictions)

### Reprocessing a Lot

To reprocess a specific lot, simply delete its prediction file:

```bash
rm outputs/predictions_s2.csv
python src/inference/batch_inference.py
```

Only s2 will be reprocessed.

## Key Features

### For Training (`make_dataset.py`)

✅ **CSV-Driven**: Only processes images listed in the CSV  
✅ **Flexible Lot Selection**: Can mix images from different lots in training  
✅ **Clear Error Reporting**: Reports missing files and processing failures  
✅ **Preserves Format**: Output format compatible with existing training pipeline  

### For Inference (`batch_inference.py`)

✅ **Incremental Processing**: Only processes new lots  
✅ **Automatic Lot Detection**: Finds all `sX` directories under `data/raw/`  
✅ **Per-Lot Outputs**: Separate CSV for each lot  
✅ **Training Set Aware**: Optionally skips images already in training  
✅ **Efficient**: No wasted computation on already-processed lots  

## DVC Pipeline Integration

The changes are fully integrated with the existing DVC pipeline:

```yaml
stages:
  prepare_train:
    deps:
      - data/raw
      - data/annotations/mapped_train.csv  # CSV controls training data
      - src/data/make_dataset.py
    outs:
      - data/processed/train_data_s1.pt

  inference_batches:
    deps:
      - models/best_model.pth
      - data/raw
      - src/inference/batch_inference.py
      - data/annotations/mapped_train.csv  # Used to skip training images
    outs:
      - outputs/  # Directory to hold all prediction CSVs
```

## Testing

Tests are provided for both features:

```bash
# Test CSV-driven dataset building
pytest tests/test_csv_driven_dataset.py -v

# Test incremental inference logic
pytest tests/test_incremental_inference.py -v
```

## Benefits

1. **Scalability**: Easy to add new data without manual file management
2. **Reproducibility**: CSV clearly defines what's in the training set
3. **Efficiency**: Inference only processes new data
4. **Flexibility**: Can selectively reprocess any lot by deleting its CSV
5. **Traceability**: Each lot's predictions are in separate, trackable files
