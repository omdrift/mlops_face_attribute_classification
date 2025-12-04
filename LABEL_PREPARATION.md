# Label Preparation Scripts

This directory contains two scripts for preparing annotation labels for training:

## Scripts

### 1. `merge_annotations.py`

Merges multiple annotation CSV files from `data/labels_raw/` into a single CSV.

**Usage:**
```bash
python merge_annotations.py
```

**What it does:**
- Scans `data/labels_raw/` recursively for all CSV files
- Merges them into a single DataFrame
- Removes duplicates based on filename (keeps most recent)
- Outputs to `data/annotations/merged_labels.csv`

**Command-line options:**
```bash
python merge_annotations.py \
  --input-dir data/labels_raw \
  --output data/annotations/merged_labels.csv \
  --key-column filename
```

### 2. `map_annotations.py`

Maps/transforms merged annotations into the final training CSV format.

**Usage:**
```bash
python map_annotations.py
```

**What it does:**
- Loads `data/annotations/merged_labels.csv`
- Cleans filenames (removes artifacts)
- Validates label values
- Filters out invalid entries
- Outputs to `data/annotations/mapped_train.csv`

**Command-line options:**
```bash
python map_annotations.py \
  --input data/annotations/merged_labels.csv \
  --output data/annotations/mapped_train.csv
```

## Workflow

When you receive new annotation CSV files:

```bash
# 1. Place CSV files in data/labels_raw/ (can be organized in subdirectories)
mkdir -p data/labels_raw/batch_2025_01
cp new_annotations/*.csv data/labels_raw/batch_2025_01/

# 2. Merge all CSVs
python merge_annotations.py

# 3. Map to training format
python map_annotations.py

# 4. Verify the output
head -20 data/annotations/mapped_train.csv

# 5. Update DVC pipeline
dvc repro prepare_train
```

## Expected CSV Format

### Input (Raw Annotations)

The raw CSV files in `data/labels_raw/` should contain at minimum:
- `filename`: Image filename (e.g., `s1_00000.png`)
- `beard`: Binary label (0 or 1)
- `mustache`: Binary label (0 or 1) 
- `glasses_binary`: Binary label (0 or 1)
- `hair_color_label`: Multi-class label (0-4)
- `hair_length`: Multi-class label (0-2)

### Output (Mapped Training CSV)

The final `mapped_train.csv` will have these columns in order:
- `filename`: Cleaned image filename
- `beard`: Validated binary label
- `mustache`: Validated binary label
- `glasses_binary`: Validated binary label
- `hair_color_label`: Validated multi-class label (0-4)
- `hair_length`: Validated multi-class label (0-2)

## Validation Rules

The `map_annotations.py` script validates:
- **Binary labels** (beard, mustache, glasses_binary): Must be 0 or 1
- **Hair color** (hair_color_label): Must be 0, 1, 2, 3, or 4
- **Hair length** (hair_length): Must be 0, 1, or 2
- **Filenames**: Must be non-empty after cleaning

Rows with invalid values are automatically filtered out.

## Troubleshooting

### No CSV files found
```
[!] No CSV files found in data/labels_raw
```
**Solution:** Make sure you've placed CSV files in `data/labels_raw/` (can be in subdirectories).

### Missing columns
```
[!] Missing required columns: ['beard', 'mustache']
```
**Solution:** Ensure your raw CSV files have all required columns.

### Invalid label values
```
[!] Found 5 invalid value(s) in 'beard' (not 0 or 1)
```
**Solution:** The script will automatically filter out rows with invalid values. Check your source data if too many rows are being filtered.

## Integration with DVC Pipeline

After running these scripts, the `mapped_train.csv` file is used by:
1. `dvc repro prepare_train` - Builds training dataset
2. `dvc repro train` - Trains the model
3. `dvc repro evaluate` - Evaluates the model

See `docs/CSV_DRIVEN_PIPELINE.md` for more details on the complete pipeline.
