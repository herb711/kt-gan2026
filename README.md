# Anonymous Code Repository

This repository contains the implementation for the paper:

> "Robust Personalized Representation Learning for Sparse User Modeling" (under anonymous review)

## Environment
- Python 3.9
- PyTorch >= 1.13
- CUDA 11.7
- pandas, numpy, scikit-learn

## Running

1. **Preprocessing (Optional if data is already prepared)**:
    You can preprocess raw datasets using `data_preprocess.py`. 
    Example for assist2009:
    ```bash
    python data_preprocess.py --dataset_name assist2009
    ```

2. **Training**:
    Run the main training script. You can specify the path to your dataset using `--data_path`.
    
    ```bash
    # Prepare your data file (e.g. train_valid_sequences.csv)
    # Then run:
    python main.py --data_path "/path/to/your/data/train_valid_sequences.csv"
    ```
    
    Or use the provided shell script:
    ```bash
    bash run.sh
    ```

## Main Structure
- `main.py`: Entry point. Handles data loading, training, and evaluation.
- `model.py`: All model architectures
- `data_preprocess.py`: Data preprocessing script for various datasets.


## Tested Environment
We tested the implementation on the following setup, demonstrating minimal resource requirements:

- **GPU**: NVIDIA L20 (Peak Memory Usage: **~1.1 GB**)
- **Training Time**: ~5 seconds per epoch
- **CPU RAM**: < 2 GB

> **Note**: Due to the low memory footprint (<1.2GB), this code can easily be reproduced on standard consumer GPUs (e.g., RTX 30/40 series) or free cloud instances (e.g., Colab T4).