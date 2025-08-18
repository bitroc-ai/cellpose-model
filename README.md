# Cellpose Model Training

Docker setup for training custom Cellpose models with GPU support.

## Usage

1. **Prepare your data:**
   ```
   data/
   ├── train/
   │   ├── image_001.tif
   │   ├── image_001_masks.tif
   │   ├── image_002.tif
   │   └── image_002_masks.tif
   └── test/
       ├── test_001.tif
       └── test_001_masks.tif
   ```

2. **Build and run:**
   ```bash
   docker build -t cellpose-train .
   docker run --gpus all -v ./data:/data -v ./models:/models cellpose-train
   ```

3. **Custom training parameters:**
   ```bash
   docker run --gpus all -v ./data:/data -v ./models:/models cellpose-train \
     python train_cellpose.py --learning_rate 0.0001 --n_epochs 50
   ```

## Output

- Trained models saved to `./models/`
- Training logs available in container output

## Requirements

- NVIDIA Docker runtime
- CUDA-compatible GPU