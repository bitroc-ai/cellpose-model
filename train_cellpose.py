#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from cellpose import models, io, train


def main():
    parser = argparse.ArgumentParser(description='Train Cellpose model')
    parser.add_argument('--train_dir', default='/data/train', help='Training data directory')
    parser.add_argument('--test_dir', default='/data/test', help='Test data directory') 
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--model_name', default='cellpose_model', help='Output model name')
    parser.add_argument('--chan', type=int, default=0, help='Channel to segment (0=grayscale, 1=red, 2=green, 3=blue)')
    parser.add_argument('--chan2', type=int, default=0, help='Second channel (0=none, 1=red, 2=green, 3=blue)')
    parser.add_argument('--pretrained_model', default='cpsam', help='Pretrained model to start from')
    
    args = parser.parse_args()
    
    # Verify directories exist
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    
    if not train_dir.exists():
        print(f"Error: Training directory {train_dir} does not exist")
        sys.exit(1)
        
    if not test_dir.exists():
        print(f"Warning: Test directory {test_dir} does not exist, using train dir for validation")
        test_dir = train_dir
    
    # Load images
    print("Loading training images...")
    train_images = io.load_train_test_data(str(train_dir), str(test_dir), mask_filter='_masks')
    
    if len(train_images[0]) == 0:
        print("Error: No training images found. Make sure images and masks follow naming convention:")
        print("  - Images: image_name.tif")
        print("  - Masks: image_name_masks.tif")
        sys.exit(1)
    
    print(f"Found {len(train_images[0])} training images")
    print(f"Found {len(train_images[1])} test images")
    
    # Initialize model
    print(f"Initializing model from {args.pretrained_model}...")
    model = models.CellposeModel(gpu=True, model_type=args.pretrained_model)
    
    # Train model
    print("Starting training...")
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images[0],
        train_labels=train_images[1], 
        test_data=train_images[2],
        test_labels=train_images[3],
        channels=[args.chan, args.chan2],
        normalize=True,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        save_path='/models/'
    )
    
    print(f"Training completed! Model saved to: {model_path}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    if test_losses:
        print(f"Final test loss: {test_losses[-1]:.6f}")


if __name__ == '__main__':
    main()