# Emotion Recognition CNN Model

This project implements a Convolutional Neural Network (CNN) for facial emotion recognition using PyTorch.

## Dataset Overview

- **Classes**: 4 emotions (angry, happy, neutral, sad)
- **Image Size**: 48x48 pixels (grayscale)
- **Training Set**: 4,000 images (1,000 per class)
- **Test Set**: 1,000 images (250 per class)

## Project Structure

```
.
├── train/                    # Training data
│   ├── angry/
│   ├── happy/
│   ├── neutral/
│   └── sad/
├── test/                     # Test data
│   ├── angry/
│   ├── happy/
│   ├── neutral/
│   └── sad/
├── train_emotion_cnn.py      # Main training script
├── predict_emotion.py        # Prediction script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

For GPU support (recommended), ensure you have CUDA installed and install the appropriate PyTorch version from https://pytorch.org/

## Usage

### Training the Model

Run the training script:
```bash
python train_emotion_cnn.py
```

**Training Configuration:**
- Batch size: 64
- Learning rate: 0.001 (with adaptive scheduling)
- Number of epochs: 30
- Optimizer: Adam
- Data augmentation: Random horizontal flip, rotation, and translation

**Output files:**
- `best_emotion_model.pth` - Model with best validation accuracy
- `final_emotion_model.pth` - Model from final epoch
- `training_history.png` - Training/validation loss and accuracy plots

### Making Predictions

Use the prediction script to test the trained model:
```bash
python predict_emotion.py
```

This will:
1. Load the best model
2. Make predictions on sample images
3. Generate visualization plots

**Prediction on custom images:**
```python
from predict_emotion import load_model, predict_image

# Load model
model, class_names = load_model('best_emotion_model.pth')

# Predict
predicted_class, confidence, all_probs = predict_image(
    model, 
    'path/to/your/image.jpg', 
    class_names
)

print(f"Emotion: {predicted_class}, Confidence: {confidence:.2f}%")
```

## Model Architecture

The CNN consists of:

**Convolutional Blocks:**
1. Block 1: 2x Conv2D (64 filters) → BatchNorm → ReLU → MaxPool → Dropout
2. Block 2: 2x Conv2D (128 filters) → BatchNorm → ReLU → MaxPool → Dropout
3. Block 3: 2x Conv2D (256 filters) → BatchNorm → ReLU → MaxPool → Dropout

**Fully Connected Layers:**
- Flatten → Linear(9216→512) → BatchNorm → ReLU → Dropout
- Linear(512→256) → BatchNorm → ReLU → Dropout
- Linear(256→4) output layer

*Total Parameters:* ~12.5 million

## Features

- Data augmentation for better generalization
- Batch normalization for stable training
- Dropout for regularization
- Learning rate scheduling
- Per-class accuracy metrics
- Model checkpointing (saves best model)
- Training visualization
- Easy-to-use prediction interface

## Expected Performance

With this architecture and dataset, you can expect:
- Training accuracy: ~95-98%
- Validation accuracy: ~75-85%

Note: The model may show some overfitting due to the limited dataset size. Consider using:
- More aggressive data augmentation
- Additional regularization
- Pre-trained models (transfer learning)
- Larger datasets for production use

## Tips for Better Performance

1. **Use GPU**: Training on GPU is significantly faster
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Experiment with hyperparameters**:
   - Try different learning rates
   - Adjust dropout rates
   - Modify batch size
   - Add more data augmentation

3. **Monitor training**:
   - Watch for overfitting (gap between train/val accuracy)
   - Use early stopping if validation loss stops improving

4. **Data preprocessing**:
   - Ensure images are properly aligned
   - Consider face detection/alignment preprocessing
   - Balance your dataset if classes are imbalanced

## Troubleshooting

**Out of Memory Error:**
- Reduce batch size
- Use smaller model
- Enable gradient accumulation

**Low Accuracy:**
- Train for more epochs
- Adjust learning rate
- Add more data augmentation
- Check data quality

**Overfitting:**
- Increase dropout rates
- Add more data augmentation
- Use weight decay
- Reduce model complexity

## License

This project is for educational purposes. Make sure to respect the license of the original dataset.

## Acknowledgments

Dataset structure follows the standard ImageFolder format compatible with PyTorch's torchvision.datasets.ImageFolder.
