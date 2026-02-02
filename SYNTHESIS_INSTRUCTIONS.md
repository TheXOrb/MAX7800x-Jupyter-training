# ECG Model Synthesis for MAX78002

## Prerequisites
✅ Trained model checkpoint: `best_ecg_model_ai8x.pth.tar` (QAT-enabled)  
✅ KAT sample: `sample_ecg_1x128.npy`  
✅ Network architecture YAML: `networks/ecg-net.yaml`

**IMPORTANT:** Your model was trained with QAT (Quantization-Aware Training) enabled in Cell 10.5 of ECG_MIT_BIH.ipynb. This automatically folds BatchNorm layers during training, making the checkpoint ready for synthesis.  

## Step 1: Setup ai8x-synthesis Environment

```bash
# Initialize conda
eval "$(~/anaconda3/bin/conda shell.bash hook)"


# Create environment (Python 3.11.8 recommended for ai8x-synthesis)
conda create -n ai8x-synthesis python=3.11.8 -y
conda activate ai8x-synthesis

# Clone ai8x-synthesis repository
cd ~
git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git
cd ai8x-synthesis

# Install requirements
pip install --no-cache-dir -r requirements.txt
```

## Step 2: Copy Files to ai8x-synthesis Directory

```bash
# Copy your trained checkpoint
cp /root/MAX7800x-Jupyter-training/best_ecg_model_ai8x.pth.tar /root/ai8x-synthesis

# Copy KAT sample
cp /root/MAX7800x-Jupyter-training/sample_ecg_1x128.npy /root/ai8x-synthesis

# Copy network YAML file
cp /root/MAX7800x-Jupyter-training/ecg-net.yaml /root/ai8x-synthesis/networks/


```


## Step 3: Quantize the Model

**IMPORTANT:** Your model was trained with QAT enabled, so BatchNorm layers are already folded.

```bash
# Quantize your QAT-trained ECG model checkpoint to int8
python3 quantize.py \
  best_ecg_model_ai8x.pth.tar \
  best_ecg_model_ai8x_q8.pth.tar \
  --device MAX78002 \
  -v
```

**Expected Output:**
- Creates `best_ecg_model_ai8x_q8.pth.tar` (quantized checkpoint)
- Shows layer-by-layer quantization statistics
- Reports any quantization warnings
- ✅ No "BatchNorm layers have been folded" errors (because QAT was used)

## Step 4: Synthesize C Code for MAX78002

```bash
# Generate C code for MAX78002 deployment
python3 ai8xize.py \
  --test-dir sdk/Examples/MAX78002/CNN \
  --prefix ecg_classifier \
  --checkpoint-file best_ecg_model_ai8x_q8.pth.tar \
  --config-file networks/ecg-net.yaml \
  --sample-input sample_ecg_1x128.npy \
  --device MAX78002 \
  --compact-data \
  --mexpress \
  --timer 0 \
  --display-checkpoint \
  --verbose \
  --overwrite

```

python3 ai8xize.py \
  --test-dir sdk/Examples/MAX78002/CNN \
  --prefix ecg_classifier \
  --checkpoint-file best_ecg_model_ai8x_q8.pth.tar \
  --config-file networks/ecg-net.yaml \
  --device MAX78002 \
  --compact-data \
  --mexpress \
  --timer 0 \
  --display-checkpoint \
  --verbose \
  --overwrite

**Parameters Explained:**
- `--test-dir`: Output directory for generated C code
- `--prefix`: Prefix for generated files (ecg_classifier_*)
- `--checkpoint-file`: Your quantized model
- `--config-file`: Network architecture YAML
- `--device MAX78002`: Target hardware
- `--compact-data`: Optimize memory usage
- `--mexpress`: Enable faster execution mode
- `--timer 0`: Use timer 0 for performance measurement
- `--display-checkpoint`: Show checkpoint details
- `--verbose`: Detailed output
- `--overwrite`: Overwrite existing files

## Step 5: Verify Generated Files

Check that these files were created in `sdk/Examples/MAX78002/CNN/`:
```
ecg_classifier/
├── cnn.c           # CNN layer implementation
├── cnn.h           # Header file
├── weights.h       # Quantized weights
├── main.c          # Example main program
├── Makefile        # Build configuration
└── project.mk      # Project settings
```

## Step 6: Build and Flash to MAX78002

```bash
# Navigate to generated project
cd sdk/Examples/MAX78002/CNN/ecg_classifier

# Build the project
make

# Flash to MAX78002 (requires connected hardware)
make flash
```

## Troubleshooting

### Issue: "The checkpoint file contains 1-dimensional weights for `conv1_1.bn.weight`... Ensure the BatchNorm layers have been folded"
**Solution:** This means QAT was not properly enabled during training.

**Action Required:**
1. Re-run the training notebook ECG_MIT_BIH.ipynb
2. Make sure to execute Cell 10.5 (Enable QAT) BEFORE Cell 11 (Training Loop)
3. After training completes, run the verification cell to confirm BatchNorm folding
4. Look for "✅ SUCCESS: No BatchNorm layers found!" in the output

**To verify your current checkpoint:**
```python
import torch
ckpt = torch.load('best_ecg_model_ai8x.pth.tar')
bn_keys = [k for k in ckpt['state_dict'].keys() if '.bn.' in k]
if bn_keys:
    print("❌ ERROR: BatchNorm layers NOT folded")
    print("   Re-run training with Cell 10.5 enabled")
else:
    print("✅ SUCCESS: Checkpoint is ready for synthesis")
```

### Issue: "Network architecture mismatch"
**Solution:** The YAML file must exactly match your PyTorch model architecture. Check:
- Number of layers
- Kernel sizes
- Padding values
- Channel counts

### Issue: "Quantization errors"
**Solution:** Some layers may have problematic weight distributions. Try:
- Adding small epsilon to avoid division by zero
- Adjusting quantization clipping ranges
- Using `--avg-pool-rounding` flag

### Issue: "Memory overflow"
**Solution:** MAX78002 has limited memory. Try:
- Reduce batch size (use `--fifo`)
- Enable `--compact-data`
- Reduce intermediate buffer sizes

### Issue: "KAT mismatch"
**Solution:** Hardware output doesn't match software:
- Verify input preprocessing matches training
- Check quantization range ([-128, 127])
- Ensure ai8x.normalize() was used during training

## Testing the Deployed Model

The generated `main.c` will:
1. Load your KAT sample (`sample_ecg_1x128.npy` embedded as C array)
2. Run inference on MAX78002 CNN accelerator
3. Print predicted class and confidence
4. Compare against expected output (from PyTorch)

Expected output should show:
```
Input: [ECG waveform data]
Predicted class: Normal (N)
Confidence: 89.81%
```

## Performance Metrics

With MAX78002 CNN accelerator:
- **Inference time**: ~1-5ms per ECG segment
- **Power consumption**: ~10-50mW (depending on clock speed)
- **Accuracy**: Should match PyTorch model (89.81% on normal beats)

## Next Steps

1. Test on real ECG data from MAX78002
2. Optimize power consumption (adjust clock speeds)
3. Implement continuous monitoring mode
4. Add wireless transmission of predictions
