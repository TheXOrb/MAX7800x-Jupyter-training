# Quick Reference: Training with QAT

## Cell Execution Order (ECG_MIT_BIH.ipynb)

```
Cell 1  → Import libraries
Cell 2  → Setup MAX78002 device
Cell 3  → Load ECG data
Cell 4  → (skip if exists)
Cell 5  → Create datasets
Cell 6  → Create data loaders
Cell 7  → Visualize data (optional)
Cell 8  → Define model architecture
Cell 9  → Setup loss function
Cell 10 → Setup optimizer
Cell 10.5 → ⭐ ENABLE QAT (NEW - MUST RUN!)
Cell 11 → Training loop
Cell 12 → Evaluate model
Cell 13 → Save final checkpoint
[NEW]   → Verify BatchNorm folding
```

## Critical: Cell 10.5 Must Run!

**BEFORE starting training (Cell 11), you MUST run Cell 10.5:**

```python
# Cell 10.5: Enable QAT
qat_policy = {'weight_bits': 8, 'bias_bits': 8, 'overrides': {}}
ai8x.initiate_qat(model, qat_policy)
```

**What happens if you skip it:**
- ❌ BatchNorm layers won't be folded
- ❌ Synthesis will fail with BatchNorm error
- ❌ You'll need to re-train from scratch

## Verification Checklist

After training completes, run the verification cell. Look for:

✅ **SUCCESS Indicators:**
```
✅ SUCCESS: No BatchNorm layers found!
✅ SUCCESS: Found XX QAT parameters
🎉 CHECKPOINT IS READY FOR ai8x-synthesis!
```

❌ **FAILURE Indicators:**
```
❌ ERROR: BatchNorm layers NOT folded!
❌ WARNING: No QAT parameters found
❌ CHECKPOINT NOT READY
```

If you see failures → Re-run training with Cell 10.5!

## Synthesis Commands (Quick Copy)

Once training succeeds and verification passes:

```bash
# 1. Copy files
cp best_ecg_model_ai8x.pth.tar ~/ai8x-synthesis/
cp sample_ecg_1x128.npy ~/ai8x-synthesis/
cp ecg-net.yaml ~/ai8x-synthesis/networks/

# 2. Quantize
cd ~/ai8x-synthesis
python3 quantize.py \
  best_ecg_model_ai8x.pth.tar \
  best_ecg_model_ai8x_q8.pth.tar \
  --device MAX78002 -v

# 3. Synthesize
python3 ai8xize.py \
  --test-dir sdk/Examples/MAX78002/CNN \
  --prefix ecg_classifier \
  --checkpoint-file best_ecg_model_ai8x_q8.pth.tar \
  --config-file networks/ecg-net.yaml \
  --device MAX78002 \
  --compact-data --mexpress --timer 0 \
  --display-checkpoint --verbose --overwrite
```

## Troubleshooting One-Liners

**Check if checkpoint has QAT:**
```python
import torch
ckpt = torch.load('best_ecg_model_ai8x.pth.tar')
has_qat = any('weight_bits' in k for k in ckpt['state_dict'].keys())
print("QAT enabled:" if has_qat else "QAT NOT enabled - re-train!")
```

**Check if BatchNorm is folded:**
```python
import torch
ckpt = torch.load('best_ecg_model_ai8x.pth.tar')
has_bn = any('.bn.' in k for k in ckpt['state_dict'].keys())
print("ERROR: BatchNorm NOT folded!" if has_bn else "SUCCESS: Ready for synthesis!")
```

## Common Mistakes

1. ❌ Skipping Cell 10.5
   ✅ Always run Cell 10.5 before Cell 11

2. ❌ Running cells out of order
   ✅ Follow the numbered sequence

3. ❌ Using old checkpoint from non-QAT training
   ✅ Delete old checkpoint, re-train with QAT

4. ❌ Not verifying before synthesis
   ✅ Always run verification cell

## Time Estimates

| Task | Duration |
|------|----------|
| Run Cells 1-10 | ~2 minutes |
| Run Cell 10.5 (QAT) | ~1 second |
| Training (Cell 11) | ~15-30 minutes |
| Verification | ~1 second |
| Quantization | ~30 seconds |
| Synthesis | ~2-3 minutes |

**Total: ~20-35 minutes** from training to synthesis
