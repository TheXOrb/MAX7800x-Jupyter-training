# Step-by-Step Guide: Re-Training with QAT

## ✅ What We've Done
- [x] Added Cell 10.5 to enable QAT in ECG_MIT_BIH.ipynb
- [x] Added verification cell to check BatchNorm folding
- [x] Updated SYNTHESIS_INSTRUCTIONS.md with QAT workflow
- [x] Created documentation and quick references

## 📋 What You Need to Do Now

### Step 1: Open the Notebook (2 minutes)
1. Open `ECG_MIT_BIH.ipynb` in Jupyter/VS Code
2. Clear all outputs: `Edit → Clear All Outputs` (optional but recommended)
3. Restart kernel: `Kernel → Restart`

### Step 2: Execute Cells in Order (2-3 minutes)

Run these cells **in sequence**:

```
✅ Cell 1:    Import libraries
✅ Cell 2:    Setup MAX78002 device  
✅ Cell 3:    Load ECG data
✅ Cell 4:    (if it exists)
✅ Cell 5:    Create datasets
✅ Cell 6:    Create data loaders
✅ Cell 7:    Visualize data (optional, can skip)
✅ Cell 8:    Define model architecture
✅ Cell 9:    Setup loss function
✅ Cell 10:   Setup optimizer & compression scheduler
```

**WAIT! Before running Cell 11...**

### Step 3: ⭐ CRITICAL - Run Cell 10.5 (1 second)

**NEW CELL - MUST RUN!**

```
✅ Cell 10.5: Enable QAT
```

**Expected output:**
```
================================================================================
ENABLING QUANTIZATION-AWARE TRAINING (QAT)
================================================================================

QAT Policy Configuration:
  - Weight quantization: 8 bits
  - Bias quantization: 8 bits
  - Layer overrides: 0

✓ QAT ENABLED SUCCESSFULLY
  ✓ Model prepared for 8-bit quantization
  ✓ BatchNorm layers will be fused automatically
  ✓ Quantization-aware parameters initialized
  ✓ Model ready for ai8x-synthesis after training

✓ QAT Verification:
  - weight_bits parameter: ✓ Found
  - quantize_weight function: ✓ Found

🎉 QAT is properly configured!
================================================================================
```

**If you see errors here:**
- Check that Cell 10 (optimizer setup) ran successfully
- Make sure `ai8x` was imported in Cell 1
- Try restarting kernel and re-running Cells 1-10

### Step 4: Continue Training (15-30 minutes)

Now run:

```
✅ Cell 11:   Main Training Loop (this takes 15-30 min)
✅ Cell 12:   Evaluate on test set
✅ Cell 13:   Final evaluation & save
```

**During training, you should see:**
- Normal training progress
- Loss decreasing
- Accuracy improving
- "✓ [BEST MODEL SAVED]" messages

**Training will save:** `best_ecg_model_ai8x.pth.tar`

### Step 5: ⭐ VERIFY BatchNorm Folding (1 second)

**NEW CELL - Run this after training!**

Run the **verification cell** at the end of the notebook.

**Expected SUCCESS output:**
```
================================================================================
VERIFYING CHECKPOINT FOR ai8x-synthesis
================================================================================

✓ Checkpoint loaded: best_ecg_model_ai8x.pth.tar
  - Epoch: XX
  - Best Top1: XX.XX%
  - Best Loss: X.XXXX

================================================================================
BATCHNORM FOLDING VERIFICATION
================================================================================
✅ SUCCESS: No BatchNorm layers found!
   All BatchNorm layers have been properly fused
   This checkpoint is ready for ai8x-synthesis

================================================================================
QAT PARAMETERS VERIFICATION
================================================================================
✅ SUCCESS: Found XX QAT parameters
   Examples:
   - conv1.weight_bits
   - conv1.bias_bits
   - conv1.output_shift
   ... and XX more

   QAT was properly applied during training

================================================================================
CHECKPOINT COMPATIBILITY SUMMARY
================================================================================
🎉 CHECKPOINT IS READY FOR ai8x-synthesis!

Next steps:
1. Copy checkpoint to ai8x-synthesis directory
2. Run quantization: python3 quantize.py ...
3. Run synthesis: python3 ai8xize.py ...
================================================================================
```

**If you see ERRORS:**
```
❌ ERROR: BatchNorm layers NOT folded!
```
→ **Solution:** You forgot to run Cell 10.5! Go back to Step 3 and re-train.

### Step 6: Proceed to Synthesis (5-10 minutes)

Once verification shows ✅ SUCCESS, follow `SYNTHESIS_INSTRUCTIONS.md`:

```bash
# Navigate to ai8x-synthesis directory
cd ~/ai8x-synthesis

# Copy your files
cp /root/MAX7800x-Jupyter-training/best_ecg_model_ai8x.pth.tar .
cp /root/MAX7800x-Jupyter-training/sample_ecg_1x128.npy .
cp /root/MAX7800x-Jupyter-training/ecg-net.yaml networks/

# Quantize (takes ~30 seconds)
python3 quantize.py \
  best_ecg_model_ai8x.pth.tar \
  best_ecg_model_ai8x_q8.pth.tar \
  --device MAX78002 -v

# Synthesize (takes ~2-3 minutes)
python3 ai8xize.py \
  --test-dir sdk/Examples/MAX78002/CNN \
  --prefix ecg_classifier \
  --checkpoint-file best_ecg_model_ai8x_q8.pth.tar \
  --config-file networks/ecg-net.yaml \
  --device MAX78002 \
  --compact-data --mexpress --timer 0 \
  --display-checkpoint --verbose --overwrite
```

**Expected quantization output:**
```
Configuring device: MAX78002
Converting checkpoint file ...
✓ No BatchNorm errors!
✓ Weights quantized successfully
```

**Expected synthesis output:**
```
Configuring device: MAX78002
Reading networks/ecg-net.yaml ...
Reading checkpoint ...
✓ Network verified
✓ C code generated in sdk/Examples/MAX78002/CNN/ecg_classifier/
```

## 🎯 Success Criteria

You know everything worked when:

1. ✅ Cell 10.5 shows "QAT ENABLED SUCCESSFULLY"
2. ✅ Training completes without errors
3. ✅ Verification cell shows "CHECKPOINT IS READY"
4. ✅ Quantization shows no BatchNorm errors
5. ✅ Synthesis generates C code successfully

## ⚠️ Common Mistakes to Avoid

| ❌ DON'T | ✅ DO |
|----------|-------|
| Skip Cell 10.5 | Always run Cell 10.5 before training |
| Run cells out of order | Follow numbered sequence |
| Use old checkpoint | Delete old checkpoint, re-train |
| Skip verification | Always verify before synthesis |
| Interrupt training mid-way | Let training complete or use early stopping |

## 📊 Progress Tracking

Use this checklist as you work:

- [ ] Step 1: Notebook opened and kernel restarted
- [ ] Step 2: Cells 1-10 executed successfully
- [ ] Step 3: Cell 10.5 executed (QAT enabled)
- [ ] Step 4: Training completed (Cell 11-13)
- [ ] Step 5: Verification passed (✅ SUCCESS)
- [ ] Step 6: Files copied to ai8x-synthesis
- [ ] Step 6: Quantization completed
- [ ] Step 6: Synthesis completed
- [ ] Final: C code ready for MAX78002!

## 🆘 Troubleshooting

### Problem: Cell 10.5 not found
**Solution:** The cell was just added. If you don't see it:
1. Close and reopen the notebook
2. Look for the cell between Cell 10 and Cell 11
3. It should say "Cell 10.5: Enable Quantization-Aware Training"

### Problem: "ai8x has no attribute 'initiate_qat'"
**Solution:**
1. Check that Cell 1 ran successfully
2. Verify: `import max78_modules.ai8x as ai8x`
3. Restart kernel and re-run Cells 1-10

### Problem: Training accuracy is lower than before
**This is normal!** QAT may show slightly lower training accuracy because:
- Model is learning with quantization constraints
- Final deployed accuracy will be similar or better
- Hardware accuracy will match training accuracy

### Problem: Verification shows BatchNorm NOT folded
**Solution:**
1. Did you run Cell 10.5? Check your execution history
2. Did Cell 10.5 show "QAT ENABLED SUCCESSFULLY"?
3. If no to either: Re-run from Cell 10.5 onwards

## 📞 Need Help?

Check these resources:
1. `QUICK_REFERENCE.md` - Fast lookup guide
2. `WORKFLOW_DIAGRAM.md` - Visual workflow
3. `QAT_IMPLEMENTATION_SUMMARY.md` - Detailed explanation
4. `SYNTHESIS_INSTRUCTIONS.md` - Complete synthesis guide

## ⏱️ Total Time Estimate

| Task | Time |
|------|------|
| Setup & run Cells 1-10 | 2-3 min |
| Run Cell 10.5 (QAT) | 1 sec |
| Training (Cell 11) | 15-30 min |
| Evaluation (Cells 12-13) | 1-2 min |
| Verification | 1 sec |
| Synthesis pipeline | 5-10 min |
| **TOTAL** | **~25-45 min** |

## 🎉 Ready to Start!

You're all set! Open `ECG_MIT_BIH.ipynb` and begin with Step 1.

Remember: The key to success is **running Cell 10.5 before Cell 11**!

Good luck! 🚀
