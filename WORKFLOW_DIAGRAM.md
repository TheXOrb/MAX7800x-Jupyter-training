# QAT Training Workflow

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    ECG_MIT_BIH.ipynb                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Cell 1-10:  Setup & Configuration                             │
│              ├── Import libraries                               │
│              ├── Load data                                      │
│              ├── Define model                                   │
│              └── Setup optimizer                                │
│                                                                  │
│  Cell 10.5:  ⭐ ENABLE QAT (CRITICAL!)                          │
│              ├── Define qat_policy                              │
│              ├── ai8x.initiate_qat(model, qat_policy)          │
│              └── Fold BatchNorm automatically                   │
│                                                                  │
│  Cell 11:    Training Loop                                      │
│              └── Model learns with quantization                 │
│                                                                  │
│  Cell 12-13: Evaluation & Save                                  │
│              └── Save: best_ecg_model_ai8x.pth.tar             │
│                                                                  │
│  New Cell:   Verify BatchNorm Folding                          │
│              ├── Check: No .bn. keys ✅                         │
│              ├── Check: QAT parameters exist ✅                 │
│              └── Display: Ready for synthesis 🎉                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Copy to ai8x-synthesis directory                    │
│  ├── best_ecg_model_ai8x.pth.tar                               │
│  ├── sample_ecg_1x128.npy                                      │
│  └── ecg-net.yaml                                               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Quantization (quantize.py)                    │
│  Input:  best_ecg_model_ai8x.pth.tar (QAT checkpoint)          │
│  Output: best_ecg_model_ai8x_q8.pth.tar (8-bit weights)        │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Synthesis (ai8xize.py)                         │
│  Input:  best_ecg_model_ai8x_q8.pth.tar + ecg-net.yaml         │
│  Output: C code for MAX78002                                    │
│          ├── cnn.c                                              │
│          ├── cnn.h                                              │
│          ├── weights.h                                          │
│          └── main.c                                             │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Build & Deploy                                │
│  make && make flash → MAX78002 Hardware                        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Decision Points

```
Training Start
      │
      ▼
Did you run Cell 10.5? ──NO──> ❌ STOP! Run Cell 10.5 first
      │
     YES
      │
      ▼
Training Complete
      │
      ▼
Run Verification Cell
      │
      ├──> BatchNorm folded? ──NO──> ❌ Re-train with Cell 10.5
      │           │
      │          YES
      │           │
      └──────────▼
            QAT params exist? ──NO──> ⚠️ Re-train recommended
                  │
                 YES
                  │
                  ▼
            ✅ Ready for synthesis!
                  │
                  ▼
            Copy to ai8x-synthesis
                  │
                  ▼
            Quantize → Synthesize → Deploy
```

## What Happens With vs Without QAT

```
WITHOUT QAT (OLD WAY - BROKEN):
────────────────────────────────
Training → Save Checkpoint
              │
              ├── Contains: conv1.weight ✅
              ├── Contains: conv1.bias ✅
              ├── Contains: conv1.bn.weight ❌ (Problem!)
              └── Contains: conv1.bn.bias ❌ (Problem!)
              │
              ▼
        ai8xize.py
              │
              ▼
        ❌ ERROR: "Ensure the BatchNorm layers have been folded"


WITH QAT (NEW WAY - WORKS):
───────────────────────────
Training with QAT → Save Checkpoint
              │
              ├── Contains: conv1.weight ✅ (includes BatchNorm)
              ├── Contains: conv1.bias ✅ (includes BatchNorm)
              ├── Contains: conv1.weight_bits ✅ (QAT parameter)
              └── Contains: conv1.output_shift ✅ (QAT parameter)
              │
              ▼
        ai8xize.py
              │
              ▼
        ✅ SUCCESS: C code generated!
```

## Memory Map: Before vs After QAT

```
BEFORE QAT:
───────────
Model Layer: Conv1d → BatchNorm → ReLU
               ↓          ↓
           Separate   Separate
           Weights    Parameters
               
           ❌ Two operations in hardware
           ❌ More memory needed
           ❌ Synthesis fails


AFTER QAT:
──────────
Model Layer: Conv1d(with fused BN) → ReLU
                      ↓
                  Combined
                  Weights
                  
           ✅ One operation in hardware
           ✅ Less memory needed
           ✅ Synthesis succeeds
```

## Checkpoint Contents Comparison

```
NON-QAT CHECKPOINT:
───────────────────
state_dict:
  conv1.op.weight        ← Conv weights
  conv1.op.bias          ← Conv bias
  conv1.bn.weight        ← BatchNorm gamma ❌
  conv1.bn.bias          ← BatchNorm beta ❌
  conv1.bn.running_mean  ← BatchNorm stats ❌
  conv1.bn.running_var   ← BatchNorm stats ❌


QAT CHECKPOINT:
───────────────
state_dict:
  conv1.op.weight        ← Fused weights (Conv + BN) ✅
  conv1.op.bias          ← Fused bias (Conv + BN) ✅
  conv1.weight_bits      ← QAT quantization config ✅
  conv1.bias_bits        ← QAT quantization config ✅
  conv1.output_shift     ← QAT scaling factor ✅
  conv1.shift_quantile   ← QAT parameter ✅
```

## Timeline: Your Journey

```
PAST (What Happened):
─────────────────────
1. Trained model without Cell 10.5
2. Saved checkpoint with unfused BatchNorm
3. Tried synthesis → ERROR ❌


PRESENT (What We Did):
──────────────────────
1. Added Cell 10.5 to enable QAT ✅
2. Added verification cell ✅
3. Updated SYNTHESIS_INSTRUCTIONS.md ✅


FUTURE (What You'll Do):
────────────────────────
1. Re-run training from Cell 1
2. Execute Cell 10.5 before Cell 11
3. Verify checkpoint → ✅ SUCCESS
4. Follow synthesis instructions
5. Deploy to MAX78002 → 🎉
```
