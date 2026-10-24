# ANDG-Net Glaucoma Dashboard

Upload a colour fundus photograph and get: glaucoma / suspicious / no glaucoma, confidence, predicted CDR, severity stage,
disc-size estimate, and a step-by-step trace of what the verifier did. It runs the same pipeline as the notebook
(border crop + CLAHE, Stage 1 MaxViT-Tiny + CBAM + GC with flip test-time augmentation, Stage 2 deviation-gated CDR head, tau check).

## 1. Get the checkpoints (from your Colab run)
In the notebook's results folder, copy these two files into `checkpoints/`:

| File | Produced by |
|---|---|
| `stage1_best.pth` | Stage 1 training (EMA weights, best validation AUC) |
| `stage2_deviation_gate_primary.pth` | Stage 2, seed 0 of the disc-size-conditioned gate |

They are in `<RESULTS_DIR>/checkpoints/` on Drive. Without them the app starts in **DEMO MODE** (red banner, placeholder numbers only).

## 2. Run
```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000, drop an image, press **Analyse image**. A GPU is used if available; CPU takes a few seconds per image.

## 3. Settings (`config.json`)
- `tau`: CDR verification threshold. Use the value your notebook printed for TAU (0.44 in the reported Run A). Can also be set with env var `ANDG_TAU`.
- `stage1_threshold`: 0.5, as in the notebook.
- `borderline_band`: 0.10. A Stage 1 probability within 0.10 of the threshold is shown as "Suspicious". This band is a dashboard convention, not part of the trained pipeline.
- `severity_thresholds`: [0.65, 0.85] (mild / moderate / severe).

## 4. How the result is decided
1. Stage 1 gives a glaucoma probability (mean of original, horizontal flip, vertical flip).
2. The verifier predicts the CDR from the gated embedding. If Stage 1 is positive, the call is confirmed only when CDR >= tau.
3. Glaucoma likely = positive and confirmed and not borderline. Suspicious = positive but not confirmed, or borderline. No glaucoma = clear negative.
4. Severity is shown only when Stage 1 is positive: mild < 0.65, moderate 0.65 to 0.85, severe > 0.85.
5. Confidence is the probability supporting the verdict. Stage 1 is only moderately calibrated (ECE about 0.05), so read it as a score, not a guarantee.

## 5. Notes and limits
- This is the Run A verifier (original Stage 2). The Run B two-sided verifier needs extra fitted objects (PCA, logistic fusion, spatial gate) that the notebook does not save as a single checkpoint, so it is not included.
- CDR error is about 0.07 and shrinks towards the middle of the range; severity was validated only on mild and moderate cases.
- Research prototype, not a medical device. Do not use for clinical decisions.
- Files: `app.py` (server), `inference.py` (pipeline and decision logic), `model_defs.py` (copied from the notebook), `templates/index.html` (UI).
