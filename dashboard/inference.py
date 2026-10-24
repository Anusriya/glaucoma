"""Inference pipeline of ANDG-Net (Run A verifier), reproducing the notebook exactly:
   pre-processing (CELL 3) -> Stage 1 with flip TTA (CELL 7/9) -> Stage 2 verification net (CELL 13-15).
"""
import os, io, json, math, base64, hashlib
import numpy as np
import cv2
from PIL import Image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CACHE_SIZE, IMG = 256, 224
BUCKETS = ['Small', 'Medium', 'Large']


# ------------------------------------------------------------------ pre-processing (notebook CELL 3)
def preprocess(img_rgb, cache_size=CACHE_SIZE):
    H, W = img_rgb.shape[:2]
    m = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) > 12
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    fg = float(m.mean())
    cropped = bool(len(rows) and len(cols) and fg > 0.05)
    if cropped:
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    else:
        y0, y1, x0, x1 = 0, H, 0, W
    crop = img_rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]; s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    canvas = cv2.resize(canvas, (cache_size, cache_size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    canvas = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    info = dict(width=int(W), height=int(H), foreground_fraction=round(fg, 3), border_cropped=cropped)
    return canvas, info


def quality_warnings(img_rgb, info):
    w = []
    if min(info['width'], info['height']) < 200:
        w.append('Image is very small (under 200 px); results may be unreliable.')
    if not info['border_cropped']:
        w.append('Little dark border was found, so the automatic border crop was skipped. Check that this is a full fundus photograph.')
    r, g, b = [float(img_rgb[..., i].mean()) for i in range(3)]
    if not (r > b * 1.15 and r > g * 0.9):
        w.append('Colour balance does not look like a typical retinal photograph. Make sure the file is a colour fundus image.')
    return w


def to_b64_jpeg(arr_rgb, quality=88):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


# ------------------------------------------------------------------ decision logic (dashboard convention)
def severity_stage(cdr, thr):
    return 'Mild' if cdr < thr[0] else ('Moderate' if cdr <= thr[1] else 'Severe')


def cdr_band(cdr):
    if cdr < 0.5: return 'Within the usual normal range (< 0.5)'
    if cdr < 0.65: return 'Borderline / suspicious range (0.5 to 0.65)'
    if cdr <= 0.85: return 'Enlarged cup (0.65 to 0.85)'
    return 'Very large cup (> 0.85)'


def decide(p, cdr, cfg):
    thr, tau, band = cfg['stage1_threshold'], cfg['tau'], cfg['borderline_band']
    stage1_pos = p >= thr
    confirmed = bool(cdr >= tau)
    borderline = abs(p - thr) <= band
    if stage1_pos and confirmed and not borderline:
        verdict, label = 'glaucoma', 'Glaucoma likely'
        reason = 'Stage 1 flagged glaucoma and the verifier confirmed it (predicted CDR reached the threshold).'
    elif stage1_pos and not confirmed:
        verdict, label = 'suspicious', 'Suspicious: review recommended'
        reason = 'Stage 1 flagged glaucoma, but the predicted CDR is below the verification threshold, so the call is not confirmed (the notebook flags this as "anatomical-scale-driven, recommend review").'
    elif borderline:
        verdict, label = 'suspicious', 'Suspicious: borderline score'
        reason = f'The Stage 1 probability is within {band:.2f} of the decision threshold, so the result is close to the boundary.'
    else:
        verdict, label = 'normal', 'No glaucoma detected'
        reason = 'Stage 1 probability is below the decision threshold with a clear margin.'
    conf = p if verdict == 'glaucoma' else (1 - p if verdict == 'normal' else p)
    return dict(verdict=verdict, label=label, reason=reason, stage1_positive=bool(stage1_pos),
                verifier_confirmed=confirmed, borderline=bool(borderline), confidence=float(conf))


# ------------------------------------------------------------------ engine
class Engine:
    def __init__(self, cfg, base_dir):
        self.cfg, self.base = cfg, base_dir
        self.demo, self.error = False, None
        self.torch = None
        s1 = os.path.join(base_dir, cfg['stage1_ckpt']); s2 = os.path.join(base_dir, cfg['stage2_ckpt'])
        try:
            if not (os.path.isfile(s1) and os.path.isfile(s2)):
                raise FileNotFoundError(f'Checkpoint(s) not found: {s1 if not os.path.isfile(s1) else s2}')
            self._load(s1, s2)
        except Exception as e:                      # missing checkpoints / torch not installed / load error
            self.error = f'{type(e).__name__}: {e}'
            if cfg.get('allow_demo_mode', True):
                self.demo = True
            else:
                raise

    def _load(self, s1, s2):
        import torch, torchvision.transforms as T
        from model_defs import Stage1Net, VerificationNet
        self.torch = torch
        dev = self.cfg.get('device', 'auto')
        self.device = torch.device('cuda' if (dev == 'auto' and torch.cuda.is_available()) else ('cpu' if dev == 'auto' else dev))
        ck = torch.load(s1, map_location=self.device, weights_only=False)
        self.m1 = Stage1Net(False).to(self.device)
        self.m1.load_state_dict(ck['model_state_dict'] if 'model_state_dict' in ck else ck['model']); self.m1.eval()
        self.ck_info = dict(epoch=ck.get('epoch'), val_auc=ck.get('val_auc'))
        self.m2 = VerificationNet('conditioned').to(self.device)
        sd = torch.load(s2, map_location=self.device, weights_only=False)
        self.m2.load_state_dict(sd['state_dict'] if isinstance(sd, dict) and 'state_dict' in sd else sd); self.m2.eval()
        self.tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])

    # ---------------------------------------------------------- real model
    def _run_model(self, canvas):
        torch = self.torch
        x = self.tf(Image.fromarray(canvas)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            views = [('original', x), ('horizontal flip', torch.flip(x, [3])), ('vertical flip', torch.flip(x, [2]))]
            probs = [float(torch.softmax(self.m1(v).float(), 1)[0, 1]) for _, v in views]
            A = self.m1.embed(x).float()
            cdr, blog, D = self.m2(A)
            bp = torch.softmax(blog, 1)
            E = self.m2.gate.reference(bp)
            rel = float((A - E).norm() / E.norm().clamp(min=1e-8))
        return dict(
            p=float(np.mean(probs)), view_probs={n: p for (n, _), p in zip(views, probs)},
            cdr=float(cdr[0]), bucket_probs=[float(v) for v in bp[0]],
            gate_mean=float(D.mean()), gate_open_fraction=float((D > 0.5).float().mean()),
            rel_distance=rel)

    # ---------------------------------------------------------- demo (UI test only, NOT a real prediction)
    def _run_demo(self, canvas):
        h = int(hashlib.sha256(canvas.tobytes()).hexdigest(), 16)
        r = lambda k: ((h >> (k * 13)) & 0xFFF) / 4095.0
        p = r(1); views = [min(1, max(0, p + (r(i) - .5) * .06)) for i in (2, 3, 4)]
        b = np.array([r(5) + .1, r(6) + .1, r(7) + .1]); b = b / b.sum()
        return dict(p=float(np.mean(views)), view_probs=dict(zip(['original', 'horizontal flip', 'vertical flip'], views)),
                    cdr=float(0.25 + 0.6 * r(8)), bucket_probs=[float(v) for v in b],
                    gate_mean=float(0.8 + 0.1 * r(9)), gate_open_fraction=float(0.9 + 0.1 * r(10)), rel_distance=float(3 + 2 * r(11)))

    # ---------------------------------------------------------- public API
    def analyze(self, img_rgb):
        canvas, info = preprocess(img_rgb)
        out = self._run_demo(canvas) if self.demo else self._run_model(canvas)
        cfg = self.cfg
        d = decide(out['p'], out['cdr'], cfg)
        sev = None
        if out['p'] >= cfg['stage1_threshold']:
            sev = severity_stage(out['cdr'], cfg['severity_thresholds'])
        vp = list(out['view_probs'].values())
        bi = int(np.argmax(out['bucket_probs']))
        steps = [
            dict(title='Stage 1: glaucoma classifier', status='info',
                 text=f"MaxViT-Tiny + CBAM + GC gave a glaucoma probability of {out['p']*100:.1f}% (mean of 3 views; decision threshold {cfg['stage1_threshold']:.2f}). Stage 1 says: {'glaucoma' if d['stage1_positive'] else 'no glaucoma'}."),
            dict(title='Disc-size estimate', status='info',
                 text=f"Predicted disc-size group: {BUCKETS[bi]} ({out['bucket_probs'][bi]*100:.0f}%). This picks the healthy reference used for comparison. The predictor is weak (about 52% accurate on test images), so treat it as a soft hint."),
            dict(title='Healthy reference and deviation gate', status='info',
                 text=f"The embedding was compared with a healthy prototype mixed from the three size groups. The gate kept {out['gate_open_fraction']*100:.0f}% of channels above 0.5 (mean gate {out['gate_mean']:.2f}); relative distance to the reference {out['rel_distance']:.2f} (unitless)."),
            dict(title='CDR regression', status='info',
                 text=f"Predicted vertical CDR {out['cdr']:.3f}. {cdr_band(out['cdr'])}. Typical error on the test set is about 0.07."),
            dict(title=f"Verification check (CDR >= {cfg['tau']:.2f})",
                 status=('pass' if (d['verifier_confirmed'] and d['stage1_positive']) else ('fail' if d['stage1_positive'] else 'skip')),
                 text=(('Confirmed: predicted CDR reaches the threshold.' if d['verifier_confirmed'] else 'Not confirmed: predicted CDR is below the threshold.')
                       if d['stage1_positive'] else 'Not applied: the verifier only checks Stage 1 positives; negative calls pass through unchanged.')),
            dict(title='Final decision', status=d['verdict'], text=d['reason']),
        ]
        warnings = quality_warnings(img_rgb, info)
        if max(vp) - min(vp) > 0.15:
            warnings.append(f'The three test-time views disagree (spread {max(vp)-min(vp):.2f}); the prediction is less stable than usual.')
        return dict(
            demo=self.demo, decision=d,
            stage1=dict(probability=out['p'], views=out['view_probs'], threshold=cfg['stage1_threshold']),
            cdr=dict(value=out['cdr'], band=cdr_band(out['cdr']), severity=sev,
                     severity_note=('Severity is shown only when Stage 1 is positive. Thresholds: mild < 0.65, moderate 0.65 to 0.85, severe > 0.85.'),
                     typical_error=0.07),
            disc=dict(buckets=dict(zip(BUCKETS, out['bucket_probs'])), predicted=BUCKETS[bi]),
            verifier=dict(tau=cfg['tau'], confirmed=d['verifier_confirmed'], gate_mean=out['gate_mean'],
                          gate_open_fraction=out['gate_open_fraction'], rel_distance=out['rel_distance'], steps=steps),
            image=dict(info=info, processed=to_b64_jpeg(canvas)),
            warnings=warnings)

    def status(self):
        return dict(mode='demo' if self.demo else 'model', error=self.error,
                    device=None if self.demo else str(self.device),
                    stage1=None if self.demo else self.ck_info,
                    tau=self.cfg['tau'], threshold=self.cfg['stage1_threshold'])
