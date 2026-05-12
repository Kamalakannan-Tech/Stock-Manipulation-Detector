"""
Quick end-to-end test of the confidence pipeline:
  1. Load trained model
  2. Run single-pass prediction
  3. Run MC Dropout (20 passes)
  4. Print all confidence fields the API would return
"""
import sys, warnings, torch, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.models.transformer import TemporalFusionTransformer

MODEL_PATH = 'models/saved_models/best_model.pth'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MC_PASSES  = 20

print(f'Device: {DEVICE}')

# ── Load model ─────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
sd   = ckpt['model_state_dict']
if any(k.startswith('_orig_mod.') for k in sd):
    sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    print('Stripped _orig_mod. prefix')

model = TemporalFusionTransformer(
    social_features  = ckpt.get('social_features', 6),
    market_features  = ckpt.get('market_features', 13),
    d_model          = ckpt.get('d_model', 128),
    nhead            = ckpt.get('nhead', 8),
    num_layers       = ckpt.get('num_layers', 3),
    dropout          = ckpt.get('dropout', 0.3),
).to(DEVICE)
model.load_state_dict(sd, strict=True)
print(f'Model loaded  (threshold={ckpt.get("best_threshold", 0.5):.2f})')

# ── Synthetic input (simulate a high-manipulation signal) ───────────────────────
torch.manual_seed(42)
# Social: high sentiment, high post_volume, high bot activity
social_sim = torch.zeros(1, 24, 6).to(DEVICE)
social_sim[:, :, 0] = 2.5    # sentiment_score spike
social_sim[:, :, 1] = 3.0    # post_volume surge
social_sim[:, :, 3] = 2.0    # bot_activity_score

# Market: high volume z-score, strong upward momentum
market_sim = torch.zeros(1, 24, 13).to(DEVICE)
market_sim[:, :, 9]  = 4.0   # Abnormal_Volume_Zscore
market_sim[:, :, 1]  = 0.08  # Returns_1h (8% in 1h)
market_sim[:, :, 2]  = 0.15  # Returns_4h (15% in 4h)

# ── Single-pass prediction ─────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    out = model(social_sim, market_sim)

det_prob = float(out['manipulation_probability'].item())
print(f'\n── Deterministic single-pass ──')
print(f'  manipulation_probability : {det_prob:.4f}  ({det_prob*100:.1f}%)')

# ── MC Dropout (20 stochastic passes) ──────────────────────────────────────────
model.train()  # enable dropout
samples = []
with torch.no_grad():
    for _ in range(MC_PASSES):
        o = model(social_sim, market_sim)
        samples.append(float(o['manipulation_probability'].item()))
model.eval()

arr  = np.array(samples, dtype=np.float32)
mean = float(arr.mean())
std  = float(arr.std())
ci5  = float(np.percentile(arr, 5))
ci95 = float(np.percentile(arr, 95))
conf = max(0.0, min(1.0, 1.0 - 2 * std))

# Confidence level
if   conf >= 0.85: level = 'Very High'
elif conf >= 0.70: level = 'High'
elif conf >= 0.50: level = 'Moderate'
elif conf >= 0.30: level = 'Low'
else:              level = 'Very Low'

print(f'\n── MC Dropout ({MC_PASSES} passes) ──')
print(f'  confidence_score     : {mean:.4f}  ({mean*100:.1f}%)')
print(f'  confidence_level     : {level}')
print(f'  uncertainty (σ)      : {std:.4f}  ({std*100:.2f}%)')
print(f'  90% CI               : [{ci5*100:.1f}%, {ci95*100:.1f}%]')
print(f'  MC sample range      : [{arr.min()*100:.1f}%, {arr.max()*100:.1f}%]')

print(f'\n── Full API response (confidence fields) ──')
import json
api_payload = {
    'manipulation_probability': round(det_prob, 4),
    'confidence_score'        : round(mean, 4),
    'confidence_level'        : level,
    'uncertainty'             : round(std, 4),
    'confidence_interval'     : [round(ci5, 4), round(ci95, 4)],
}
print(json.dumps(api_payload, indent=2))
print('\n[OK] Confidence pipeline working end-to-end')
