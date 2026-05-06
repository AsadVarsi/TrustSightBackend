"""
SynthScan — AI Image Forensics Engine
Final Year Project: AI & Computer Vision

Multi-layer detection pipeline:
  1. Transformer-based image classification (primary signal)
  2. Error Level Analysis (ELA) — detects compression artefact anomalies
  3. DCT high-frequency energy analysis — AI images lack natural HF noise
  4. Laplacian noise estimation — AI images are unusually smooth
  5. Edge structure analysis via Canny
  6. Colour channel cross-correlation
  7. Weighted hybrid scoring for final verdict
"""

import io
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from scipy.fftpack import dct



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

print("[SynthScan] Loading transformer model...")
_classifier = pipeline(
    "image-classification",
    model="Organika/sdxl-detector"
)
print("[SynthScan] Model ready.\n")



def extract_noise_level(gray: np.ndarray) -> float:
    """
    Estimate high-frequency noise using the standard deviation of
    the Laplacian response. Real photographs contain sensor/JPEG noise
    that produces a higher Laplacian sigma. AI-generated images —
    especially diffusion models — are over-smoothed during sampling,
    resulting in suppressed high-frequency detail.

    Returns: sigma of Laplacian (higher = more natural noise)
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.std(laplacian))



def extract_ela_score(image: Image.Image, quality: int = 90):
    """
    Error Level Analysis (ELA) re-saves the image at a known JPEG quality
    and measures per-pixel difference from the original.

    Real images re-compressed at Q=90 show uniform error levels because
    they were already compressed. AI-generated images — which have no
    prior compression history — show anomalously low or spatially
    inconsistent ELA residuals.

    Returns a suspicion score in [0, 1]:  0 = natural, 1 = synthetic
    Reference: Krawetz (2007), "A Picture's Worth"
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    orig = np.array(image,        dtype=np.float32)
    recp = np.array(recompressed, dtype=np.float32)
    ela_map = np.abs(orig - recp)

    ela_mean = float(np.mean(ela_map))
    ela_std  = float(np.std(ela_map))

    # Real images: ela_mean typically > 8; AI images often < 4
    mean_susp = float(np.clip(1.0 - (ela_mean / 15.0), 0.0, 1.0))
    std_susp  = float(np.clip(1.0 - (ela_std  / 20.0), 0.0, 1.0))
    suspicion = 0.5 * mean_susp + 0.5 * std_susp

    return float(np.clip(suspicion, 0.0, 1.0)), ela_mean, ela_std



def extract_dct_hf_ratio(gray: np.ndarray) -> float:
    """
    The 2D Discrete Cosine Transform decomposes an image into frequency
    components. Real photographs contain natural high-frequency (HF)
    energy from sensor noise and fine texture.

    AI-generated images from diffusion/GAN models lack this natural HF
    noise and concentrate energy in low-to-mid frequencies. A low
    HF-to-total energy ratio is therefore suspicious.

    We compute the 2D DCT on a centre crop and measure the fraction of
    energy in the high-frequency quadrant.

    Returns: HF energy ratio in [0, 1] — lower values more suspicious.
    """
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    crop = gray[max(0, cy-64):cy+64, max(0, cx-64):cx+64].astype(np.float32)

    dct2d = dct(dct(crop, axis=0, norm="ortho"), axis=1, norm="ortho")

    total_energy = float(np.sum(dct2d ** 2)) + 1e-8
    hf_energy    = float(np.sum(dct2d[dct2d.shape[0]//2:, dct2d.shape[1]//2:] ** 2))

    return hf_energy / total_energy



def extract_edge_density(gray: np.ndarray) -> float:
    """
    Canny edge density: fraction of pixels classified as edges.
    AI images tend toward slightly lower, more structured edge maps
    compared to real photos with organic fine-detail texture.
    """
    edges = cv2.Canny(gray, 100, 200)
    return float(np.sum(edges > 0) / edges.size)



def extract_channel_correlation(rgb: np.ndarray) -> float:
    """
    In real images, RGB channels have moderate cross-correlation driven
    by scene lighting. GAN and diffusion models produce unnaturally high
    inter-channel correlation due to the smoothing effect of learned
    decoders that do not model independent channel noise.

    Returns: mean absolute pairwise Pearson correlation of R, G, B.
    Higher = more suspicious (more synthetic).
    """
    r = rgb[:,:,0].flatten().astype(np.float64)
    g = rgb[:,:,1].flatten().astype(np.float64)
    b = rgb[:,:,2].flatten().astype(np.float64)

    rg = abs(float(np.corrcoef(r, g)[0, 1]))
    rb = abs(float(np.corrcoef(r, b)[0, 1]))
    gb = abs(float(np.corrcoef(g, b)[0, 1]))

    return float(np.mean([rg, rb, gb]))


def compute_forensic_ai_probability(image: Image.Image):
    """
    Aggregate all forensic signals into a single AI-probability [0, 100].

    Signal weights:
      ELA suspicion:         30%  (strong — compression history reliable)
      DCT HF deficit:        30%  (strong — frequency structure consistent)
      Laplacian noise:       25%  (medium — resolution-dependent)
      Channel correlation:   15%  (weak   — scene-dependent)

    Edge density returned in metadata but excluded from the score
    since it correlates with content rather than origin.
    """
    rgb  = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    noise                          = extract_noise_level(gray)
    ela_susp, ela_mean, ela_std    = extract_ela_score(image)
    dct_hf                         = extract_dct_hf_ratio(gray)
    edge_density                   = extract_edge_density(gray)
    chan_corr                      = extract_channel_correlation(rgb)

    # Convert to suspicion scores [0, 1]
    # Noise:  real > 60, AI often < 40
    noise_susp = float(np.clip(1.0 - (noise / 80.0), 0.0, 1.0))
    # DCT HF: real > 0.15, AI often < 0.08
    dct_susp   = float(np.clip(1.0 - (dct_hf / 0.20), 0.0, 1.0))
    # Channel corr: real ~0.6–0.85, AI often > 0.90
    corr_susp  = float(np.clip((chan_corr - 0.55) / 0.45, 0.0, 1.0))

    forensic_prob = (
        0.30 * ela_susp  +
        0.30 * dct_susp  +
        0.25 * noise_susp +
        0.15 * corr_susp
    ) * 100.0

    features = {
        "noise":          round(noise,         2),
        "ela_mean":       round(ela_mean,       2),
        "ela_std":        round(ela_std,        2),
        "ela_suspicion":  round(ela_susp,       3),
        "dct_hf_ratio":   round(dct_hf,         4),
        "dct_suspicion":  round(dct_susp,       3),
        "edge_density":   round(edge_density,   4),
        "channel_corr":   round(chan_corr,       3),
        "forensic_prob":  round(forensic_prob,  2),
    }

    return forensic_prob, features



_MODEL_WEIGHT    = 0.65
_FORENSIC_WEIGHT = 0.35
_THRESHOLD       = 50.0


def hybrid_score(model_ai_prob: float, forensic_prob: float) -> float:
    """
    Final decision score: weighted combination of transformer output
    and forensic pipeline, both in [0, 100].

    The transformer carries higher weight (0.65) because it was trained
    on labelled real/AI image pairs. Forensics (0.35) are valuable when
    the model is uncertain (score near 50%) or for novel generators
    not well-represented in the model's training distribution.
    """
    return (_MODEL_WEIGHT * model_ai_prob) + (_FORENSIC_WEIGHT * forensic_prob)



@app.route("/analyze", methods=["POST"])
def analyze_image():
    """
    POST /analyze
    Expects:  multipart/form-data with field 'image'
    Returns:  JSON verdict with confidence and per-feature breakdown
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not decode image — unsupported format"}), 400

    # ── Stage 1: Transformer ─────────────────────────────────────
    raw        = _classifier(image)[0]
    label_str  = raw["label"].lower()
    model_conf = raw["score"] * 100.0

    ai_keywords = ("artificial", "ai", "fake", "generated", "synthetic")
    model_ai_prob = model_conf if any(k in label_str for k in ai_keywords) \
                    else 100.0 - model_conf

    # ── Stage 2: Forensic pipeline ───────────────────────────────
    forensic_prob, features = compute_forensic_ai_probability(image)

    # ── Stage 3: Hybrid decision ─────────────────────────────────
    final_score = hybrid_score(model_ai_prob, forensic_prob)
    verdict     = "AI Generated" if final_score >= _THRESHOLD else "Real Image"

    print(
        f"[SynthScan] model={model_ai_prob:.1f}%  "
        f"forensic={forensic_prob:.1f}%  "
        f"final={final_score:.1f}%  → {verdict}"
    )

    return jsonify({
        "result":     verdict,
        "confidence": round(final_score, 2),
        "analysis": {
            "model_confidence": round(model_conf,    2),
            "model_ai_prob":    round(model_ai_prob, 2),
            "noise":            features["noise"],
            "edge_density":     features["edge_density"],
            "ela_mean":         features["ela_mean"],
            "ela_suspicion":    features["ela_suspicion"],
            "dct_hf_ratio":     features["dct_hf_ratio"],
            "channel_corr":     features["channel_corr"],
            "forensic_prob":    features["forensic_prob"],
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
