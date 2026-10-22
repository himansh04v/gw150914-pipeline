"""Builds expo.html: one file with the match-finder animation (sound synced to the picture),
the crash lab, and your results (plots + audio). Run from the project folder:
    python make_expo_page.py
Needs: data/mf_animation_inputs.npz (saved by the animation notebook), optional outputs/plots/*.png and outputs/audio/*.wav
"""
import base64, json, sys
from pathlib import Path
import numpy as np
from scipy.signal import hilbert, decimate

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
D = 4                                   # keep every 4th sample for the web page (smaller file)
TEMPLATE = Path(__file__).with_name("expo_template.html")

def b64(p): return base64.b64encode(Path(p).read_bytes()).decode()

# ---- 1. the search data -------------------------------------------------
npz = ROOT / "data" / "mf_animation_inputs.npz"
scan = None
if npz.exists():
    d = np.load(npz)
    data, template = d["data"].astype(float), d["template"].astype(float)
    fs, tail = float(d["fs"]), int(round(float(d["tail_s"]) * float(d["fs"])))
    t = float(d["t_start"]) + np.arange(len(data)) / fs
    nT = len(template)
    tq = np.imag(hilbert(template))
    rho = np.hypot(np.correlate(data, template, "valid"), np.correlate(data, tq, "valid"))
    snr = rho / (data[t < -1.0].std() * np.sqrt(np.sum(template ** 2)))
    snr_t = t[np.arange(len(snr)) + nT - tail]
    data_d = decimate(data, D, zero_phase=True)
    tmpl_d = decimate(template, D, zero_phase=True)
    t_d = t[::D][:len(data_d)]
    snr_d, snrt_d = snr[::D], snr_t[::D]
    n = min(len(snr_d), len(data_d) - 1)
    snr_p = None
    if "pyc_t" in d.files and "pyc_snr" in d.files:
        snr_p = np.interp(snrt_d[:n], d["pyc_t"], d["pyc_snr"], left=0, right=0)
    r = lambda a, k=4: [round(float(x), k) for x in a]
    scan = dict(t=r(t_d, 4), data=r(data_d / np.abs(data_d).max(), 4), tmpl=r(tmpl_d / np.abs(tmpl_d).max(), 4),
                snrT=r(snrt_d[:n], 4), snrH=r(snr_d[:n], 2),
                snrP=None if snr_p is None else r(snr_p, 2), masses=[int(x) for x in d["masses"]])
    print(f"search data: {n} positions, hand-made peak SNR {snr.max():.1f}")
else:
    print("!! data/mf_animation_inputs.npz not found - the 'Find the signal' tab will show a notice")

# ---- 2. plots and audio -------------------------------------------------
gal = ""
for p in sorted((ROOT / "outputs" / "plots").glob("*.png")):
    cap = p.stem.replace("_", " ")
    gal += f'<figure><img alt="{cap}" src="data:image/png;base64,{b64(p)}"><figcaption>{cap}</figcaption></figure>'
aud = ""
for p in sorted((ROOT / "outputs" / "audio").glob("*.wav")) + sorted((ROOT / "outputs" / "animation").glob("*.wav")):
    aud += f'<div><b>{p.stem.replace("_", " ")}</b><audio controls preload="none" src="data:audio/wav;base64,{b64(p)}"></audio></div>'
if not gal: gal = "<p class='small'>No plots found in outputs/plots.</p>"
if not aud: aud = "<p class='small'>No audio found in outputs/audio.</p>"

html = TEMPLATE.read_text(encoding="utf-8")
html = html.replace("__SCAN_JSON__", json.dumps(scan, separators=(",", ":")))
html = html.replace("__GALLERY__", gal).replace("__AUDIO__", aud)
out = ROOT / "expo.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) - double-click it to open")
