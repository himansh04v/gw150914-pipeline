# When Black Holes Collide: Code, Listen, and Calculate

A Python reproduction of the first gravitational-wave detection, **GW150914**, from public LIGO data.
We clean the data, search for the black holes with matched filtering, turn the signal into sound, and work out the sizes and surface areas of the black holes involved.

> This project **reproduces a known detection** with the same method LIGO uses. It is not a new discovery.

*Built for our college science fest expo. Team: **Name Surname, Name Surname, Name Surname**.*

---

## What is GW150914?

On 14 September 2015, the two LIGO detectors (Hanford, H1, and Livingston, L1) recorded a tiny ripple in space-time from two black holes of about 36 and 30 Suns merging roughly 1.3 billion light-years away. It was the first direct detection of gravitational waves.

## What this project does

| Step | What we do | Main tools |
|---|---|---|
| 1. Get the data | Download public strain data for H1 and L1 around the event (with a local-copy fallback if the site is down) | Python, `gwosc` |
| 2. Prepare it | 15 Hz high-pass filter, resample to 2048 Hz | SciPy, PyCBC |
| 3. Measure the noise | Estimate the noise spectrum (PSD) for each detector | PyCBC |
| 4. Whiten and filter | Divide out the noise level, keep 30 to 350 Hz | NumPy, SciPy |
| 5. See the chirp | Q-transform (time-frequency) plot | Matplotlib |
| 6. Search the masses | Template-bank search over mass pairs (m1 30 to 50, m2 24 to 44 Suns) | PyCBC |
| 7. Match and score | Matched filtering: SNR for H1 and L1, plus our own hand-written NumPy version | PyCBC, NumPy |
| 8. Listen and calculate | Three audio clips (raw, enhanced, model), event-horizon sizes and areas, Hawking area-theorem check | NumPy, SciPy |

## Results

| Quantity | Our result | Published |
|---|---|---|
| Peak SNR, H1 / L1 | 19.4 / 13.8 | about 24 combined |
| Combined SNR | about 24 | about 24 |
| Best template (detector frame) | about 44 + 29 Suns | 35.6 + 30.6 Suns source frame, times 1.09 for redshift |
| Chirp mass (detector frame) | about 31 Suns (range 29.5 to 32.6) | about 28.6 source frame, about 31 detector frame |
| Horizon area before / after | 241,519 km² / 376,109 km² (× 1.56) | area never decreases (Hawking) |
| Mass turned into waves | about 3 Suns | about 3 Suns |

**Honest notes**
- The data fixes the **chirp mass** well, but not each mass alone. Many mass pairs along a ridge fit almost equally well, so 44 + 29 is one good fit, not a unique answer. LIGO uses a full Bayesian analysis with spin to narrow this down.
- Detector-frame masses are source-frame masses × (1 + z), with z about 0.09.
- The horizon-area check uses the published masses and final spin.
- Please verify published values against the papers listed under [References](#references).

## Repository layout

> Adjust this to match your actual folders.

```
.
├── README.md
├── requirements.txt              # core pipeline dependencies
├── requirements-app.txt          # extra dependencies for the Streamlit app
├── data/                         # downloaded LIGO data (not committed, see .gitignore)
│   └── mf_animation_inputs.npz   # saved inputs for the animation and expo page
├── notebooks/
│   ├── 01_pipeline.ipynb                 # full analysis, start to finish
│   └── 02_matched_filter_animation.ipynb # template sliding along the data, with sound
├── outputs/
│   ├── plots/                    # figures used in the report and posters
│   ├── audio/                    # raw, enhanced and model .wav files
│   └── animation/                # animation videos
├── app.py                        # Streamlit app
├── .streamlit/config.toml
├── make_expo_page.py             # builds expo.html from the saved results
└── expo_template.html
```

## Getting started

### 1. Set up the environment
We used Windows 11 with WSL2 (Ubuntu), Miniconda and Python 3.10. PyCBC does not install cleanly on the newest Python versions, so use 3.10.

```bash
conda create -n ligo python=3.10 -y
conda activate ligo
sudo apt install build-essential        # PyCBC needs a C/C++ compiler
pip install -r requirements.txt
pip install "setuptools<81"             # PyCBC still imports pkg_resources
```

### 2. Run the main analysis
```bash
jupyter lab
```
Open `notebooks/01_pipeline.ipynb` and run the cells in order. The first run downloads the data from the LIGO Open Science Center (internet needed). The data is cached in `data/`, and the notebook uses the local copy when the download fails.

### 3. Run the animation notebook
Open `notebooks/02_matched_filter_animation.ipynb`. It loads `data/mf_animation_inputs.npz`, slides the template along the data, shows the SNR peak, and can play the matching sound. It needs `ffmpeg` for the video with sound.

### 4. Run the Streamlit app (optional)
```bash
pip install -r requirements-app.txt
streamlit run app.py
```
Tabs: Our results, Match finder, Crash lab, How we built it. You need to run `streamlit run app.py` each time you want to open it.

### 5. Build the single-page demo (optional)
```bash
python make_expo_page.py
```
This creates `expo.html`, one self-contained file you can open by double-clicking. It has three tabs:
- **Find the signal:** the template slides along the data, and the sound is driven by the same animation loop so it stays in step with the picture.
- **Crash lab:** pick two masses, watch a simulated merger, and see the horizon areas and energy radiated. It can play either a made-up chirp or our real sonification clips.
- **Our results:** our plots, our audio clips and a published-vs-ours table.

## The physics in brief

- **Chirp mass:** M_c = (m1 m2)^(3/5) / (m1 + m2)^(1/5), the combination the signal measures best.
- **Inspiral chirp:** df/dt = (96/5) π^(8/3) (G M_c / c³)^(5/3) f^(11/3), so the pitch climbs as the holes spiral in.
- **Matched filter:** ρ = ⟨d|h⟩ / √⟨h|h⟩, with ⟨a|b⟩ = 4 Re ∫ ã(f) b̃*(f) / S_n(f) df, where d is the data, h the template and S_n the noise level.
- **Event horizon:** r_s = 2GM / c². For a spinning hole, A = 8π (GM/c²)² (1 + √(1 − χ²)).
- **Hawking area theorem:** A_final ≥ A1 + A2.
- **Energy radiated:** E = Δm c².

The template waveform comes from PyCBC's waveform models (inspiral, merger and ringdown, fitted to numerical-relativity simulations of Einstein's equations).

## Sonification

We made three audio clips of the event so it can be heard:
1. **Raw:** the whitened data, shifted into the audible range.
2. **Enhanced:** the same data with less noise and a clearer chirp.
3. **Model:** a clean chirp synthesised from our template's amplitude and phase.

The clips are time-stretched so the short chirp is easy to hear. They are for explanation, not measurement.

## Data and credits

- **Data:** LIGO Open Science Center, [gwosc.org](https://gwosc.org). LIGO data is public.
- **Libraries:** [PyCBC](https://pycbc.org), NumPy, SciPy, Matplotlib, [GWOSC client](https://gwosc.readthedocs.io).
- **Images used in our posters** (not stored in this repo) are credited on the posters: LIGO Laboratory, SXS Collaboration, and the Event Horizon Telescope Collaboration.

## References

- B. P. Abbott et al. (LIGO Scientific and Virgo Collaborations), *Observation of Gravitational Waves from a Binary Black Hole Merger*, Phys. Rev. Lett. **116**, 061102 (2016).
- B. P. Abbott et al., *Properties of the Binary Black Hole Merger GW150914*, Phys. Rev. Lett. **116**, 241102 (2016).
- B. P. Abbott et al., *GWTC-1: A Gravitational-Wave Transient Catalog*, Phys. Rev. X **9**, 031040 (2019).
- M. Isi et al., *Testing the Black-Hole Area Law with GW150914*, Phys. Rev. Lett. **127**, 011103 (2021).

## Limitations and next steps

- The search uses a simple grid of masses and does not fit spin.
- Masses are not given with proper uncertainties. A simple Bayesian fit and corner plot would be the natural next step.
- We reproduced one event only. The same pipeline could be tried on other public events.
