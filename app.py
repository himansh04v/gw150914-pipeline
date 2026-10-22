"""
GW150914 Expo App (Streamlit)

Run from the repo root:   streamlit run app.py

Tabs
  1. Our results     : plots and numbers produced by the pipeline notebooks
  2. Match finder    : the template sliding over the data, with the SNR drawing itself
  3. Crash lab       : pick two masses, watch the merger, hear the chirp, read the horizon areas
  4. How we built it : the steps, the tools, and the key code
"""
import inspect
import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile
from scipy.signal import hilbert

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent
NPZ = ROOT / "data" / "mf_animation_inputs.npz"      # written by the export cell in the main notebook
PLOTS = ROOT / "outputs" / "plots"
AUDIO = ROOT / "outputs" / "audio"
ANIM = ROOT / "outputs" / "animation"

st.set_page_config(page_title="Black Hole Crash Lab", layout="wide")

# ------------------------------------------------------------------ physics (same formulas as the notebook)
GM = 1.4766                      # km, G * (1 solar mass) / c^2
SCH = 2 * GM                     # Schwarzschild radius per solar mass, km
R0, T_SIM = 130.0, 9.0           # starting separation and length of the cartoon inspiral
PX_PER_MASS = 0.42               # drawing size of a black hole per solar mass


def physics(m1, m2):
    """Areas, final mass and spin, and the timing numbers used by the animation and the sound."""
    M = m1 + m2
    eta = m1 * m2 / M**2
    chi = min(0.95, 2 * np.sqrt(3) * eta - 3.5171 * eta**2 + 2.5763 * eta**3)   # fitted final spin
    frac = 0.048 * (4 * eta) ** 2                                              # fitted energy fraction
    Mf = M * (1 - frac)
    A1 = 16 * np.pi * (m1 * GM) ** 2
    A2 = 16 * np.pi * (m2 * GM) ** 2
    Af = 8 * np.pi * (Mf * GM) ** 2 * (1 + np.sqrt(1 - chi**2))
    r1, r2 = m1 * PX_PER_MASS, m2 * PX_PER_MASS
    rm = r1 + r2                                                               # horizons touch
    rf = (PX_PER_MASS / SCH) * np.sqrt(Af / (4 * np.pi))                       # same area scale as the inputs
    xm = (rm / R0) ** 4
    return dict(M=M, eta=eta, chi=chi, Mf=Mf, dm=M - Mf, A1=A1, A2=A2, Af=Af,
                r1=r1, r2=r2, rm=rm, rf=rf, xm=xm, tauM=T_SIM * xm,
                w0=1.8 * np.sqrt(M / 65), fA0=70 * np.sqrt(65 / M))


def chirp_wav(P, sr=22050, dur=4.0, ring=0.7):
    """Synthesize the chirp: pitch rises as (time left)^(-3/8), then a short ringdown."""
    n = int(dur * sr)
    x = 1 - (1 - P["xm"]) * np.arange(n) / n                  # (time left) / (total time), from 1 down to xm
    f = P["fA0"] * x ** -0.375
    amp = 0.04 + 0.22 * np.clip(((1 / x) ** 0.25 - 1) / ((1 / P["xm"]) ** 0.25 - 1), 0, 1)
    ph = 2 * np.pi * np.cumsum(f) / sr                        # adding up frequency keeps the pitch smooth
    y = amp * (np.sin(ph) + 0.35 * np.sin(2 * ph))
    tr = np.arange(int(ring * sr)) / sr
    f_end = P["fA0"] * P["xm"] ** -0.375 * 0.95
    y_ring = 0.26 * np.exp(-tr / 0.2) * (np.sin(ph[-1] + 2 * np.pi * f_end * tr)
                                         + 0.35 * np.sin(2 * ph[-1] + 4 * np.pi * f_end * tr))
    y = np.r_[y, y_ring]
    y = 0.9 * y / np.abs(y).max()
    buf = io.BytesIO()
    wavfile.write(buf, sr, (y * 32767).astype(np.int16))
    return buf.getvalue()


def merger_figure(P, m1, m2, n_in=110, n_post=24):
    """Plotly animation: two black holes spiral in, merge, and the new one settles."""
    M = P["M"]
    u = np.linspace(0, 1, n_in)
    tau = P["tauM"] + (T_SIM - P["tauM"]) * (1 - u) ** 3     # more frames near the merger (cartoon time-warp)
    x = tau / T_SIM
    r = R0 * x**0.25
    phi = 0.6 + P["w0"] * T_SIM * (8 / 5) * (1 - x ** (5 / 8))
    k = 3.1                                                   # marker pixels per drawing unit

    def frame(x1, y1, x2, y2, s1, s2, xf, yf, sf, name):
        return go.Frame(name=name, data=[
            go.Scatter(x=[x1], y=[y1], mode="markers", marker=dict(size=s1, color="#4fd8d2", line=dict(color="#4fd8d2", width=2))),
            go.Scatter(x=[x2], y=[y2], mode="markers", marker=dict(size=s2, color="#ffb04a", line=dict(color="#ffb04a", width=2))),
            go.Scatter(x=[xf], y=[yf], mode="markers", marker=dict(size=sf, color="#c9a7ff", line=dict(color="white", width=2))),
        ])

    frames = []
    for i in range(n_in):
        c, s = np.cos(phi[i]), np.sin(phi[i])
        frames.append(frame(-r[i] * m2 / M * c, -r[i] * m2 / M * s, r[i] * m1 / M * c, r[i] * m1 / M * s,
                            2 * P["r1"] * k, 2 * P["r2"] * k, 0, 0, 0, f"in{i}"))
    for j in range(n_post):
        e = min(1, j / 8)
        size = P["rm"] + (P["rf"] - P["rm"]) * e
        size *= 1 + 0.2 * np.exp(-j / 5) * np.sin(j * 1.4)
        frames.append(frame(0, 0, 0, 0, 0, 0, 0, 0, 2 * size * k, f"post{j}"))

    first = frames[0].data
    fig = go.Figure(data=list(first), frames=frames)
    fig.update_layout(
        template="plotly_dark", height=520, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-170, 170], visible=False, fixedrange=True),
        yaxis=dict(range=[-170, 170], visible=False, scaleanchor="x", fixedrange=True),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, buttons=[
            dict(label="Play", method="animate",
                 args=[None, {"frame": {"duration": 45, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
        ])],
    )
    return fig


# ------------------------------------------------------------------ matched filter (same as the animation notebook)
@st.cache_data(show_spinner=False)
def load_scan():
    d = np.load(NPZ)
    data = d["data"].astype(float)
    template = d["template"].astype(float)
    fs = float(d["fs"])
    tail = int(round(float(d["tail_s"]) * fs))
    t = float(d["t_start"]) + np.arange(len(data)) / fs
    nT = len(template)

    tq = np.imag(hilbert(template))                                    # quarter-cycle shifted copy
    rho = np.hypot(np.correlate(data, template, "valid"), np.correlate(data, tq, "valid"))
    snr_hand = rho / (data[t < -1.0].std() * np.sqrt(np.sum(template**2)))
    snr_t = t[np.arange(len(snr_hand)) + nT - tail]
    snr_pyc = np.interp(snr_t, d["pyc_t"], d["pyc_snr"]) if "pyc_t" in d.files else None
    return dict(t=t, data=data, template=template, nT=nT, snr_t=snr_t,
                snr_hand=snr_hand, snr_pyc=snr_pyc, masses=d["masses"], fs=fs)


def scan_figure(S, snr, i, threshold):
    t, data, template, nT, snr_t = S["t"], S["data"], S["template"], S["nT"], S["snr_t"]
    scale = 3 * data.std() / np.abs(template).max()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.8))
    fig.subplots_adjust(hspace=0.55)
    a1.plot(t[::4], data[::4], color="0.55", lw=0.6)      # every 4th point is enough to draw and keeps frames fast
    a1.plot(t[i:i + nT], template * scale, color="darkorange", lw=2)
    a1.axvspan(t[i], t[i + nT - 1], color="orange", alpha=0.12)
    a1.set(xlim=(t[0], t[-1]), xlabel="time relative to merger (s)", title="Template (orange) sliding along the data")
    a2.plot(snr_t[:i + 1], snr[:i + 1], color="tab:blue")
    a2.plot([snr_t[i]], [snr[i]], "o", color="gold", ms=8)
    a2.axhline(threshold, ls="--", color="red", lw=1)
    a2.set(xlim=(snr_t[0], snr_t[-1]), ylim=(0, snr.max() * 1.2), xlabel="time of merger in template (s)",
           ylabel="SNR", title="Match score so far")
    best = snr[:i + 1].max()
    found = best >= threshold
    a2.text(0.02, 0.88, f"SIGNAL FOUND!  SNR = {best:.1f}" if found else f"Searching...  best so far: {best:.1f}",
            transform=a2.transAxes, fontsize=12, fontweight="bold", color="green" if found else "black")
    return fig


def show_plot(name, caption):
    p = PLOTS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.caption(f"(missing: outputs/plots/{name})")


# ------------------------------------------------------------------ page
st.title("Black Hole Crash Lab")
st.write("Real LIGO data of GW150914, searched with our own code. Explore the results, "
         "watch the search, and crash your own black holes.")

tab_res, tab_scan, tab_lab, tab_how = st.tabs(["Our results", "Match finder", "Crash lab", "How we built it"])

# ---------------------------------------------------------------- 1. results
with tab_res:
    st.subheader("From noise to a detection")
    c1, c2 = st.columns(2)
    with c1:
        show_plot("01_raw_strain.png", "Before: the raw detector data, signal buried in noise")
        show_plot("03_noise_spectrum.png", "How noisy each detector is at each pitch")
    with c2:
        show_plot("04_snr.png", "After: the matched filter finds a clear spike in both detectors")
        show_plot("05_chirp.png", "The chirp: pitch rising as the black holes spiral together")
    show_plot("06_template_bank.png", "Which template masses fit best (red star: published masses in the detector frame)")
    show_plot("07_event_horizons.png", "Horizon area before and after the merger")

    st.subheader("Ours against the published values")
    st.table({
        "Quantity": ["Combined signal strength (SNR)", "Chirp mass, detector frame", "Delay between detectors",
                     "Horizon area growth", "Mass turned into waves"],
        "Our result": ["23.8  (H1 19.4, L1 13.8)", "31.0  (range 29.5 to 32.6)", "about 8 ms",
                       "x1.56", "3.1 solar masses"],
        "Published (LIGO)": ["about 24", "about 31 (28.6 x 1.09)", "6.9 ms", "area never decreases", "about 3 solar masses"],
    })
    st.caption("Published values: Abbott et al. 2016, PRL 116, 061102, and the GWTC-1 catalog (2019).")

# ---------------------------------------------------------------- 2. match finder
with tab_scan:
    st.subheader("Watch the search")
    st.write("A template (the predicted chirp) slides along the data. At each position we measure how well "
             "they match. That match score is the SNR. A big jump means the signal is there.")
    if not NPZ.exists():
        st.warning("The data file is missing. Run the export cell in the main notebook "
                   "(it saves `data/mf_animation_inputs.npz`), then reload this page.")
    else:
        S = load_scan()
        opts = ["Our PyCBC result", "Hand-written filter"] if S["snr_pyc"] is not None else ["Hand-written filter"]
        src = st.radio("SNR curve from", opts, horizontal=True)
        snr = S["snr_pyc"] if src == "Our PyCBC result" else S["snr_hand"]
        peak_i = int(np.argmax(snr))
        n = len(snr)

        left, right = st.columns([1, 1])
        threshold = left.slider("Detection threshold (SNR)", 4.0, 12.0, 8.0, 0.5)
        play_speed = right.select_slider("Play speed", ["slow", "medium", "fast"], value="medium")
        st.session_state.setdefault("pos", 0)

        def jump():
            st.session_state.pos = peak_i

        b1, b2, _ = st.columns([1, 1, 3])
        play = b1.button("Play the search", type="primary")
        b2.button("Jump to the peak", on_click=jump)
        pos = st.slider("Template position (drag it yourself)", 0, n - 1, key="pos")
        holder = st.empty()

        if play:
            step = {"slow": 60, "medium": 110, "fast": 220}[play_speed]
            frames = sorted(set(range(0, n, step)) | {peak_i})
            for i in frames:
                fig = scan_figure(S, snr, i, threshold)
                holder.pyplot(fig)
                plt.close(fig)
                time.sleep(1.2 if i == peak_i else 0.02)
        else:
            fig = scan_figure(S, snr, pos, threshold)
            holder.pyplot(fig)
            plt.close(fig)

        st.metric("Peak SNR in this search", f"{snr.max():.1f}",
                  help="PyCBC's own value for the same template is the 'Our PyCBC result' curve.")
        st.caption(f"Template masses used (detector frame): {int(S['masses'][0])} + {int(S['masses'][1])} solar masses. "
                   "The data is the public LIGO recording of GW150914, cleaned by our pipeline.")

    with st.expander("Pre-made video and sound"):
        vid, wav = ANIM / "matched_filter_with_sound.mp4", ANIM / "scan_sound.wav"
        if vid.exists():
            st.video(str(vid))
        if wav.exists():
            st.audio(str(wav))
        if not vid.exists() and not wav.exists():
            st.caption("Run the animation notebook to create these files in outputs/animation/.")

    with st.expander("Sonified data and model"):
        for f, label in [("gw150914_H1_single.wav", "1. Raw data from one detector"),
                         ("gw150914_combined_enhanced.wav", "2. Both detectors combined, background reduced"),
                         ("gw150914_model_clean.wav", "3. Model: the template the search looks for")]:
            if (AUDIO / f).exists():
                st.write(label)
                st.audio(str(AUDIO / f))

# ---------------------------------------------------------------- 3. crash lab
with tab_lab:
    st.subheader("Crash your own black holes")
    PRESETS = {"Real one: GW150914": (35.6, 30.6), "Twins": (30.0, 30.0),
               "Big and small": (55.0, 12.0), "Light pair": (12.0, 10.0)}
    st.session_state.setdefault("m1", 35.6)
    st.session_state.setdefault("m2", 30.6)

    def set_preset():
        st.session_state.m1, st.session_state.m2 = PRESETS[st.session_state.preset]

    st.selectbox("Preset", list(PRESETS), key="preset", on_change=set_preset)
    sa, sb = st.columns(2)
    m1 = sa.slider("Black hole 1 (solar masses)", 5.0, 60.0, step=0.1, key="m1")
    m2 = sb.slider("Black hole 2 (solar masses)", 5.0, 60.0, step=0.1, key="m2")
    P = physics(m1, m2)

    col_anim, col_num = st.columns([3, 2])
    with col_anim:
        st.plotly_chart(merger_figure(P, m1, m2), width="stretch")
        st.write("Press **Play** on the picture, and press play on the sound below at the same time.")
        st.audio(chirp_wav(P), format="audio/wav")
        st.caption("The pitch rises because the black holes circle faster as they get closer. Heavier black holes sound lower.")
    with col_num:
        ratio = P["Af"] / (P["A1"] + P["A2"])
        st.metric("Surface area grows by", f"x{ratio:.2f}")
        st.write(f"**Radius of each hole:** {m1 * SCH:.1f} km and {m2 * SCH:.1f} km")
        st.write(f"**Area before (added up):** {P['A1'] + P['A2']:,.0f} km²")
        st.write(f"**Area after:** {P['Af']:,.0f} km²")
        st.write(f"**New black hole:** {P['Mf']:.1f} solar masses, spin {P['chi']:.2f}")
        joules = P["dm"] * 1.989e30 * 2.998e8**2
        st.write(f"**Mass turned into ripples:** {P['dm']:.1f} solar masses ({joules:.1e} J)")
        st.success("Total horizon area never decreases. That is Hawking's area rule.")
        st.caption("The final mass, spin and energy come from fitted formulas, so they are approximate. "
                   "For the real masses they land close to the published values. "
                   "The orbit shape is a cartoon and is not to scale.")

# ---------------------------------------------------------------- 4. how
with tab_how:
    st.subheader("How we built it")
    st.markdown(
        """
1. **Data:** the public LIGO recording of GW150914 from GWOSC (H1 and L1), loaded with PyCBC.
2. **Cleaning:** a high-pass filter at 15 Hz, resampling to 2048 Hz, and cropping the edges.
3. **Noise spectrum (PSD):** how loud the detector is at each pitch.
4. **Template bank:** about 200 mass pairs tried, each matched against the data. The best-fitting
   templates fixed the chirp mass at about 31 (detector frame), not the two masses separately.
5. **Matched filter:** the chosen template slid along the data, giving the SNR curve.
6. **Whitening and Q-transform:** flatten the noise so the rising chirp becomes visible.
7. **Sonification:** the whitened data and the template turned into sound files.
8. **Event horizons:** areas from the published masses and spin, checked against Hawking's area rule.
        """
    )
    st.table({
        "Tool": ["Python 3.10 (WSL2 + Conda)", "PyCBC / GWpy", "NumPy / SciPy", "Matplotlib / Plotly",
                 "Jupyter", "Streamlit", "Git + GitHub"],
        "Used for": ["The environment on Windows", "LIGO data, templates, matched filter", "Arrays, correlation, audio files",
                     "Plots and animations", "Notebooks that explain each step", "This app", "History and backup"],
    })
    st.subheader("Key code from this app")
    for fn in (physics, chirp_wav, scan_figure):
        with st.expander(f"{fn.__name__}()"):
            st.code(inspect.getsource(fn), language="python")
    with st.expander("The whole app.py"):
        st.code(Path(__file__).read_text(), language="python")
