import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import spaces

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")

song_pool = pd.read_csv(os.path.join(DATA_PATH, "song_pool_predictions.csv"))
song_index = pd.read_csv(os.path.join(DATA_PATH, "deam_full_with_ids.csv"), usecols=["song_id"])
trained_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
shap_v = np.load(os.path.join(DATA_PATH, "shap_valence.npy"))
shap_a = np.load(os.path.join(DATA_PATH, "shap_arousal.npy"))

FEATURE_GLOSSARY = {
    "audspec_lengthL1norm": "overall loudness",
    "audspecRasta_lengthL1norm": "perceived loudness",
    "pcm_RMSenergy": "signal energy",
    "pcm_zcr": "noisiness",
    "spectralVariance": "spread of frequencies",
    "spectralHarmonicity": "tonal clarity",
    "spectralKurtosis": "peakiness of the spectrum",
    "spectralCentroid": "brightness",
    "spectralRollOff": "high-frequency content",
    "spectralFlux": "how fast the sound changes",
    "spectralEntropy": "spectral complexity",
    "spectralSkewness": "spectral balance",
    "fband": "energy in a frequency band",
    "mfcc": "timbre / tone colour",
    "audSpec_Rfilt": "energy in a hearing band",
    "F0final": "pitch",
    "voicingFinalUnclipped": "how tonal vs noisy",
    "jitter": "pitch instability",
    "shimmer": "loudness instability",
    "logHNR": "clarity vs breathiness",
    "spectralSlope": "spectral slope (bass vs treble balance)",
}

ddef humanise(name):
    for key, desc in FEATURE_GLOSSARY.items():
        if key.lower() in name.lower():
            prefix = "variation in " if "_std" in name else ""
            label = prefix + desc
            m = re.search(r"sma[_\[]?(?:de_)?(\d+)", name)
            if m:
                label += f" #{m.group(1)}"
            return label
    return name
    
def infer_mood(ids):
    lis = song_pool[song_pool["song_id"].isin(ids)]
    return {
        "valence": lis["pred_valence"].mean(),
        "arousal": lis["pred_arousal"].mean(),
        "spread": np.sqrt(lis["pred_valence"].var() + lis["pred_arousal"].var()),
        "n": len(lis),
    }


def label_quadrant(v, a, threshold=0.15):
    if abs(v) < threshold and abs(a) < threshold:
        return "Neutral / Ambiguous"
    if v >= 0 and a >= 0:
        return "Excited / Elated"
    if v < 0 and a >= 0:
        return "Angry / Stressed"
    if v < 0 and a < 0:
        return "Sad / Depressed"
    return "Calm / Content"


def rec_songs(v, a, n=5, exclude_ids=None):
    pool = song_pool.copy()
    if exclude_ids:
        pool = pool[~pool["song_id"].isin(exclude_ids)]
    pool["dist"] = np.sqrt((pool["pred_valence"] - v) ** 2 + (pool["pred_arousal"] - a) ** 2)
    return pool.sort_values("dist").head(n).reset_index(drop=True)


def circumplex_fig(mood, listened, recommended):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(song_pool["pred_valence"], song_pool["pred_arousal"],
               alpha=0.12, color="grey", s=12)
    lis = song_pool[song_pool["song_id"].isin(listened)]
    ax.scatter(lis["pred_valence"], lis["pred_arousal"],
               color="steelblue", s=70, label="your listens")
    rec = song_pool[song_pool["song_id"].isin(recommended)]
    ax.scatter(rec["pred_valence"], rec["pred_arousal"],
               color="seagreen", s=70, marker="s", label="recommended")
    ax.scatter(mood["valence"], mood["arousal"], color="crimson",
               marker="*", s=400, edgecolors="black", label="your mood")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.axvline(0, color="black", ls="--", lw=0.8)

    ax.text(0.55, 0.72, "energetic\n& positive", fontsize=8, ha="center",
            color="dimgrey", style="italic")
    ax.text(-0.55, 0.72, "energetic\n& tense", fontsize=8, ha="center",
            color="dimgrey", style="italic")
    ax.text(-0.55, -0.72, "low energy\n& negative", fontsize=8, ha="center",
            color="dimgrey", style="italic")
    ax.text(0.55, -0.72, "calm\n& positive", fontsize=8, ha="center",
            color="dimgrey", style="italic")

    ax.set_xlabel("Valence — how positive the music feels")
    ax.set_ylabel("Arousal — how energetic the music feels")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    return fig


def shap_fig(song_id, top_n=8):
    i = song_index.index[song_index["song_id"] == song_id][0]
    row = song_pool[song_pool["song_id"] == song_id].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (name, sv, pred) in zip(
        axes,
        [("Valence", shap_v, row["pred_valence"]),
         ("Arousal", shap_a, row["pred_arousal"])]
    ):
        c = sv[i]
        order = np.argsort(np.abs(c))[::-1][:top_n][::-1]
        vals = c[order]
        ax.barh(range(len(vals)), vals,
                color=["seagreen" if v > 0 else "crimson" for v in vals])
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([humanise(trained_cols[j]) for j in order], fontsize=8)        
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{name} = {pred:.3f}", fontsize=10)
        ax.grid(axis="x", ls=":", alpha=0.4)

    fig.suptitle(f"{row['title']} - {row['Artist']}", fontsize=11)
    plt.tight_layout()
    return fig


CHOICES = [f"{r.title} - {r.Artist} ({r.Genre}) [{r.song_id}]" for r in song_pool.itertuples()]
TO_ID = {c: int(c.rsplit("[", 1)[1].rstrip("]")) for c in CHOICES}

def explain_text(song_id):
    i = song_index.index[song_index["song_id"] == song_id][0]
    row = song_pool[song_pool["song_id"] == song_id].iloc[0]

    def top_drivers(sv, n=2):
        c = sv[i]
        order = np.argsort(np.abs(c))[::-1][:n]
        return [(humanise(trained_cols[j]), "raised" if c[j] > 0 else "lowered") for j in order]

    v_drivers = top_drivers(shap_v)
    a_drivers = top_drivers(shap_a)

    return (
        f"**Why '{row['title']}' was suggested**\n\n"
        f"How positive it sounds ({row['pred_valence']:.2f}) was mostly shaped by "
        f"{v_drivers[0][0]}, which {v_drivers[0][1]} it, and {v_drivers[1][0]}, "
        f"which {v_drivers[1][1]} it.\n\n"
        f"How energetic it sounds ({row['pred_arousal']:.2f}) came mainly from "
        f"{a_drivers[0][0]}, which {a_drivers[0][1]} it, and {a_drivers[1][0]}, "
        f"which {a_drivers[1][1]} it.\n\n"
        f"Together these place it close to where your listening sits on the map."
    )

@spaces.GPU
def analyse(selected, n_recs):
    if not selected:
        return None, "Select at least one song.", pd.DataFrame(), None

    ids = [TO_ID[s] for s in selected]
    mood = infer_mood(ids)
    coherence = "consistent" if mood["spread"] < 0.35 else "mixed - estimate less certain"
    summary = (
        f"### {label_quadrant(mood['valence'], mood['arousal'])}\n"
        f"valence **{mood['valence']:.3f}** | arousal **{mood['arousal']:.3f}**\n\n"
        f"{mood['n']} songs | session is {coherence} (spread {mood['spread']:.3f})"
    )

    recs = rec_songs(mood["valence"], mood["arousal"], n=int(n_recs), exclude_ids=ids)
    table = recs[["title", "Artist", "Genre", "pred_valence", "pred_arousal", "dist"]].round(3)

    return (circumplex_fig(mood, ids, recs["song_id"].tolist()),
            summary,
            table,
            int(recs.iloc[0]["song_id"]))


def build_ui():
    with gr.Blocks(title="Emotion-Aware Music Recommender") as demo:
        gr.Markdown("# Emotion-aware music recommender")
        gr.Markdown(
            "Pick a few songs you've been listening to. Every song sits somewhere "
            "on the map: left to right is how positive it feels, bottom to top is "
            "how energetic. The star shows where your listening puts you, and the "
            "green squares are what we'd suggest next."
        )

        with gr.Row():
            with gr.Column():
                sel = gr.Dropdown(CHOICES, multiselect=True,
                                  label="Your recent listens", max_choices=8)
                n = gr.Slider(3, 10, value=5, step=1, label="Recommendations")
                btn = gr.Button("Analyse my mood", variant="primary")
                mood_md = gr.Markdown()
            with gr.Column():
                plot = gr.Plot(label="Circumplex space")

        gr.Markdown("### Recommended for you")
        table = gr.Dataframe()

        top = gr.State()
        xai_btn = gr.Button("Explain top recommendation")
        xai_md = gr.Markdown()
        xai = gr.Plot()

        btn.click(analyse, [sel, n], [plot, mood_md, table, top])
        xai_btn.click(lambda s: explain_text(int(s)) if s else "", [top], [xai_md])
        xai_btn.click(lambda s: shap_fig(int(s)) if s else None, [top], [xai])

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
