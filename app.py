import streamlit as st
import tempfile
import os
import pandas as pd  # ✅ per la tabella

from model_utils import load_models, discover_model_paths
from video_utils import process_video

# --- MUST be the first Streamlit call ---
st.set_page_config(page_title="Military AI Stream — Auto", layout="wide")

# --- Minimal, modern, canna di fucile visuals (no logic changes) ---
st.markdown("""
<style>
:root{
  --bg:#2a3439;       /* canna di fucile */
  --bg-2:#1f2327;
  --txt:#E8E8E8;
  --txt-dim:#B9C1C8;
  --btn:#3A3F45;      /* button grey */
  --btn-hov:#4A5057;
  --white:#FFFFFF;
}
html, body, .stApp { background: var(--bg) !important; color: var(--txt) !important; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
h1,h2,h3,h4,h5 { color: var(--white) !important; font-weight: 700; letter-spacing:.2px; }
p, label, span, li, .stMarkdown { color: var(--txt) !important; }
small, .stCaption { color: var(--txt-dim) !important; }

.stButton > button{
  background: var(--btn) !important;
  color: var(--white) !important;
  border: 1px solid #3F444A !important;
  border-radius: 10px !important;
  padding: .45rem .9rem !important;
  font-size: .9rem !important;
  box-shadow: none !important;
}
.stButton > button:hover{ background: var(--btn-hov) !important; }

[data-testid="stFileUploaderDropzone"]{
  background: var(--bg-2) !important;
  border: 1px dashed #3c4349 !important;
  color: var(--txt-dim) !important;
  border-radius: 12px !important;
}

/* WHITE confidence/progress bars in Streamlit */
[data-testid="stProgressBar"]{ background: #ffffff22 !important; border-radius: 8px !important; }
[data-testid="stProgressBar"] > div[role="progressbar"]{
  background: var(--white) !important; border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Military AI Stream — Auto Discovery")

@st.cache_resource(show_spinner=False)
def _load_models_cached(mpath=None, tpath=None, ipath=None):
    models = load_models(mpath, tpath, ipath)
    return models  # (mobilenet, cnn_tank, cnn_ifv, paths)

# ====== Auto-discovery all'avvio ======
with st.spinner("🔎 Cerco i modelli automaticamente..."):
    try:
        discovered = discover_model_paths()
        mobilenet, cnn_tank, cnn_ifv, paths = _load_models_cached(
            discovered["mobilenet"], discovered["cnn_tank"], discovered["cnn_ifv"]
        )
        st.success("Modelli caricati automaticamente ✅")
        st.caption(
            f"MobileNet: `{os.path.basename(paths['mobilenet'])}` | "
            f"Threat Tank: `{os.path.basename(paths['cnn_tank'])}` | "
            f"Threat IFV: `{os.path.basename(paths['cnn_ifv'])}`"
        )
    except Exception as e:
        st.error(f"Auto-discovery fallita: {e}")
        st.stop()

with st.expander("⚙️ Avanzate (override manuale)", expanded=False):
    m_override = st.text_input("Percorso MobileNet", value=paths["mobilenet"])
    t_override = st.text_input("Percorso Threat Tank", value=paths["cnn_tank"])
    i_override = st.text_input("Percorso Threat IFV", value=paths["cnn_ifv"])
    if st.button("Ricarica con override"):
        with st.spinner("Ricarico modelli..."):
            mobilenet, cnn_tank, cnn_ifv, paths = _load_models_cached(m_override, t_override, i_override)
        st.success("Ricaricati ✅")
        st.experimental_rerun()

st.divider()
st.header("📹 Video di input")
uploaded = st.file_uploader("Carica un video (mp4, mov, avi)", type=["mp4", "mov", "avi"])
process_btn = st.button("▶️ Processa")

if process_btn:
    if not uploaded:
        st.error("Carica un file video.")
    else:
        with st.spinner("Elaborazione in corso..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                    tmp_in.write(uploaded.read())
                    input_path = tmp_in.name

                output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
                out_path, report = process_video(input_path, mobilenet, cnn_tank, cnn_ifv, output_path=output_path)

                st.success("Video processato ✅")
                st.video(out_path)

                # 📊 Report in tabella
                st.subheader("📊 Report finale")
                df_report = pd.DataFrame([
                    {"Metric": "Total Frames", "Value": report.get("total_frames", 0)},
                    *(
                        {"Metric": f"Equipment Count - {k}", "Value": v}
                        for k, v in report.get("equipment_counts", {}).items()
                    ),
                    *(
                        {"Metric": f"Threat Count - {k}", "Value": v}
                        for k, v in report.get("threat_counts", {}).items()
                    ),
                    *(
                        {"Metric": f"Avg Threat Prob - {k}", "Value": round(v, 3)}
                        for k, v in report.get("avg_threat_probs", {}).items()
                    )
                ])
                st.table(df_report)

                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Scarica video annotato", f, file_name="output_annotato.mp4", mime="video/mp4")
                st.caption("Nota: se i livelli 'Medium' risultano sempre 0, riallena i threat a 3 classi.")
            except Exception as e:
                st.error(f"Errore durante l'elaborazione: {e}")
