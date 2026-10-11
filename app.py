"""Streamlit MVP: upload a contact-lens image -> Good / Defective (+ defect type)."""

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from lens_classifier import CATEGORIES, MODEL, classify, load_references, preprocess

load_dotenv()
st.set_page_config(page_title="Lens Inspector", page_icon="🔍", layout="wide")

VERDICT_STYLE = {
    "GOOD": ("✅ Good lens", st.success),
    "DEFECTIVE": ("❌ Defective lens", st.error),
    "REVIEW": ("⚠️ Needs manual review", st.warning),
    "NOT_A_LENS": ("🚫 Not a contact lens", st.info),
}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    api_key = os.getenv("OPENAI_API_KEY") or st.text_input("OpenAI API key", type="password")
    st.caption(f"Model: `{MODEL}` (Decisions API)")
    detail = st.selectbox("Image detail", ["high", "original", "auto", "low"],
                          help="Defects are tiny — 'high' or 'original' is recommended.")
    enhance = st.toggle("Auto-contrast (match dataset preprocessing)", value=True)
    use_refs = st.toggle("Send reference examples (few-shot)", value=True,
                         help="Uses images in ./references/<category>/")
    st.subheader("Decision thresholds")
    defect_threshold = st.slider("Defect probability threshold", 0.1, 0.9, 0.5, 0.05)
    review_band = st.slider("Manual-review band (±)", 0.0, 0.3, 0.15, 0.05)
    min_conf = st.slider("Min. category confidence", 0.0, 1.0, 0.5, 0.05)


@st.cache_resource
def get_refs():
    return load_references(os.path.join(os.path.dirname(__file__), "references"))


# ---------------- Main ----------------
st.title("🔍 Contact Lens Defect Inspector")
st.write("Upload a raw lens image. The OpenAI Decisions API decides whether it is **good**, "
         "**defective** (Scratch, Collapse, Digs, Glass Stick, Internal Crack) or needs review.")

uploaded = st.file_uploader("Lens image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if not uploaded:
    st.stop()

image = Image.open(uploaded)
col_img, col_res = st.columns([1, 1])
with col_img:
    tab_raw, tab_proc = st.tabs(["Uploaded", "Sent to model"])
    tab_raw.image(image, width="stretch")
    tab_proc.image(preprocess(image, enhance), width="stretch")

if not api_key:
    col_res.warning("Add an OpenAI API key in the sidebar or a .env file.")
    st.stop()

refs = get_refs() if use_refs else []
if use_refs and not refs:
    col_res.info("No reference images found in ./references — running zero-shot.")

with col_res, st.spinner("Inspecting lens..."):
    try:
        result = classify(image, OpenAI(api_key=api_key), references=refs, detail=detail,
                          enhance=enhance, defect_threshold=defect_threshold,
                          review_band=review_band, min_confidence=min_conf)
    except Exception as e:  # surface API errors in the UI
        st.error(f"API call failed: {e}")
        st.stop()

with col_res:
    title, box = VERDICT_STYLE[result.verdict]
    box(f"### {title}\n{result.reason}")

    m1, m2, m3 = st.columns(3)
    if result.defect_prob is not None:
        m1.metric("Defect probability", f"{result.defect_prob:.0%}")
    if result.defect_type:
        m2.metric("Category", CATEGORIES.get(result.defect_type, (result.defect_type,))[0],
                  f"{result.defect_confidence:.0%} confidence", delta_color="off")
    if result.severity_label:
        m3.metric("Severity", result.severity_label, f"score {result.severity_score:.2f} / 3",
                  delta_color="off")

    if result.defect_probs:
        st.caption("Category probabilities")
        st.bar_chart({CATEGORIES.get(k, (k,))[0]: v for k, v in result.defect_probs.items()},
                     horizontal=True)

    st.caption(f"Input tokens: {result.input_tokens:,} · est. cost ${result.cost_usd:.5f}"
               + (f" · is-lens prob {result.is_lens_prob:.0%}" if result.is_lens_prob is not None else ""))
    with st.expander("Raw API response"):
        st.json(result.raw)
