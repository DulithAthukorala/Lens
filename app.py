from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LENS", page_icon="🔍", layout="centered")
st.title("🔍 LENS")
st.caption("Multi-agent RAG pitch pipeline — paste a prospect URL and get a validated cold outreach pitch.")

url = st.text_input("Prospect URL", placeholder="https://example.com")

if st.button("Generate Pitch", type="primary", disabled=not url):

    # Auto-ingest KB if missing
    with st.spinner("Checking knowledge base..."):
        import chromadb
        client = chromadb.PersistentClient(path="chroma_data")
        collections = [c.name for c in client.list_collections()]
        if "lens_case_studies_sections" not in collections:
            from src.ingestion.kb_builder import ingest
            ingest("sections")
            ingest("sentences")
            ingest("fixed512")

    from src.graph import run_pipeline

    with st.spinner("Researching prospect website..."):
        result = run_pipeline(url if url.startswith("http") else f"https://{url}")

    ro = result.get("researcher_output")
    ao = result.get("analyst_output")
    wo = result.get("writer_output")
    vo = result.get("validator_output")

    # --- Signals ---
    if ro and ro.signals:
        st.subheader("Detected Failure Signals")
        signals_data = [
            {
                "Tier": s.tier.value,
                "Signal": s.signal_name,
                "Evidence": s.evidence_value,
                "Confidence": f"{s.confidence:.0%}",
            }
            for s in sorted(ro.signals, key=lambda x: x.tier.value)
        ]
        st.dataframe(signals_data, use_container_width=True)
    else:
        st.warning("No failure signals detected.")

    # --- Case study match ---
    if ao and ao.top_matches:
        st.subheader("Top Case Study Match")
        top = ao.top_matches[0]
        st.metric("Similarity Score", f"{top.similarity_score:.3f}")
        st.caption(f"**{top.case_study_id}** — {top.match_rationale}")

    # --- Pitch ---
    if wo:
        st.subheader("Generated Pitch")
        pitch = wo.pitch
        st.markdown(f"**Subject:** {pitch.subject_line}")
        st.markdown(f"**Opening:** {pitch.opening_hook}")
        st.markdown(f"**Pain framing:** {pitch.pain_framing}")
        st.markdown(f"**Proof:** {pitch.proof_paragraph}")
        st.markdown(f"**CTA:** {pitch.cta}")

        with st.expander("Raw JSON"):
            st.code(json.dumps(pitch.model_dump(), indent=2), language="json")

    # --- Quality gate ---
    if vo:
        st.subheader("Quality Gate")
        passed = vo.passed
        color = "green" if passed else "red"
        label = "PASS ✅" if passed else "FAIL ❌"
        if vo.warning_flag:
            label += " ⚠️ max retries reached"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Specificity (40%)", f"{vo.specificity_score:.3f}")
        col2.metric("Alignment (40%)", f"{vo.alignment_score:.3f}")
        col3.metric("Comparability (20%)", f"{vo.comparability_score:.3f}")
        col4.metric("Composite", f"{vo.composite_score:.3f}", delta=label, delta_color="normal" if passed else "inverse")

        if vo.failure_reason:
            st.error(f"Failure reason: {vo.failure_reason}")
