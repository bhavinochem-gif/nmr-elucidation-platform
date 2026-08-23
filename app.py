import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nmrglue as ng
from streamlit_ketcher import st_ketcher

from dsp_engine import process_fid, filter_solvent_peaks, deconvolve_spectrum
from chem_engine import analyze_and_number_molecule, draw_molecule_annotated
from gnn_predictor import ShiftPredictor
from assignment_engine import solve_assignment_2d
from quantum_engine import solve_quantum_spin_system
from report_engine import build_pdf_report

st.set_page_config(page_title="Structure Elucidation Platform", layout="wide")
st.title("AI NMR Structure Elucidation & Assignment Platform")

@st.cache_resource
def load_gnn_model():
    return ShiftPredictor()

predictor = load_gnn_model()

# Sidebar Settings
with st.sidebar:
    st.header("Spectrometer Settings")
    solvent = st.selectbox("Deuterated Solvent:", ["CDCl3", "DMSO-d6", "Methanol-d4", "Acetone-d6", "D2O"])
    freq = st.number_input("Field Frequency (MHz):", value=400.0, step=100.0)
    sample_id = st.text_input("Sample Identifier:", value="EXP-NMR-2026")

# 1. Structure Sketcher & Processing
st.subheader("1. Structure Input & Stereochemical Numbering")
default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

with st.expander("Draw / Edit Molecular Structure", expanded=True):
    drawn = st_ketcher(initial_smiles=default_smiles, height=380)

active_smiles = drawn.strip() if drawn else default_smiles
mol, mol_h, c_map, h_map, topo_2d, diastereotopic = analyze_and_number_molecule(active_smiles)

if mol is None:
    st.error("Invalid chemical structure syntax.")
    st.stop()

col_v1, col_v2 = st.columns(2)
with col_v1:
    h_img = draw_molecule_annotated(mol_h, h_map)
    st.image(h_img, caption="1H Numbering & Topicity Map (Ha/Hb)", use_container_width=True)
with col_v2:
    c_img = draw_molecule_annotated(mol, c_map)
    st.image(c_img, caption="13C Numbering Map", use_container_width=True)

if diastereotopic:
    st.info(f"Diastereotopic Centers Resolved: {', '.join([f'{a}/{b}' for a, b in diastereotopic])}")

# 2. Spectroscopic Data Ingestion & Deconvolution
st.subheader("2. Spectrum Processing & Automated Deconvolution")
in_mode = st.radio("Ingestion Mode:", ["Raw JEOL (.jdf) / Bruker FID", "Manual Peak Entry"], horizontal=True)

exp_1h_peaks, exp_13c_peaks = [], []

if in_mode == "Raw JEOL (.jdf) / Bruker FID":
    up_fid = st.file_uploader("Upload raw spectrometer file (.jdf):", type=["jdf"])
    if up_fid:
        with st.spinner("Executing FFT, Entropy Autophasing & Whittaker Baseline..."):
            with open("temp.jdf", "wb") as f:
                f.write(up_fid.getbuffer())
            dic, raw = ng.jeol.read("temp.jdf")
            ppm_axis, spec = process_fid(raw, dic, solvent=solvent, nucleus="1H")

        with st.spinner("Deconvolving overlapping multiplets and extracting J-couplings..."):
            raw_multiplets = deconvolve_spectrum(ppm_axis, spec, spec_freq=freq)
            exp_1h_peaks = filter_solvent_peaks(raw_multiplets, solvent=solvent, nucleus="1H")

        st.success(f"Extracted {len(exp_1h_peaks)} real multiplets after solvent/artifact rejection.")

        fig, ax = plt.subplots(figsize=(10, 2.8), dpi=180)
        ax.plot(ppm_axis, spec, color="#0B3C5D", lw=1.1)
        for p in exp_1h_peaks:
            ax.axvline(p["ppm"], color="#B82601", linestyle=":", alpha=0.5)
            ax.annotate(f"{p['ppm']:.2f}\n({p['multiplicity']})", xy=(p["ppm"], np.max(spec)*0.7), rotation=90, fontsize=6.5, ha="center", color="#B82601")
        ax.set_xlim(max(ppm_axis), min(ppm_axis))
        ax.set_xlabel("Chemical Shift δ (ppm)")
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
else:
    h_raw = st.text_input("1H Peaks (ppm):", "11.00, 8.12, 7.62, 7.35, 7.15, 2.35")
    c_raw = st.text_input("13C Peaks (ppm):", "170.1, 169.8, 151.2, 134.8, 132.4, 126.1, 123.9, 122.2, 20.9")
    if h_raw:
        exp_1h_peaks = [{"ppm": float(x.strip()), "range": x.strip(), "multiplicity": "m", "protons": 1} for x in h_raw.split(",") if x.strip()]
    if c_raw:
        exp_13c_peaks = [{"ppm": float(x.strip())} for x in c_raw.split(",") if x.strip()]

# 3. Deep Learning Prediction & Assignment Tables
st.subheader("3. Structure Elucidation & 2D Hungarian Assignment")
h_pred, c_pred = predictor.predict(mol_h, c_map, h_map, solvent=solvent)
df_1h_res, df_13c_res = solve_assignment_2d(h_pred, c_pred, exp_1h_peaks, exp_13c_peaks, topo_2d)

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("**¹H NMR Elucidation Matrix**")
    st.dataframe(df_1h_res, use_container_width=True)
with col_t2:
    st.markdown("**¹³C NMR Elucidation Matrix**")
    st.dataframe(df_13c_res, use_container_width=True)

# 4. Quantum Simulation Tab
with st.expander("🔬 Quantum Mechanical Spin System Simulator (AB/ABX Systems)"):
    qc1, qc2 = st.columns([1, 2])
    with qc1:
        q_shifts = st.text_input("Coupled Spins δ (ppm):", "3.00, 3.03")
        q_j = st.number_input("J-coupling (Hz):", value=14.0)
    with qc2:
        s_list = [float(x.strip()) for x in q_shifts.split(",")]
        j_m = np.zeros((len(s_list), len(s_list)))
        if len(s_list) == 2:
            j_m[0, 1] = j_m[1, 0] = q_j
        trans, q_ppm, q_spec = solve_quantum_spin_system(s_list, j_m, spec_freq=freq)
        
        fig_q, ax_q = plt.subplots(figsize=(6, 2.0), dpi=150)
        ax_q.plot(q_ppm, q_spec, color="#0B3C5D", lw=1.2)
        ax_q.set_xlim(max(q_ppm), min(q_ppm))
        ax_q.set_xlabel("δ (ppm)")
        ax_q.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig_q)

# 5. Export Report
st.subheader("4. Analytical PDF Export")
if st.button("Generate Final PDF Report", use_container_width=True):
    pdf_bytes = build_pdf_report(
        sample_id=sample_id, smiles=active_smiles, solvent=solvent,
        freq=f"{freq:.1f} MHz", df_1h=df_1h_res, df_13c=df_13c_res, img_buf=h_img
    )
    st.download_button(
        label="📥 Download Complete Report (.PDF)",
        data=pdf_bytes,
        file_name=f"{sample_id}_NMR_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
