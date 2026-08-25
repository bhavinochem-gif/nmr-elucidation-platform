import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nmrglue as ng
from streamlit_ketcher import st_ketcher

from dsp_engine import (
    process_fid, filter_solvent_peaks, deconvolve_spectrum,
    generate_predicted_spectrum, SOLVENT_TABLE
)
from chem_engine import analyze_and_number_molecule, draw_molecule_annotated
from gnn_predictor import ShiftPredictor
from assignment_engine import solve_assignment_2d
from quantum_engine import solve_quantum_spin_system
from report_engine import build_pdf_report

st.set_page_config(
    page_title="AI NMR Structure Elucidation Platform",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Automated NMR Structure Elucidation & Assignment Platform")
st.markdown("Automated 1D/2D NMR processing, in silico spectral synthesis, JEOL data ingestion, and publication-grade reporting.")

@st.cache_resource
def load_gnn_model():
    return ShiftPredictor()

predictor = load_gnn_model()

# -----------------------------------------------------------------------------
# SIDEBAR: SPECTROMETER PARAMETERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Spectrometer Parameters")
    solvent = st.selectbox(
        "Deuterated Solvent:",
        list(SOLVENT_TABLE.keys()),
        index=0,
        help="Select solvent for lock calibration and residual peak exclusion."
    )
    ref_data = SOLVENT_TABLE[solvent]
    st.caption(f"Ref Shifts: ¹H = {ref_data['1H']} ppm | ¹³C = {ref_data['13C']} ppm")
    st.divider()

    st.subheader("Field Frequencies")
    freq_1h = st.number_input(
        "¹H Spectrometer Frequency (MHz):",
        value=400.13,
        step=50.0,
        min_value=60.0,
        max_value=1200.0,
        format="%.2f"
    )

    auto_c_freq = round(freq_1h * 0.25144, 2)
    sync_c = st.checkbox("Auto-link ¹³C Frequency (Larmor ratio)", value=True)

    if sync_c:
        freq_13c = auto_c_freq
        st.info(f"¹³C Operating Frequency: **{freq_13c:.2f} MHz**")
    else:
        freq_13c = st.number_input(
            "¹³C Spectrometer Frequency (MHz):",
            value=auto_c_freq,
            step=10.0,
            min_value=15.0,
            max_value=300.0,
            format="%.2f"
        )

    st.divider()
    sample_id = st.text_input("Sample Identifier:", value="EXP-NMR-2026")

# -----------------------------------------------------------------------------
# 1. MOLECULAR STRUCTURE INPUT & TOPICITY NUMBERING
# -----------------------------------------------------------------------------
st.subheader("1. Structure Input & Stereochemical Numbering")
default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

input_mode = st.radio(
    "Structure Input Method:",
    ["Draw with Ketcher", "Direct SMILES Input"],
    horizontal=True
)

if input_mode == "Draw with Ketcher":
    with st.expander("Molecular Structure Sketcher", expanded=True):
        drawn = st_ketcher(value=default_smiles, height=400)
    active_smiles = drawn.strip() if drawn else default_smiles
else:
    active_smiles = st.text_input("Enter Canonical SMILES:", value=default_smiles).strip()

mol, mol_h, c_map, h_map, topo_2d, diastereotopic = analyze_and_number_molecule(active_smiles)

if mol is None:
    st.error("❌ Invalid chemical structure syntax. Please verify SMILES formatting.")
    st.stop()

col_v1, col_v2 = st.columns(2)
with col_v1:
    h_img = draw_molecule_annotated(mol_h, h_map)
    st.image(h_img, caption="¹H Proton Numbering & Topicity Map (Ha/Hb)", use_container_width=True)
with col_v2:
    c_img = draw_molecule_annotated(mol, c_map)
    st.image(c_img, caption="¹³C Carbon Numbering Map", use_container_width=True)

if diastereotopic:
    st.warning(f"⚠️ **Diastereotopic Protons Detected:** {', '.join([f'{a}/{b}' for a, b in diastereotopic])}")

# Compute In Silico Chemical Shift Predictions
h_pred, c_pred = predictor.predict(mol_h, c_map, h_map, solvent=solvent)

# -----------------------------------------------------------------------------
# 2. EXPERIMENTAL JEOL INGESTION
# -----------------------------------------------------------------------------
st.subheader("2. Experimental NMR File Ingestion")
up_col1, up_col2 = st.columns([2, 1])

exp_1h_peaks, exp_13c_peaks = [], []
exp_ppm_axis, exp_spec = None, None

with up_col1:
    up_fid = st.file_uploader(
        "Upload JEOL Raw NMR Data File (.jdf):",
        type=["jdf"],
        help="Upload 1D FID/processed .jdf file from JEOL spectrometer."
    )
    if up_fid:
        with st.spinner("Processing JEOL FID (FFT, Autophase & Whittaker Baseline)..."):
            with open("temp.jdf", "wb") as f:
                f.write(up_fid.getbuffer())
            dic, raw = ng.jeol.read("temp.jdf")
            exp_ppm_axis, exp_spec = process_fid(raw, dic, solvent=solvent, nucleus="1H", spec_freq_mhz=freq_1h)

        with st.spinner("Deconvolving overlapping sub-peaks and extracting J-couplings..."):
            raw_multiplets = deconvolve_spectrum(exp_ppm_axis, exp_spec, spec_freq=freq_1h, nucleus="1H")
            exp_1h_peaks = filter_solvent_peaks(raw_multiplets, solvent=solvent, nucleus="1H")

        st.success(f"✅ Extracted {len(exp_1h_peaks)} real multiplets after {solvent} artifact filtering.")

with up_col2:
    st.markdown("**Manual Fallback (Optional)**")
    h_manual = st.text_input("¹H Peaks (ppm):", "11.00, 8.12, 7.62, 7.35, 7.15, 2.35")
    if not exp_1h_peaks and h_manual:
        exp_1h_peaks = [
            {"ppm": float(x.strip()), "range": x.strip(), "multiplicity": "m", "protons": 1}
            for x in h_manual.split(",") if x.strip()
        ]

# -----------------------------------------------------------------------------
# 3. SIDE-BY-SIDE SPECTRA WINDOWS (PREDICTED VS EXPERIMENTAL JEOL)
# -----------------------------------------------------------------------------
st.subheader("3. Comparative NMR Spectra (In Silico Predicted vs. Experimental JEOL)")

view_nucleus = st.radio("Select Nucleus to Display:", ["¹H NMR Spectrum", "¹³C NMR Spectrum"], horizontal=True)

spec_col1, spec_col2 = st.columns(2)

# === WINDOW 1: PREDICTED SPECTRUM (LEFT) ===
with spec_col1:
    st.markdown(f"#### 📊 In Silico Predicted {view_nucleus}")
    
    target_pred_df = h_pred if view_nucleus == "¹H NMR Spectrum" else c_pred
    target_freq = freq_1h if view_nucleus == "¹H NMR Spectrum" else freq_13c
    target_nuc = "1H" if view_nucleus == "¹H NMR Spectrum" else "13C"
    
    pred_ppm_axis, pred_sim_spec, pred_annotations = generate_predicted_spectrum(
        pred_df=target_pred_df,
        spec_freq_mhz=target_freq,
        nucleus=target_nuc
    )
    
    fig_pred, ax_pred = plt.subplots(figsize=(6.5, 3.6), dpi=200)
    ax_pred.plot(pred_ppm_axis, pred_sim_spec, color="#0B3C5D", lw=1.2, label="Synthetic Lineshape")
    
    # Annotate predicted chemical shifts with atom labels
    max_y_pred = np.max(pred_sim_spec) if len(pred_sim_spec) > 0 and np.max(pred_sim_spec) > 0 else 1.0
    for ann in pred_annotations:
        ax_pred.axvline(ann["ppm"], color="#0B3C5D", linestyle=":", alpha=0.35)
        lbl_text = f"{ann['label']}\n{ann['ppm']:.2f}" if target_nuc == "1H" else f"{ann['label']}\n{ann['ppm']:.1f}"
        ax_pred.annotate(
            lbl_text,
            xy=(ann["ppm"], max_y_pred * 0.72),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            rotation=90,
            color="#0B3C5D",
            fontweight="bold"
        )
        
    ax_pred.set_xlim(max(pred_ppm_axis), min(pred_ppm_axis))  # Reversed NMR delta scale
    ax_pred.set_xlabel("Chemical Shift δ (ppm)", fontweight="bold", fontsize=8)
    ax_pred.set_ylabel("Peak Intensity (a.u.)", fontweight="bold", fontsize=8)
    ax_pred.set_title(f"Predicted {target_nuc} Spectrum ({solvent}, {target_freq:.1f} MHz)", fontsize=9, fontweight="bold")
    ax_pred.grid(True, linestyle="--", alpha=0.3)
    ax_pred.legend(loc="upper left", fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig_pred)

# === WINDOW 2: EXPERIMENTAL JEOL SPECTRUM (RIGHT) ===
with spec_col2:
    st.markdown(f"#### 📈 Uploaded JEOL Experimental {view_nucleus}")
    
    if exp_ppm_axis is not None and exp_spec is not None:
        fig_exp, ax_exp = plt.subplots(figsize=(6.5, 3.6), dpi=200)
        ax_exp.plot(exp_ppm_axis, exp_spec, color="#B82601", lw=1.1, label="JEOL Processed Spectrum")
        
        max_y_exp = np.max(exp_spec) if np.max(exp_spec) > 0 else 1.0
        for p in exp_1h_peaks:
            ax_exp.axvline(p["ppm"], color="#B82601", linestyle=":", alpha=0.4)
            ax_exp.annotate(
                f"{p['ppm']:.2f}\n({p.get('multiplicity', 'm')})",
                xy=(p["ppm"], max_y_exp * 0.72),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                rotation=90,
                color="#B82601",
                fontweight="bold"
            )
            
        ax_exp.set_xlim(max(exp_ppm_axis), min(exp_ppm_axis))  # Reversed NMR delta scale
        ax_exp.set_xlabel("Chemical Shift δ (ppm)", fontweight="bold", fontsize=8)
        ax_exp.set_ylabel("Peak Intensity (a.u.)", fontweight="bold", fontsize=8)
        ax_exp.set_title(f"Experimental Spectrum ({solvent}, {freq_1h:.1f} MHz)", fontsize=9, fontweight="bold")
        ax_exp.grid(True, linestyle="--", alpha=0.3)
        ax_exp.legend(loc="upper left", fontsize=7.5)
        plt.tight_layout()
        st.pyplot(fig_exp)
    else:
        # Placeholder window before file upload
        fig_dummy, ax_dummy = plt.subplots(figsize=(6.5, 3.6), dpi=200)
        dummy_ppm = np.linspace(14.0, -1.0, 500)
        ax_dummy.plot(dummy_ppm, np.zeros_like(dummy_ppm), color="gray", linestyle="--", alpha=0.5, label="No Data Loaded")
        ax_dummy.set_xlim(14.0, -1.0)
        ax_dummy.set_xlabel("Chemical Shift δ (ppm)", fontweight="bold", fontsize=8)
        ax_dummy.set_ylabel("Peak Intensity (a.u.)", fontweight="bold", fontsize=8)
        ax_dummy.set_title("Experimental Spectrum (Awaiting .jdf Upload)", fontsize=9, color="gray")
        ax_dummy.text(6.5, 0.5, "Upload a JEOL .jdf file\nin Section 2 to display", ha="center", va="center", color="gray", fontsize=9)
        ax_dummy.grid(True, linestyle="--", alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_dummy)

# -----------------------------------------------------------------------------
# 4. STRUCTURE ELUCIDATION & 2D BIPARTITE ASSIGNMENT MATRICES
# -----------------------------------------------------------------------------
st.subheader("4. Structure Elucidation & Assignment Matrices")

df_1h_res, df_13c_res = solve_assignment_2d(h_pred, c_pred, exp_1h_peaks, exp_13c_peaks, topo_2d)

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown(f"**¹H NMR Assignment Table ({solvent}, {freq_1h:.1f} MHz)**")
    st.dataframe(df_1h_res, use_container_width=True)
with col_t2:
    st.markdown(f"**¹³C NMR Assignment Table ({solvent}, {freq_13c:.1f} MHz)**")
    st.dataframe(df_13c_res, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. QUANTUM MECHANICAL SPIN SIMULATOR (EXPANDER)
# -----------------------------------------------------------------------------
with st.expander("🔬 Quantum Mechanical Spin System Simulator (AB / ABX Systems)", expanded=False):
    st.caption("Simulates second-order strong coupling effects (the roof effect) by diagonalizing the isotropic spin Hamiltonian matrix.")
    qc1, qc2 = st.columns([1, 2])
    with qc1:
        q_shifts = st.text_input("Coupled Spins δ (ppm):", "3.00, 3.03")
        q_j = st.number_input("Coupling Constant J (Hz):", value=14.0, step=1.0)
    with qc2:
        try:
            s_list = [float(x.strip()) for x in q_shifts.split(",") if x.strip()]
            j_m = np.zeros((len(s_list), len(s_list)))
            if len(s_list) == 2:
                j_m[0, 1] = j_m[1, 0] = q_j
            
            trans, q_ppm, q_spec = solve_quantum_spin_system(s_list, j_m, spec_freq=freq_1h)
            
            fig_q, ax_q = plt.subplots(figsize=(6, 2.0), dpi=160)
            ax_q.plot(q_ppm, q_spec, color="#0B3C5D", lw=1.2)
            ax_q.set_xlim(max(q_ppm), min(q_ppm))
            ax_q.set_xlabel("δ (ppm)", fontsize=8)
            ax_q.set_ylabel("Intensity", fontsize=8)
            ax_q.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_q)
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")

# -----------------------------------------------------------------------------
# 6. PUBLICATION-READY PDF REPORT EXPORT
# -----------------------------------------------------------------------------
st.subheader("5. Analytical PDF Report Export")

if st.button("📄 Generate Analytical PDF Report", type="primary", use_container_width=True):
    with st.spinner("Compiling publication-ready PDF document..."):
        try:
            pdf_bytes = build_pdf_report(
                sample_id=sample_id,
                smiles=active_smiles,
                solvent=solvent,
                freq_1h=f"{freq_1h:.2f} MHz",
                freq_13c=f"{freq_13c:.2f} MHz",
                df_1h=df_1h_res,
                df_13c=df_13c_res,
                img_buf=h_img
            )
            
            st.download_button(
                label="📥 Download Complete Elucidation Report (.PDF)",
                data=pdf_bytes,
                file_name=f"{sample_id}_NMR_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("Report generated successfully!")
        except Exception as e:
            st.error(f"Failed to compile PDF report: {str(e)}")
