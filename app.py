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

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CACHED RESOURCES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI NMR Structure Elucidation Platform",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Automated NMR Structure Elucidation & Assignment Platform")
st.markdown("Automated 1D/2D NMR processing, in silico spectral synthesis, dual JEOL data ingestion, and publication-grade reporting.")

@st.cache_resource
def load_gnn_model():
    return ShiftPredictor()

predictor = load_gnn_model()

# -----------------------------------------------------------------------------
# SIDEBAR: SPECTROMETER PARAMETERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Spectrometer Parameters")
    
    # 1. Deuterated Solvent Selection
    solvent = st.selectbox(
        "Deuterated Solvent:",
        list(SOLVENT_TABLE.keys()),
        index=0,
        help="Select NMR solvent for automatic lock calibration and solvent peak exclusion."
    )
    ref_data = SOLVENT_TABLE[solvent]
    st.caption(f"Ref Shifts: ¹H = {ref_data['1H']} ppm | ¹³C = {ref_data['13C']} ppm")
    st.divider()

    # 2. Field Frequencies Selection
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

# Analyze chemical topology, stereocenters, and 2D connectivity
mol, mol_h, c_map, h_map, topo_2d, diastereotopic = analyze_and_number_molecule(active_smiles)

if mol is None:
    st.error("❌ Invalid chemical structure syntax. Please verify SMILES formatting or atom valencies.")
    st.stop()

col_v1, col_v2 = st.columns(2)
with col_v1:
    h_img = draw_molecule_annotated(mol_h, h_map)
    st.image(h_img, caption="¹H Proton Numbering & Topicity Map (Ha/Hb)", use_container_width=True)
with col_v2:
    c_img = draw_molecule_annotated(mol, c_map)
    st.image(c_img, caption="¹³C Carbon Numbering Map", use_container_width=True)

if diastereotopic:
    st.warning(f"⚠️ **Diastereotopic Protons Detected:** {', '.join([f'{a}/{b}' for a, b in diastereotopic])} — resolved as non-equivalent resonances.")

# In Silico Chemical Shift Predictions (GNN / Multi-hop Topological Additivity)
h_pred, c_pred = predictor.predict(mol_h, c_map, h_map, solvent=solvent)

# -----------------------------------------------------------------------------
# 2. DUAL EXPERIMENTAL NMR INGESTION (1H & 13C .jdf)
# -----------------------------------------------------------------------------
st.subheader("2. Experimental NMR File Ingestion (¹H and ¹³C)")

col_ingest_1h, col_ingest_13c = st.columns(2)

exp_1h_peaks, exp_13c_peaks = [], []
exp_1h_ppm, exp_1h_spec = None, None
exp_13c_ppm, exp_13c_spec = None, None

# --- PROTON NMR INGESTION (LEFT) ---
with col_ingest_1h:
    st.markdown("#### 🔹 ¹H NMR Ingestion")
    up_fid_1h = st.file_uploader("Upload ¹H JEOL File (.jdf):", type=["jdf"], key="fid_1h")
    
    if up_fid_1h:
        with st.spinner("Processing ¹H FID (FFT, Autophase, Baseline & Lock)..."):
            with open("temp_1h.jdf", "wb") as f:
                f.write(up_fid_1h.getbuffer())
            dic_1h, raw_1h = ng.jeol.read("temp_1h.jdf")
            exp_1h_ppm, exp_1h_spec = process_fid(
                raw_1h, dic_1h, solvent=solvent, nucleus="1H", spec_freq_mhz=freq_1h
            )
            raw_mults_1h = deconvolve_spectrum(exp_1h_ppm, exp_1h_spec, spec_freq=freq_1h, nucleus="1H")
            exp_1h_peaks = filter_solvent_peaks(raw_mults_1h, solvent=solvent, nucleus="1H")
        st.success(f"✅ Extracted {len(exp_1h_peaks)} ¹H resonances.")
    else:
        h_manual = st.text_input("Manual ¹H Peaks (ppm, comma-separated):", "11.00, 8.12, 7.62, 7.35, 7.15, 2.35")
        if h_manual.strip():
            exp_1h_peaks = [
                {"ppm": float(x.strip()), "range": f"{float(x.strip()):.2f}", "multiplicity": "m", "protons": 1}
                for x in h_manual.split(",") if x.strip()
            ]

# --- CARBON NMR INGESTION (RIGHT) ---
with col_ingest_13c:
    st.markdown("#### 🔸 ¹³C NMR Ingestion")
    up_fid_13c = st.file_uploader("Upload ¹³C JEOL File (.jdf):", type=["jdf"], key="fid_13c")
    
    if up_fid_13c:
        with st.spinner("Processing ¹³C FID (FFT, Autophase, Baseline & Lock)..."):
            with open("temp_13c.jdf", "wb") as f:
                f.write(up_fid_13c.getbuffer())
            dic_13c, raw_13c = ng.jeol.read("temp_13c.jdf")
            exp_13c_ppm, exp_13c_spec = process_fid(
                raw_13c, dic_13c, solvent=solvent, nucleus="13C", spec_freq_mhz=freq_13c
            )
            raw_peaks_13c = deconvolve_spectrum(exp_13c_ppm, exp_13c_spec, spec_freq=freq_13c, nucleus="13C")
            exp_13c_peaks = filter_solvent_peaks(raw_peaks_13c, solvent=solvent, nucleus="13C")
        st.success(f"✅ Extracted {len(exp_13c_peaks)} ¹³C resonances.")
    else:
        c_manual = st.text_input("Manual ¹³C Peaks (ppm, comma-separated):", "170.1, 169.8, 151.2, 134.8, 132.4, 126.1, 123.9, 122.2, 20.9")
        if c_manual.strip():
            exp_13c_peaks = [
                {"ppm": float(x.strip())}
                for x in c_manual.split(",") if x.strip()
            ]

# -----------------------------------------------------------------------------
# 3. COMPARATIVE SPECTRA WINDOWS (ABOVE AND BELOW)
# -----------------------------------------------------------------------------
st.subheader("3. Comparative NMR Spectra (In Silico Predicted vs. Experimental JEOL)")

view_nucleus = st.radio("Select Nucleus to Display:", ["¹H NMR Spectrum", "¹³C NMR Spectrum"], horizontal=True)

if view_nucleus == "¹H NMR Spectrum":
    target_pred_df = h_pred
    target_freq = freq_1h
    target_nuc = "1H"
    exp_ppm_show = exp_1h_ppm
    exp_spec_show = exp_1h_spec
    exp_peaks_show = exp_1h_peaks
else:
    target_pred_df = c_pred
    target_freq = freq_13c
    target_nuc = "13C"
    exp_ppm_show = exp_13c_ppm
    exp_spec_show = exp_13c_spec
    exp_peaks_show = exp_13c_peaks

# --- TOP WINDOW: IN SILICO PREDICTED ---
st.markdown(f"#### 📊 In Silico Predicted {view_nucleus} (Top Window)")
pred_ppm_axis, pred_sim_spec, pred_annotations = generate_predicted_spectrum(
    pred_df=target_pred_df,
    spec_freq_mhz=target_freq,
    nucleus=target_nuc
)

fig_pred, ax_pred = plt.subplots(figsize=(10, 3.0), dpi=200)
ax_pred.plot(pred_ppm_axis, pred_sim_spec, color="#0B3C5D", lw=1.2, label="In Silico Lineshape")
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

ax_pred.set_xlim(max(pred_ppm_axis), min(pred_ppm_axis))
ax_pred.set_xlabel("Chemical Shift δ (ppm)", fontweight="bold", fontsize=8)
ax_pred.set_ylabel("Intensity (a.u.)", fontweight="bold", fontsize=8)
ax_pred.set_title(f"Predicted {target_nuc} Spectrum ({solvent}, {target_freq:.1f} MHz)", fontsize=9, fontweight="bold")
ax_pred.grid(True, linestyle="--", alpha=0.3)
ax_pred.legend(loc="upper left", fontsize=7.5)
plt.tight_layout()
st.pyplot(fig_pred)

st.divider()

# --- BOTTOM WINDOW: EXPERIMENTAL JEOL ---
st.markdown(f"#### 📈 Uploaded JEOL Experimental {view_nucleus} (Bottom Window)")

if exp_ppm_show is not None and exp_spec_show is not None:
    fig_exp, ax_exp = plt.subplots(figsize=(10, 3.0), dpi=200)
    ax_exp.plot(exp_ppm_show, exp_spec_show, color="#B82601", lw=1.1, label="JEOL Processed Spectrum")
    max_y_exp = np.max(exp_spec_show) if np.max(exp_spec_show) > 0 else 1.0
    
    for p in exp_peaks_show:
        ax_exp.axvline(p["ppm"], color="#B82601", linestyle=":", alpha=0.4)
        ax_exp.annotate(
            f"{p['ppm']:.1f}" if target_nuc == "13C" else f"{p['ppm']:.2f}\n({p.get('multiplicity', 'm')})",
            xy=(p["ppm"], max_y_exp * 0.72),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            rotation=90,
            color="#B82601",
            fontweight="bold"
        )
        
    ax_exp.set_xlim(max(exp_ppm_show), min(exp_ppm_show))
    ax_exp.set_xlabel("Chemical Shift δ (ppm)", fontweight="bold", fontsize=8)
    ax_exp.set_ylabel("Intensity (a.u.)", fontweight="bold", fontsize=8)
    ax_exp.set_title(f"Experimental {target_nuc} Spectrum ({solvent}, {target_freq:.1f} MHz)", fontsize=9, fontweight="bold")
    ax_exp.grid(True, linestyle="--", alpha=0.3)
    ax_exp.legend(loc="upper left", fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig_exp)
else:
    fig_dummy, ax_dummy = plt.subplots(figsize=(10, 2.2), dpi=200)
    d_range = (14.0, -1.0) if target_nuc == "1H" else (220.0, -10.0)
    dummy_ppm = np.linspace(d_range[0], d_range[1], 500)
    ax_dummy.plot(dummy_ppm, np.zeros_like(dummy_ppm), color="gray", linestyle="--", alpha=0.5)
    ax_dummy.set_xlim(d_range[0], d_range[1])
    ax_dummy.set_xlabel("Chemical Shift δ (ppm)", fontweight="bold", fontsize=8)
    ax_dummy.set_ylabel("Intensity (a.u.)", fontweight="bold", fontsize=8)
    ax_dummy.set_title(f"Experimental {target_nuc} Spectrum (Awaiting File Upload)", fontsize=9, color="gray")
    ax_dummy.text((d_range[0] + d_range[1]) / 2.0, 0.5, f"Upload a {target_nuc} .jdf file in Section 2 to display", ha="center", va="center", color="gray", fontsize=9)
    ax_dummy.grid(True, linestyle="--", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_dummy)

# -----------------------------------------------------------------------------
# 4. STRUCTURE ELUCIDATION & ASSIGNMENT MATRICES
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
