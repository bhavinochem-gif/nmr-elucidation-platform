import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nmrglue as ng
from streamlit_ketcher import st_ketcher

# Import local analytical and processing engines
from dsp_engine import process_fid, filter_solvent_peaks, deconvolve_spectrum
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
st.markdown("Automated 1D/2D NMR processing, GNN shift prediction, quantum spin simulation, and publication-grade reporting.")

@st.cache_resource
def load_gnn_model():
    return ShiftPredictor()

predictor = load_gnn_model()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Spectrometer Parameters")
    solvent = st.selectbox(
        "Deuterated Solvent:",
        ["CDCl3", "DMSO-d6", "Methanol-d4", "Acetone-d6", "D2O", "CD3CN"],
        index=0
    )
    freq = st.number_input("Field Frequency (MHz):", value=400.0, step=100.0, min_value=60.0, max_value=1200.0)
    sample_id = st.text_input("Sample Identifier:", value="EXP-NMR-2026")
    st.divider()
    st.info("💡 **Workflow:**\n1. Draw / Input SMILES\n2. Ingest raw FID / Peaks\n3. Match shifts via Hungarian Algorithm\n4. Export Analytical PDF Report")

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
        # Uses 'value=' parameter to maintain compatibility with streamlit-ketcher
        drawn = st_ketcher(value=default_smiles, height=400)
    active_smiles = drawn.strip() if drawn else default_smiles
else:
    active_smiles = st.text_input("Enter Canonical SMILES:", value=default_smiles).strip()

# Analyze chemical structure and compute topicity
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
    st.warning(f"⚠️ **Diastereotopic Protons Detected:** {', '.join([f'{a}/{b}' for a, b in diastereotopic])} — will be resolved as distinct resonances.")

# -----------------------------------------------------------------------------
# 2. SPECTROSCOPIC DATA INGESTION & AUTOMATED DECONVOLUTION
# -----------------------------------------------------------------------------
st.subheader("2. Spectrum Processing & Automated Deconvolution")
in_mode = st.radio(
    "Data Source Mode:",
    ["Raw Spectrometer Data (JEOL .jdf)", "Manual Peak Entry / Table"],
    horizontal=True
)

exp_1h_peaks = []
exp_13c_peaks = []

if in_mode == "Raw Spectrometer Data (JEOL .jdf)":
    up_fid = st.file_uploader("Upload JEOL Raw NMR Data (.jdf):", type=["jdf"])
    if up_fid:
        with st.spinner("Executing FFT, Entropy-Based Autophasing & Whittaker Baseline Correction..."):
            with open("temp.jdf", "wb") as f:
                f.write(up_fid.getbuffer())
            dic, raw = ng.jeol.read("temp.jdf")
            ppm_axis, spec = process_fid(raw, dic, solvent=solvent, nucleus="1H")

        with st.spinner("Deconvolving overlapping sub-peaks and extracting J-couplings..."):
            raw_multiplets = deconvolve_spectrum(ppm_axis, spec, spec_freq=freq)
            exp_1h_peaks = filter_solvent_peaks(raw_multiplets, solvent=solvent, nucleus="1H")

        st.success(f"✅ Extracted {len(exp_1h_peaks)} real multiplets after solvent and artifact filtering.")

        # Interactive Spectrum Plot with annotations
        fig, ax = plt.subplots(figsize=(10, 3.0), dpi=200)
        ax.plot(ppm_axis, spec, color="#0B3C5D", lw=1.1, label="Processed Spectrum")
        for p in exp_1h_peaks:
            ax.axvline(p["ppm"], color="#B82601", linestyle=":", alpha=0.5)
            ax.annotate(
                f"{p['ppm']:.2f}\n({p['multiplicity']})",
                xy=(p["ppm"], np.max(spec) * 0.75),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                rotation=90,
                color="#B82601",
                fontweight="bold"
            )
        ax.set_xlim(max(ppm_axis), min(ppm_axis))  # Invert NMR axis
        ax.set_xlabel("Chemical Shift δ (ppm)", fontweight="bold", fontsize=8)
        ax.set_ylabel("Intensity (a.u.)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("📊 View Extracted Multiplets Table"):
            st.dataframe(pd.DataFrame(exp_1h_peaks), use_container_width=True)

else:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        h_raw = st.text_area(
            "¹H Peaks (ppm, comma-separated):",
            "11.00, 8.12, 7.62, 7.35, 7.15, 2.35"
        )
        if h_raw:
            exp_1h_peaks = [
                {"ppm": float(x.strip()), "range": x.strip(), "multiplicity": "m", "protons": 1}
                for x in h_raw.split(",") if x.strip()
            ]
    with col_p2:
        c_raw = st.text_area(
            "¹³C Peaks (ppm, comma-separated):",
            "170.1, 169.8, 151.2, 134.8, 132.4, 126.1, 123.9, 122.2, 20.9"
        )
        if c_raw:
            exp_13c_peaks = [
                {"ppm": float(x.strip())}
                for x in c_raw.split(",") if x.strip()
            ]

# -----------------------------------------------------------------------------
# 3. GNN SHIFT PREDICTION & 2D BIPARTITE ASSIGNMENT
# -----------------------------------------------------------------------------
st.subheader("3. Structure Elucidation & Assignment Matrices")

# Predict shifts via Graph Neural Network
h_pred, c_pred = predictor.predict(mol_h, c_map, h_map, solvent=solvent)

# Solve optimal bipartite matching
df_1h_res, df_13c_res = solve_assignment_2d(h_pred, c_pred, exp_1h_peaks, exp_13c_peaks, topo_2d)

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown(f"**¹H NMR Elucidation Matrix ({solvent}, {freq:.0f} MHz)**")
    st.dataframe(df_1h_res, use_container_width=True)
with col_t2:
    st.markdown(f"**¹³C NMR Elucidation Matrix ({solvent})**")
    st.dataframe(df_13c_res, use_container_width=True)

# -----------------------------------------------------------------------------
# 4. QUANTUM MECHANICAL SPIN SYSTEM SIMULATOR (EXPANDER)
# -----------------------------------------------------------------------------
with st.expander("🔬 Quantum Mechanical Spin System Simulator (AB / ABX Systems)", expanded=False):
    st.caption("Simulates second-order strong coupling effects (the roof effect) by diagonalizing the isotropic spin Hamiltonian matrix.")
    qc1, qc2 = st.columns([1, 2])
    with qc1:
        q_shifts = st.text_input("Coupled Chemical Shifts δ (ppm):", "3.00, 3.03")
        q_j = st.number_input("Coupling Constant J (Hz):", value=14.0, step=1.0)
    with qc2:
        try:
            s_list = [float(x.strip()) for x in q_shifts.split(",") if x.strip()]
            j_m = np.zeros((len(s_list), len(s_list)))
            if len(s_list) == 2:
                j_m[0, 1] = j_m[1, 0] = q_j
            
            trans, q_ppm, q_spec = solve_quantum_spin_system(s_list, j_m, spec_freq=freq)
            
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
# 5. PUBLICATION-READY PDF REPORT EXPORT
# -----------------------------------------------------------------------------
st.subheader("4. Analytical PDF Report Export")

if st.button("📄 Generate Analytical PDF Report", type="primary", use_container_width=True):
    with st.spinner("Compiling publication-ready PDF document..."):
        try:
            pdf_bytes = build_pdf_report(
                sample_id=sample_id,
                smiles=active_smiles,
                solvent=solvent,
                freq=f"{freq:.1f} MHz",
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
