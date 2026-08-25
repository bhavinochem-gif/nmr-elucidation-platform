import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks
import nmrglue as ng

SOLVENT_TABLE = {
    "CDCl3": {"1H": 7.26, "13C": 77.16, "water_1H": 1.56, "1H_tol": 0.15, "13C_tol": 1.5},
    "DMSO-d6": {"1H": 2.50, "13C": 39.52, "water_1H": 3.33, "1H_tol": 0.15, "13C_tol": 1.5},
    "Methanol-d4": {"1H": 3.31, "13C": 49.00, "water_1H": 4.87, "1H_tol": 0.15, "13C_tol": 1.5},
    "Acetone-d6": {"1H": 2.05, "13C": 29.84, "water_1H": 2.84, "1H_tol": 0.15, "13C_tol": 1.5},
    "D2O": {"1H": 4.79, "13C": None, "water_1H": 4.79, "1H_tol": 0.25, "13C_tol": 0.0},
    "CD3CN": {"1H": 1.94, "13C": 1.32, "water_1H": 2.13, "1H_tol": 0.15, "13C_tol": 1.5},
    "Benzene-d6": {"1H": 7.16, "13C": 128.06, "water_1H": 0.40, "1H_tol": 0.15, "13C_tol": 1.5},
    "Pyridine-d5": {"1H": 7.22, "13C": 123.87, "water_1H": 4.90, "1H_tol": 0.20, "13C_tol": 1.5}
}

# -----------------------------------------------------------------------------
# 1. FID PROCESSING & DSP
# -----------------------------------------------------------------------------
def apply_phase(data: np.ndarray, p0: float, p1: float) -> np.ndarray:
    n = len(data)
    phase_ramp = np.linspace(0, np.radians(p1), n) + np.radians(p0)
    return data * np.exp(1j * phase_ramp)

def autophase_entropy(complex_spec: np.ndarray) -> np.ndarray:
    def obj(phases):
        p0, p1 = phases
        phased = np.real(apply_phase(complex_spec, p0, p1))
        d_spec = np.abs(np.diff(phased))
        d_sum = np.sum(d_spec) + 1e-12
        h_entropy = -np.sum((d_spec / d_sum) * np.log((d_spec / d_sum) + 1e-12))
        neg_penalty = np.sum(np.square(np.minimum(phased, 0.0))) * 100.0
        return h_entropy + neg_penalty

    res = minimize(obj, x0=[0.0, 0.0], method="Nelder-Mead", options={"maxiter": 300})
    return np.real(apply_phase(complex_spec, res.x[0], res.x[1]))

def baseline_als(y: np.ndarray, lam: float = 1e6, p: float = 0.005, niter: int = 10) -> np.ndarray:
    L = len(y)
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), format="csc")
    w = np.ones(L)
    for _ in range(niter):
        W = sp.spdiags(w, 0, L, L, format="csc")
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z

def process_fid(fid_data: np.ndarray, dic: dict, solvent: str = "CDCl3", nucleus: str = "1H", spec_freq_mhz: float = 400.0) -> tuple:
    sw = dic.get("sw", spec_freq_mhz * 10.0)
    data_em = ng.proc_base.em(fid_data, lb=0.5 / sw)
    data_zf = ng.proc_base.zf(data_em, pad=len(data_em))
    complex_spec = ng.proc_base.fft(data_zf)
    phased = autophase_entropy(complex_spec)
    baseline = baseline_als(phased)
    corrected = phased - baseline

    try:
        udic = ng.jeol.guess_udic(dic, fid_data) if "jeol" in str(type(dic)).lower() else ng.bruker.guess_udic(dic, fid_data)
        uc = ng.fileiobase.unit_conversion(udic[0]['size'], udic[0]['complex'], udic[0]['sw'], udic[0]['obs'], udic[0]['car'])
        raw_ppm = uc.ppm_scale()
    except Exception:
        raw_ppm = np.linspace(14.0 if nucleus == "1H" else 230.0, -2.0, len(corrected))

    ref = SOLVENT_TABLE.get(solvent, SOLVENT_TABLE["CDCl3"])
    target_ppm = ref.get(nucleus)
    tol = ref.get(f"{nucleus}_tol", 0.2)

    if target_ppm is not None:
        mask = (raw_ppm >= target_ppm - tol) & (raw_ppm <= target_ppm + tol)
        if np.any(mask):
            local_max = np.where(mask)[0][np.argmax(corrected[mask])]
            raw_ppm += (target_ppm - raw_ppm[local_max])

    return raw_ppm, corrected

def filter_solvent_peaks(peaks: list, solvent: str = "CDCl3", nucleus: str = "1H") -> list:
    ref = SOLVENT_TABLE.get(solvent, {})
    s_ppm = ref.get(nucleus)
    w_ppm = ref.get("water_1H") if nucleus == "1H" else None

    clean = []
    for p in peaks:
        val = p["ppm"]
        if s_ppm and abs(val - s_ppm) < (0.04 if nucleus == "1H" else 0.5):
            continue
        if w_ppm and abs(val - w_ppm) < 0.06:
            continue
        if solvent == "CDCl3" and nucleus == "1H" and abs(val - 0.07) < 0.02:
            continue
        clean.append(p)
    return clean

# -----------------------------------------------------------------------------
# 2. EXPERIMENTAL MULTIPLET DECONVOLUTION
# -----------------------------------------------------------------------------
def multi_lorentzian(x, *params):
    y = np.zeros_like(x)
    n_peaks = (len(params) - 1) // 3
    for i in range(n_peaks):
        a, c, g = params[3*i], params[3*i+1], params[3*i+2]
        y += (a / np.pi) * (g / ((x - c)**2 + g**2))
    return y + params[-1]

def extract_j_couplings(sub_peaks: list) -> tuple:
    n = len(sub_peaks)
    if n == 0:
        return "m", [], 0.0
    if n == 1:
        return "s", [], sub_peaks[0]["ppm"]

    freqs = np.array([p["hz"] for p in sub_peaks])
    center_ppm = float(np.mean([p["ppm"] for p in sub_peaks]))

    if n == 2:
        return "d", [round(float(abs(freqs[1] - freqs[0])), 2)], center_ppm
    if n == 3:
        j_avg = round(float((abs(freqs[1] - freqs[0]) + abs(freqs[2] - freqs[1])) / 2.0), 2)
        return "t", [j_avg], center_ppm
    if n == 4:
        diffs = np.diff(freqs)
        if abs(diffs[0] - diffs[1]) <= 0.6 and abs(diffs[1] - diffs[2]) <= 0.6:
            return "q", [round(float(np.mean(diffs)), 2)], center_ppm
        j1 = (abs(freqs[3] - freqs[1]) + abs(freqs[2] - freqs[0])) / 2.0
        j2 = (abs(freqs[3] - freqs[2]) + abs(freqs[1] - freqs[0])) / 2.0
        return "dd", [round(float(max(j1, j2)), 2), round(float(min(j1, j2)), 2)], center_ppm

    return "m", [round(float(abs(freqs[-1] - freqs[0])), 2)], center_ppm

def deconvolve_spectrum(ppm_axis: np.ndarray, spectrum: np.ndarray, spec_freq: float = 400.0, nucleus: str = "1H") -> list:
    threshold = np.max(spectrum) * (0.03 if nucleus == "1H" else 0.05)
    cluster_gap = 0.08 if nucleus == "1H" else 1.0
    
    p_indices, _ = find_peaks(spectrum, height=threshold, distance=4)
    if len(p_indices) == 0:
        return []

    clusters, curr = [], [p_indices[0]]
    for i in range(1, len(p_indices)):
        if abs(ppm_axis[p_indices[i]] - ppm_axis[p_indices[i-1]]) <= cluster_gap:
            curr.append(p_indices[i])
        else:
            clusters.append(curr)
            curr = [p_indices[i]]
    clusters.append(curr)

    results = []
    hz_axis = ppm_axis * spec_freq

    for cl in clusters:
        mn, mx = max(0, min(cl) - 20), min(len(ppm_axis), max(cl) + 20)
        if mn > mx:
            mn, mx = mx, mn
        p_sub = ppm_axis[mn:mx]
        h_sub = hz_axis[mn:mx]
        s_sub = spectrum[mn:mx]

        sub_pks, _ = find_peaks(s_sub, prominence=np.max(s_sub) * 0.05)
        init = []
        for idx in sub_pks:
            init.extend([s_sub[idx] * np.pi * 0.5, h_sub[idx], 0.5])
        init.append(np.min(s_sub))

        try:
            popt, _ = curve_fit(multi_lorentzian, h_sub, s_sub, p0=init, maxfev=1500)
            fitted_peaks = [{"ppm": popt[3*i+1] / spec_freq, "hz": popt[3*i+1]} for i in range((len(popt) - 1) // 3)]
            fitted_peaks.sort(key=lambda x: x["hz"])
        except Exception:
            fitted_peaks = [{"ppm": p_sub[i], "hz": h_sub[i]} for i in sub_pks]

        mult, j_list, center = extract_j_couplings(fitted_peaks)
        j_str = f", J = {', '.join([str(j) for j in j_list])} Hz" if j_list else ""
        results.append({
            "ppm": round(center, 3),
            "range": f"{min(p_sub):.2f}-{max(p_sub):.2f}",
            "multiplicity": mult,
            "protons": 1,
            "formatted": f"{center:.2f} ({mult}{j_str})"
        })

    return results

# -----------------------------------------------------------------------------
# 3. PREDICTED 1D SYNTHETIC SPECTRUM GENERATOR
# -----------------------------------------------------------------------------
def generate_predicted_spectrum(
    pred_df,
    spec_freq_mhz: float = 400.0,
    nucleus: str = "1H",
    num_points: int = 4000
) -> tuple:
    """
    Synthesizes a continuous 1D NMR spectrum from predicted shifts, integrals, and multiplicities.
    Returns (ppm_axis, simulated_spectrum, annotations_list).
    """
    if pred_df.empty:
        ppm_axis = np.linspace(14.0, -1.0, num_points) if nucleus == "1H" else np.linspace(230.0, -10.0, num_points)
        return ppm_axis, np.zeros_like(ppm_axis), []

    shifts = pred_df["Shift"].values
    if nucleus == "1H":
        ppm_min = max(-1.0, np.min(shifts) - 1.5)
        ppm_max = min(15.0, np.max(shifts) + 1.5)
    else:
        ppm_min = max(-10.0, np.min(shifts) - 10.0)
        ppm_max = min(240.0, np.max(shifts) + 15.0)

    ppm_axis = np.linspace(ppm_max, ppm_min, num_points)  # Reversed NMR axis
    sim_spectrum = np.zeros_like(ppm_axis)
    annotations = []

    fwhm_hz = 0.8 if nucleus == "1H" else 1.5
    fwhm_ppm = fwhm_hz / spec_freq_mhz
    gamma = fwhm_ppm / 2.0

    for _, row in pred_df.iterrows():
        c_shift = float(row["Shift"])
        protons = float(row.get("Protons", 1.0))
        mult = str(row.get("Mult", "s")).lower()
        lbl = str(row["Label"])

        # Construct sub-peak splitting offsets in Hz
        if mult == "d":
            offsets_hz = [-3.8, 3.8]
            weights = [0.5, 0.5]
        elif mult == "t":
            offsets_hz = [-7.0, 0.0, 7.0]
            weights = [0.25, 0.5, 0.25]
        elif mult == "q":
            offsets_hz = [-10.5, -3.5, 3.5, 10.5]
            weights = [0.125, 0.375, 0.375, 0.125]
        elif mult == "dd":
            offsets_hz = [-4.8, -3.2, 3.2, 4.8]
            weights = [0.25, 0.25, 0.25, 0.25]
        elif mult == "br s":
            offsets_hz = [0.0]
            weights = [1.0]
            gamma_eff = (2.5 / spec_freq_mhz) / 2.0
        else:
            offsets_hz = [0.0]
            weights = [1.0]

        gamma_use = gamma if mult != "br s" else (2.5 / spec_freq_mhz) / 2.0

        for off_hz, w in zip(offsets_hz, weights):
            sub_ppm = c_shift + (off_hz / spec_freq_mhz)
            area = protons * w
            sim_spectrum += area * (1.0 / np.pi) * (gamma_use / ((ppm_axis - sub_ppm)**2 + gamma_use**2))

        annotations.append({
            "label": lbl,
            "ppm": c_shift,
            "mult": mult,
            "protons": int(protons) if nucleus == "1H" else None
        })

    return ppm_axis, sim_spectrum, annotations
