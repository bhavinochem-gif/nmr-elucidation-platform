import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks
import nmrglue as ng

# -----------------------------------------------------------------------------
# SOLVENT REFERENCE LIBRARY (JEOL Delta Standard Referencing)
# -----------------------------------------------------------------------------
SOLVENT_TABLE = {
    "DMSO-d6": {
        "1H": 2.50, "13C": 39.52, "water_1H": 3.33,
        "1H_tol": 0.35, "13C_tol": 2.0
    },
    "CDCl3": {
        "1H": 7.26, "13C": 77.16, "water_1H": 1.56,
        "1H_tol": 0.35, "13C_tol": 2.0
    },
    "Methanol-d4": {
        "1H": 3.31, "13C": 49.00, "water_1H": 4.87,
        "1H_tol": 0.35, "13C_tol": 2.0
    },
    "Acetone-d6": {
        "1H": 2.05, "13C": 29.84, "water_1H": 2.84,
        "1H_tol": 0.35, "13C_tol": 2.0
    },
    "D2O": {
        "1H": 4.79, "13C": None, "water_1H": 4.79,
        "1H_tol": 0.40, "13C_tol": 0.0
    },
    "CD3CN": {
        "1H": 1.94, "13C": 1.32, "water_1H": 2.13,
        "1H_tol": 0.35, "13C_tol": 2.0
    },
    "Benzene-d6": {
        "1H": 7.16, "13C": 128.06, "water_1H": 0.40,
        "1H_tol": 0.35, "13C_tol": 2.0
    },
    "Pyridine-d5": {
        "1H": 7.22, "13C": 123.87, "water_1H": 4.90,
        "1H_tol": 0.35, "13C_tol": 2.0
    }
}

# -----------------------------------------------------------------------------
# 1. DSP ENGINE & JEOL UNIT CONVERSION
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
    y_arr = np.asarray(y, dtype=np.float64).flatten()
    L = len(y_arr)
    if L < 3:
        return np.zeros_like(y_arr)

    D = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), format="csc")
    D_T_D = D.T.dot(D)

    w = np.ones(L, dtype=np.float64)
    for _ in range(niter):
        W = sp.spdiags(w, 0, L, L, format="csc")
        Z = W + (lam * D_T_D)
        z = spsolve(Z, w * y_arr)
        w = p * (y_arr > z) + (1.0 - p) * (y_arr <= z)
        
    return z

def get_jeol_ppm_axis(dic: dict, data_len: int, spec_freq_mhz: float = 400.0, nucleus: str = "1H") -> np.ndarray:
    """Extracts the true PPM scale from JEOL Delta headers or native unit converters."""
    # 1. Native nmrglue unit conversion
    try:
        uc = ng.jeol.make_uc(dic)
        ppm = uc.ppm_scale()
        if len(ppm) == data_len and (np.max(ppm) - np.min(ppm)) > 4.0:
            return np.asarray(ppm, dtype=np.float64)
    except Exception:
        pass

    # 2. Universal dictionary inspection with unit normalization
    try:
        udic = ng.jeol.guess_udic(dic, np.zeros(data_len))
        obs_mhz = udic[0]['obs']
        if obs_mhz > 1e4:
            obs_mhz = obs_mhz / 1e6
        if obs_mhz <= 0:
            obs_mhz = spec_freq_mhz

        sw_hz = udic[0]['sw']
        if sw_hz < 50.0:  # If SW was parsed in ppm/kHz
            sw_hz = sw_hz * obs_mhz

        car_hz = udic[0].get('car', 0.0)
        if car_hz > 1e5:
            car_hz = car_hz - (obs_mhz * 1e6)

        uc = ng.fileiobase.unit_conversion(data_len, False, sw_hz, obs_mhz, car_hz)
        ppm = uc.ppm_scale()
        if len(ppm) == data_len and (np.max(ppm) - np.min(ppm)) > 4.0:
            return np.asarray(ppm, dtype=np.float64)
    except Exception:
        pass

    # 3. Direct header parameters fallback
    try:
        head = dic.get('head', {})
        sw_hz = head.get('x_sweep', head.get('sweep', None))
        base_freq = head.get('x_freq', head.get('base_freq', None))
        x_offset = head.get('x_offset', head.get('offset', 0.0))

        if sw_hz is not None and base_freq is not None:
            sw_hz = float(sw_hz)
            bf_mhz = float(base_freq) / 1e6 if float(base_freq) > 1e4 else float(base_freq)
            sw_ppm = sw_hz / bf_mhz
            center_ppm = float(x_offset) / bf_mhz if float(x_offset) > 100 else float(x_offset)
            if center_ppm == 0.0:
                center_ppm = 7.0 if nucleus == "1H" else 100.0
            return np.linspace(center_ppm + sw_ppm / 2.0, center_ppm - sw_ppm / 2.0, data_len)
    except Exception:
        pass

    # 4. Standard full spectral scale fallback
    default_max = 16.0 if nucleus == "1H" else 230.0
    default_min = -1.0 if nucleus == "1H" else -10.0
    return np.linspace(default_max, default_min, data_len)

def process_fid(fid_data: np.ndarray, dic: dict, solvent: str = "DMSO-d6", nucleus: str = "1H", spec_freq_mhz: float = 400.0) -> tuple:
    """Processes JEOL .jdf files with automatic domain detection and solvent lock calibration."""
    data = np.squeeze(fid_data)
    is_complex = np.iscomplexobj(data) or (data.ndim > 1 and data.shape[-1] == 2)

    if is_complex:
        # Time-domain raw FID processing
        sw = dic.get("sw", spec_freq_mhz * 16.0)
        data_em = ng.proc_base.em(data, lb=0.5 / sw if sw > 0 else 0.001)
        data_zf = ng.proc_base.zf(data_em, pad=len(data_em))
        complex_spec = ng.proc_base.fft(data_zf)
        phased = autophase_entropy(complex_spec)
        baseline = baseline_als(phased, lam=1e6, p=0.005)
        corrected = np.asarray(phased - baseline, dtype=np.float64)
    else:
        # Already Fourier-transformed frequency-domain spectrum from JEOL Delta
        spec_real = np.real(data).astype(np.float64)
        baseline = baseline_als(spec_real, lam=1e6, p=0.005)
        corrected = np.asarray(spec_real - baseline, dtype=np.float64)

    n_points = len(corrected)
    raw_ppm = get_jeol_ppm_axis(dic, n_points, spec_freq_mhz=spec_freq_mhz, nucleus=nucleus)

    # Automated Deuterated Solvent Referencing (Zero-Point Lock)
    ref = SOLVENT_TABLE.get(solvent, SOLVENT_TABLE["DMSO-d6"])
    target_ppm = ref.get(nucleus)
    tol = ref.get(f"{nucleus}_tol", 0.35)

    if target_ppm is not None:
        mask = (raw_ppm >= target_ppm - tol) & (raw_ppm <= target_ppm + tol)
        if np.any(mask):
            local_max_idx = np.where(mask)[0][np.argmax(corrected[mask])]
            observed_solvent_ppm = raw_ppm[local_max_idx]
            offset = target_ppm - observed_solvent_ppm
            raw_ppm = raw_ppm + offset

    return raw_ppm, corrected

def filter_solvent_peaks(peaks: list, solvent: str = "DMSO-d6", nucleus: str = "1H") -> list:
    ref = SOLVENT_TABLE.get(solvent, {})
    s_ppm = ref.get(nucleus)
    w_ppm = ref.get("water_1H") if nucleus == "1H" else None

    clean = []
    for p in peaks:
        val = p["ppm"]
        if s_ppm and abs(val - s_ppm) < (0.05 if nucleus == "1H" else 0.8):
            continue
        if w_ppm and abs(val - w_ppm) < 0.08:
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
        a, c, g = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        y += (a / np.pi) * (g / ((x - c) ** 2 + g ** 2))
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
    ppm_arr = np.asarray(ppm_axis, dtype=np.float64).flatten()
    spec_arr = np.asarray(spectrum, dtype=np.float64).flatten()

    if len(ppm_arr) != len(spec_arr):
        ppm_arr = np.linspace(ppm_arr[0], ppm_arr[-1], len(spec_arr))

    threshold = np.max(spec_arr) * (0.025 if nucleus == "1H" else 0.04)
    cluster_gap = 0.08 if nucleus == "1H" else 1.0

    p_indices, _ = find_peaks(spec_arr, height=threshold, distance=4)
    p_indices = p_indices[p_indices < len(ppm_arr)]
    if len(p_indices) == 0:
        return []

    clusters, curr = [], [p_indices[0]]
    for i in range(1, len(p_indices)):
        if abs(ppm_arr[p_indices[i]] - ppm_arr[p_indices[i - 1]]) <= cluster_gap:
            curr.append(p_indices[i])
        else:
            clusters.append(curr)
            curr = [p_indices[i]]
    clusters.append(curr)

    results = []
    hz_axis = ppm_arr * spec_freq

    for cl in clusters:
        mn = max(0, min(cl) - 20)
        mx = min(len(ppm_arr), max(cl) + 20)
        if mn >= mx:
            continue

        p_sub = ppm_arr[mn:mx]
        h_sub = hz_axis[mn:mx]
        s_sub = spec_arr[mn:mx]

        sub_pks, _ = find_peaks(s_sub, prominence=np.max(s_sub) * 0.05)
        init = []
        for idx in sub_pks:
            init.extend([s_sub[idx] * np.pi * 0.5, h_sub[idx], 0.5])
        init.append(np.min(s_sub))

        try:
            popt, _ = curve_fit(multi_lorentzian, h_sub, s_sub, p0=init, maxfev=1500)
            fitted_peaks = [{"ppm": popt[3 * i + 1] / spec_freq, "hz": popt[3 * i + 1]} for i in range((len(popt) - 1) // 3)]
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
# 3. FULL-SCALE PREDICTED 1D SPECTRUM SYNTHESIZER
# -----------------------------------------------------------------------------
def generate_predicted_spectrum(
    pred_df,
    spec_freq_mhz: float = 400.0,
    nucleus: str = "1H",
    num_points: int = 5000
) -> tuple:
    """Synthesizes an in silico spectrum spanning the full standard range (-1 to 16 ppm)."""
    ppm_max = 16.0 if nucleus == "1H" else 230.0
    ppm_min = -1.0 if nucleus == "1H" else -10.0
    ppm_axis = np.linspace(ppm_max, ppm_min, num_points)

    if pred_df.empty:
        return ppm_axis, np.zeros_like(ppm_axis), []

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
        else:
            offsets_hz = [0.0]
            weights = [1.0]

        gamma_use = (2.5 / spec_freq_mhz) / 2.0 if mult == "br s" else gamma

        for off_hz, w in zip(offsets_hz, weights):
            sub_ppm = c_shift + (off_hz / spec_freq_mhz)
            area = protons * w
            sim_spectrum += area * (1.0 / np.pi) * (gamma_use / ((ppm_axis - sub_ppm) ** 2 + gamma_use ** 2))

        annotations.append({
            "label": lbl,
            "ppm": c_shift,
            "mult": mult,
            "protons": int(protons) if nucleus == "1H" else None
        })

    return ppm_axis, sim_spectrum, annotations
