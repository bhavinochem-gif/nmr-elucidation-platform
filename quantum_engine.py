import numpy as np
import pandas as pd

Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
Id = np.eye(2, dtype=complex)

def solve_quantum_spin_system(shifts_ppm: list, j_matrix: np.ndarray, spec_freq: float = 400.0) -> tuple:
    n_spins = len(shifts_ppm)
    shifts_hz = [s * spec_freq for s in shifts_ppm]
    dim = 2 ** n_spins

    Ix, Iy, Iz = [], [], []
    for i in range(n_spins):
        ox, oy, oz = 1.0, 1.0, 1.0
        for j in range(n_spins):
            ox = np.kron(ox, Sx if i == j else Id)
            oy = np.kron(oy, Sy if i == j else Id)
            oz = np.kron(oz, Sz if i == j else Id)
        Ix.append(ox)
        Iy.append(oy)
        Iz.append(oz)

    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_spins):
        H += shifts_hz[i] * Iz[i]

    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            j_val = j_matrix[i, j]
            if abs(j_val) > 1e-4:
                H += j_val * (Ix[i] @ Ix[j] + Iy[i] @ Iy[j] + Iz[i] @ Iz[j])

    eigenvalues, V = np.linalg.eigh(H)
    Fx = sum(Ix)
    Fx_eigen = V.conj().T @ Fx @ V

    transitions = []
    for k in range(dim):
        for l in range(k + 1, dim):
            intensity = np.abs(Fx_eigen[k, l]) ** 2
            if intensity > 1e-5:
                freq_hz = np.abs(eigenvalues[l] - eigenvalues[k])
                transitions.append({"ppm": freq_hz / spec_freq, "intensity": intensity})

    df_trans = pd.DataFrame(transitions).sort_values(by="ppm", ascending=False)
    
    # Synthesize lineshape
    center = np.mean(shifts_ppm)
    ppm_axis = np.linspace(center + 0.15, center - 0.15, 2048)
    gamma = (0.5 / spec_freq) / 2.0
    spectrum = np.zeros_like(ppm_axis)

    for _, row in df_trans.iterrows():
        spectrum += row["intensity"] * (1.0 / np.pi) * (gamma / ((ppm_axis - row["ppm"])**2 + gamma**2))

    return df_trans, ppm_axis, spectrum
