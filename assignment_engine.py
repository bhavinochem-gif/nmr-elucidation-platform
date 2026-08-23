import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def solve_assignment_2d(
    pred_h_df: pd.DataFrame,
    pred_c_df: pd.DataFrame,
    exp_1h_peaks: list,
    exp_13c_peaks: list,
    topo_2d: dict,
    exp_cosy_peaks: list = None,
    exp_hmbc_peaks: list = None
) -> tuple:
    # 1. 1H Matching Matrix
    n_p = len(pred_h_df)
    n_e = len(exp_1h_peaks)
    dim_h = max(n_p, n_e)
    cost_h = np.full((dim_h, dim_h), 35.0)

    for i in range(n_p):
        p_row = pred_h_df.iloc[i]
        for j in range(n_e):
            diff = abs(p_row["Shift"] - exp_1h_peaks[j]["ppm"])
            cost_h[i, j] = 2.0 * (diff ** 2)

    r_h, c_h = linear_sum_assignment(cost_h)
    h_rows = []
    for r, c in zip(r_h, c_h):
        if r < n_p:
            p_row = pred_h_df.iloc[r]
            if c < n_e:
                e_row = exp_1h_peaks[c]
                h_rows.append({
                    "Atom Label": p_row["Label"],
                    "Pred δ (ppm)": p_row["Shift"],
                    "Exp δ (ppm)": e_row["ppm"],
                    "Range (ppm)": e_row.get("range", f"{e_row['ppm']:.2f}"),
                    "Mult.": e_row.get("multiplicity", p_row["Mult"]),
                    "Integral": f"{p_row['Protons']}H",
                    "Status": "Matched"
                })
            else:
                h_rows.append({
                    "Atom Label": p_row["Label"],
                    "Pred δ (ppm)": p_row["Shift"],
                    "Exp δ (ppm)": np.nan,
                    "Range (ppm)": "-",
                    "Mult.": p_row["Mult"],
                    "Integral": f"{p_row['Protons']}H",
                    "Status": "Unassigned"
                })

    df_h_res = pd.DataFrame(h_rows).sort_values(by="Pred δ (ppm)", ascending=False).reset_index(drop=True)

    # 2. 13C Matching Matrix
    n_cp = len(pred_c_df)
    n_ce = len(exp_13c_peaks)
    dim_c = max(n_cp, n_ce)
    cost_c = np.full((dim_c, dim_c), 50.0)

    for i in range(n_cp):
        p_row = pred_c_df.iloc[i]
        for j in range(n_ce):
            diff = abs(p_row["Shift"] - exp_13c_peaks[j]["ppm"])
            cost_c[i, j] = 0.05 * (diff ** 2)

    r_c, c_c = linear_sum_assignment(cost_c)
    c_rows = []
    for r, c in zip(r_c, c_c):
        if r < n_cp:
            p_row = pred_c_df.iloc[r]
            if c < n_ce:
                e_row = exp_13c_peaks[c]
                c_rows.append({
                    "Atom Label": p_row["Label"],
                    "Type": "Quaternary" if p_row["IsQuat"] else "CH/CH2/CH3",
                    "Pred δ (ppm)": p_row["Shift"],
                    "Exp δ (ppm)": e_row["ppm"],
                    "Status": "Matched"
                })
            else:
                c_rows.append({
                    "Atom Label": p_row["Label"],
                    "Type": "Quaternary" if p_row["IsQuat"] else "CH/CH2/CH3",
                    "Pred δ (ppm)": p_row["Shift"],
                    "Exp δ (ppm)": np.nan,
                    "Status": "Unassigned"
                })

    df_c_res = pd.DataFrame(c_rows).sort_values(by="Pred δ (ppm)", ascending=False).reset_index(drop=True)

    return df_h_res, df_c_res
