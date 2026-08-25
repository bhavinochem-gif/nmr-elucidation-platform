import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def solve_assignment_2d(
    pred_h_df: pd.DataFrame,
    pred_c_df: pd.DataFrame,
    exp_1h_peaks: list = None,
    exp_13c_peaks: list = None,
    topo_2d: dict = None,
    exp_cosy_peaks: list = None,
    exp_hmbc_peaks: list = None
) -> tuple:
    """
    Matches predicted vs experimental shifts using Rectangular Linear Sum Assignment.
    Guarantees optimal global pairing without artificial dummy cost dropouts.
    """
    exp_1h = exp_1h_peaks if exp_1h_peaks is not None else []
    exp_13c = exp_13c_peaks if exp_13c_peaks is not None else []

    # =========================================================================
    # 1. 1H ASSIGNMENT MATRIX
    # =========================================================================
    if pred_h_df is None or pred_h_df.empty:
        df_h_res = pd.DataFrame(columns=["Atom Label", "Pred δ (ppm)", "Exp δ (ppm)", "Range (ppm)", "Mult.", "Integral", "Status"])
    else:
        n_p = len(pred_h_df)
        n_e = len(exp_1h)
        matched_exp_for_pred = {}

        if n_e > 0 and n_p > 0:
            # Rectangular Cost Matrix: (n_predicted x n_experimental)
            cost_h = np.zeros((n_p, n_e), dtype=np.float64)
            for i in range(n_p):
                p_shift = float(pred_h_df.iloc[i]["Shift"])
                for j in range(n_e):
                    diff = abs(p_shift - float(exp_1h[j]["ppm"]))
                    cost_h[i, j] = diff ** 2

            # Solve rectangular optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_h)
            for r, c in zip(row_ind, col_ind):
                matched_exp_for_pred[r] = exp_1h[c]

        # Build assignment table for all predicted protons
        h_rows = []
        for i in range(n_p):
            p_row = pred_h_df.iloc[i]
            if i in matched_exp_for_pred:
                e_match = matched_exp_for_pred[i]
                exp_val = float(e_match["ppm"])
                h_rows.append({
                    "Atom Label": str(p_row["Label"]),
                    "Pred δ (ppm)": round(float(p_row["Shift"]), 2),
                    "Exp δ (ppm)": round(exp_val, 2),
                    "Range (ppm)": str(e_match.get("range", f"{exp_val:.2f}")),
                    "Mult.": str(e_match.get("multiplicity", p_row.get("Mult", "s"))),
                    "Integral": f"{p_row.get('Protons', 1)}H",
                    "Status": "Matched"
                })
            else:
                h_rows.append({
                    "Atom Label": str(p_row["Label"]),
                    "Pred δ (ppm)": round(float(p_row["Shift"]), 2),
                    "Exp δ (ppm)": np.nan,
                    "Range (ppm)": "-",
                    "Mult.": str(p_row.get("Mult", "s")),
                    "Integral": f"{p_row.get('Protons', 1)}H",
                    "Status": "Unassigned"
                })

        df_h_res = pd.DataFrame(h_rows).sort_values(by="Pred δ (ppm)", ascending=False).reset_index(drop=True)

    # =========================================================================
    # 2. 13C ASSIGNMENT MATRIX
    # =========================================================================
    if pred_c_df is None or pred_c_df.empty:
        df_c_res = pd.DataFrame(columns=["Atom Label", "Type", "Pred δ (ppm)", "Exp δ (ppm)", "Status"])
    else:
        n_cp = len(pred_c_df)
        n_ce = len(exp_13c)
        matched_exp_for_c = {}

        if n_ce > 0 and n_cp > 0:
            cost_c = np.zeros((n_cp, n_ce), dtype=np.float64)
            for i in range(n_cp):
                p_shift = float(pred_c_df.iloc[i]["Shift"])
                for j in range(n_ce):
                    diff = abs(p_shift - float(exp_13c[j]["ppm"]))
                    cost_c[i, j] = diff ** 2

            row_ind_c, col_ind_c = linear_sum_assignment(cost_c)
            for r, c in zip(row_ind_c, col_ind_c):
                matched_exp_for_c[r] = exp_13c[c]

        c_rows = []
        for i in range(n_cp):
            p_row = pred_c_df.iloc[i]
            is_quat = bool(p_row.get("IsQuat", False))
            c_type = "Quaternary" if is_quat else "CH/CH2/CH3"

            if i in matched_exp_for_c:
                e_match = matched_exp_for_c[i]
                exp_val = float(e_match["ppm"])
                c_rows.append({
                    "Atom Label": str(p_row["Label"]),
                    "Type": c_type,
                    "Pred δ (ppm)": round(float(p_row["Shift"]), 1),
                    "Exp δ (ppm)": round(exp_val, 1),
                    "Status": "Matched"
                })
            else:
                c_rows.append({
                    "Atom Label": str(p_row["Label"]),
                    "Type": c_type,
                    "Pred δ (ppm)": round(float(p_row["Shift"]), 1),
                    "Exp δ (ppm)": np.nan,
                    "Status": "Unassigned"
                })

        df_c_res = pd.DataFrame(c_rows).sort_values(by="Pred δ (ppm)", ascending=False).reset_index(drop=True)

    return df_h_res, df_c_res
