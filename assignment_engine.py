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
    # 1. Expand 1H virtual nodes
    pred_h_nodes = []
    for _, r in pred_h_df.iterrows():
        n_p = int(r.get("Protons", 1))
        for sub in range(n_p):
            pred_h_nodes.append({"label": r["Label"], "shift": r["Shift"], "mult": r.get("Mult", "m"), "tot": n_p})

    exp_h_nodes = []
    for p in exp_1h_peaks:
        n_p = int(p.get("protons", 1))
        for _ in range(n_p):
            exp_h_nodes.append(p)

    # 2. 1H Assignment Matrix
    dim_h = max(len(pred_h_nodes), len(exp_h_nodes))
    cost_h = np.full((dim_h, dim_h), 35.0)

    for i, pn in enumerate(pred_h_nodes):
        for j, en in enumerate(exp_h_nodes):
            diff = abs(pn["shift"] - en["ppm"])
            cost_h[i, j] = 2.0 * (diff ** 2)

    # 3. Apply COSY Topological Relaxation Penalties
    if exp_cosy_peaks and len(exp_cosy_peaks) > 0:
        for _ in range(3):
            r_idx, c_idx = linear_sum_assignment(cost_h)
            assigned_map = {pred_h_nodes[r]["label"]: exp_h_nodes[c]["ppm"] for r, c in zip(r_idx, c_idx) if r < len(pred_h_nodes) and c < len(exp_h_nodes)}

            for h1, h2 in topo_2d.get("COSY", []):
                if h1 in assigned_map and h2 in assigned_map:
                    s1, s2 = assigned_map[h1], assigned_map[h2]
                    has_cross = any((abs(c1 - s1) < 0.08 and abs(c2 - s2) < 0.08) or (abs(c1 - s2) < 0.08 and abs(c2 - s1) < 0.08) for c1, c2 in exp_cosy_peaks)
                    if not has_cross:
                        for idx_r, node in enumerate(pred_h_nodes):
                            if node["label"] in (h1, h2):
                                cost_h[idx_r, :] += 5.0

    r_h, c_h = linear_sum_assignment(cost_h)
    h_rows = []
    for r, c in zip(r_h, c_h):
        if r < len(pred_h_nodes):
            p_node = pred_h_nodes[r]
            if c < len(exp_h_nodes):
                e_node = exp_h_nodes[c]
                h_rows.append({
                    "Atom Label": p_node["label"],
                    "Pred δ (ppm)": p_node["shift"],
                    "Exp δ (ppm)": e_node["ppm"],
                    "Range (ppm)": e_node.get("range", f"{e_node['ppm']:.2f}"),
                    "Mult.": e_node.get("multiplicity", p_node["mult"]),
                    "Integral": f"{p_node['tot']}H",
                    "Status": "Matched"
                })
            else:
                h_rows.append({
                    "Atom Label": p_node["label"],
                    "Pred δ (ppm)": p_node["shift"],
                    "Exp δ (ppm)": np.nan,
                    "Range (ppm)": "-",
                    "Mult.": "-",
                    "Integral": f"{p_node['tot']}H",
                    "Status": "Unassigned"
                })

    df_h_res = pd.DataFrame(h_rows).groupby("Atom Label", as_index=False).first()

    # 4. 13C Assignment Matrix with HMBC Correlation Bonuses
    dim_c = max(len(pred_c_df), len(exp_13c_peaks))
    cost_c = np.full((dim_c, dim_c), 50.0)

    for i, (_, pr) in enumerate(pred_c_df.iterrows()):
        for j, ep in enumerate(exp_13c_peaks):
            diff = abs(pr["Shift"] - ep["ppm"])
            c_val = 0.05 * (diff ** 2)

            if exp_hmbc_peaks and pr["IsQuat"]:
                h_linked = [h_lbl for (h_lbl, c_lbl, _) in topo_2d.get("HMBC", []) if c_lbl == pr["Label"]]
                for h_lbl in h_linked:
                    if h_lbl in df_h_res.set_index("Atom Label")["Exp δ (ppm)"].dropna():
                        h_exp = df_h_res.set_index("Atom Label").loc[h_lbl, "Exp δ (ppm)"]
                        if any(abs(hp["h_ppm"] - h_exp) < 0.06 and abs(hp["c_ppm"] - ep["ppm"]) < 2.0 for hp in exp_hmbc_peaks):
                            c_val = max(0.0, c_val - 15.0)
            cost_c[i, j] = c_val

    r_c, c_c = linear_sum_assignment(cost_c)
    c_rows = []
    for r, c in zip(r_c, c_c):
        if r < len(pred_c_df):
            p_row = pred_c_df.iloc[r]
            if c < len(exp_13c_peaks):
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

    df_c_res = pd.DataFrame(c_rows)
    return df_h_res.sort_values(by="Pred δ (ppm)", ascending=False), df_c_res.sort_values(by="Pred δ (ppm)", ascending=False)
