import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem

SOLVENT_VECTORS = {
    "CDCl3": [1, 0, 0, 0, 0],
    "DMSO-d6": [0, 1, 0, 0, 0],
    "Methanol-d4": [0, 0, 1, 0, 0],
    "Acetone-d6": [0, 0, 0, 1, 0],
    "D2O": [0, 0, 0, 0, 1]
}

class MPNNLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.upd = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x, edge_index, edge_weight):
        row, col = edge_index
        m = self.msg(torch.cat([x[row], x[col], edge_weight.unsqueeze(-1)], dim=-1))
        aggr = torch.zeros((x.size(0), x.size(-1)), device=x.device)
        aggr.index_add_(0, col, m)
        return x + self.upd(torch.cat([x, aggr], dim=-1))

class NMRShiftGNN(nn.Module):
    def __init__(self, in_dim: int = 11, solvent_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.embed_node = nn.Linear(in_dim, hidden_dim)
        self.embed_solv = nn.Linear(solvent_dim, hidden_dim)
        self.conv1 = MPNNLayer(hidden_dim)
        self.conv2 = MPNNLayer(hidden_dim)
        self.conv3 = MPNNLayer(hidden_dim)
        self.out_1h = nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        self.out_13c = nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, edge_index, edge_weight, solv_vec):
        h = self.embed_node(x) + self.embed_solv(solv_vec).unsqueeze(0)
        h = self.conv1(h, edge_index, edge_weight)
        h = self.conv2(h, edge_index, edge_weight)
        h = self.conv3(h, edge_index, edge_weight)
        return self.out_1h(h) * 15.0, self.out_13c(h) * 230.0

class ShiftPredictor:
    def __init__(self, weights_path: str = "gnn_nmr_weights.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NMRShiftGNN().to(self.device)
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def predict(self, mol_h, c_map: dict, h_map: dict, solvent: str = "CDCl3"):
        nodes = []
        for a in mol_h.GetAtoms():
            hyb = [float(a.GetHybridization() == h) for h in [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ]]
            nodes.append([
                a.GetAtomicNum() / 53.0, a.GetTotalDegree() / 4.0, a.GetFormalCharge(),
                float(a.GetIsAromatic()), float(a.IsInRing()), a.GetTotalNumHs() / 4.0
            ] + hyb)

        edges, weights = [], []
        for b in mol_h.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            w = b.GetBondTypeAsDouble()
            edges.extend([[i, j], [j, i]])
            weights.extend([w, w])

        x = torch.tensor(nodes, dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_weight = torch.tensor(weights, dtype=torch.float32).to(self.device)
        solv = torch.tensor(SOLVENT_VECTORS.get(solvent, [1, 0, 0, 0, 0]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            p_1h, p_13c = self.model(x, edge_index, edge_weight, solv)

        h_preds = [{"Label": lbl, "Shift": max(round(float(p_1h[idx].item()), 2), 0.0), "Protons": 1, "Mult": "m"} for idx, lbl in h_map.items()]
        c_preds = [{"Label": lbl, "Shift": max(round(float(p_13c[idx].item()), 1), 0.0), "IsQuat": len([n for n in mol_h.GetAtomWithIdx(idx).GetNeighbors() if n.GetSymbol() == 'H']) == 0} for idx, lbl in c_map.items()]

        return pd.DataFrame(h_preds), pd.DataFrame(c_preds)
