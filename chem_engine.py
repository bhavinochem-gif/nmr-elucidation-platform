from io import BytesIO
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

def classify_topicity(mol_h, h1_idx: int, h2_idx: int) -> str:
    m_a = Chem.Mol(mol_h)
    m_a.GetAtomWithIdx(h1_idx).SetIsotope(2)
    Chem.AssignStereochemistry(m_a, cleanIt=True, force=True)

    m_b = Chem.Mol(mol_h)
    m_b.GetAtomWithIdx(h2_idx).SetIsotope(2)
    Chem.AssignStereochemistry(m_b, cleanIt=True, force=True)

    if Chem.MolToInchi(m_a) == Chem.MolToInchi(m_b):
        return "Homotopic"
    if len(Chem.FindMolChiralCenters(m_a, includeUnassigned=True)) > 1:
        return "Diastereotopic"
    return "Enantiotopic"

def analyze_and_number_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, {}, {}, {}, []

    mol_h = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol_h, cleanIt=True, force=True)

    c_map, h_map, diastereotopic = {}, {}, []
    c_count = 1
    
    # 1. Number Carbons
    for a in mol_h.GetAtoms():
        if a.GetSymbol() == "C":
            c_map[a.GetIdx()] = f"C{c_count}"
            c_count += 1

    # 2. Number and group Protons
    hetero_count = {"O": 1, "N": 1, "S": 1}
    for a in mol_h.GetAtoms():
        if a.GetSymbol() == "C":
            c_lbl = c_map[a.GetIdx()]
            h_nbrs = [n.GetIdx() for n in a.GetNeighbors() if n.GetSymbol() == "H"]
            
            # Check for diastereotopic methylene protons
            if len(h_nbrs) == 2 and classify_topicity(mol_h, h_nbrs[0], h_nbrs[1]) == "Diastereotopic":
                h_map[h_nbrs[0]] = f"{c_lbl}-Ha"
                h_map[h_nbrs[1]] = f"{c_lbl}-Hb"
                diastereotopic.append((f"{c_lbl}-Ha", f"{c_lbl}-Hb"))
            else:
                for idx in h_nbrs:
                    h_map[idx] = f"{c_lbl}-H"
                    
        elif a.GetSymbol() in ("O", "N", "S"):
            sym = a.GetSymbol()
            h_nbrs = [n.GetIdx() for n in a.GetNeighbors() if n.GetSymbol() == "H"]
            for idx in h_nbrs:
                h_map[idx] = f"{sym}{hetero_count[sym]}-H"
            if h_nbrs:
                hetero_count[sym] += 1

    # 3. Extract 2D NMR correlation topology
    dist = Chem.GetDistanceMatrix(mol_h)
    topo_2d = {"HSQC": [], "COSY": [], "HMBC": []}

    for h_idx, h_lbl in h_map.items():
        for c_idx, c_lbl in c_map.items():
            d = dist[h_idx, c_idx]
            if d == 1:
                topo_2d["HSQC"].append((h_lbl, c_lbl))
            elif d in (2, 3):
                topo_2d["HMBC"].append((h_lbl, c_lbl, int(d)))

    for h1_idx, h1_lbl in h_map.items():
        for h2_idx, h2_lbl in h_map.items():
            if h1_idx < h2_idx and dist[h1_idx, h2_idx] == 3:
                topo_2d["COSY"].append((h1_lbl, h2_lbl))

    return mol, mol_h, c_map, h_map, topo_2d, diastereotopic

def draw_molecule_annotated(mol, atom_labels: dict) -> BytesIO:
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 420)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 2

    for idx, lbl in atom_labels.items():
        opts.atomLabels[idx] = lbl

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return BytesIO(drawer.GetDrawingText())
