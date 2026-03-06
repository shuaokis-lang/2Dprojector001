# =========================================
# LigPocket Surface Map v10
# バッジ小型化 + 番号黒字
# =========================================

import os, sys, math, subprocess, urllib.request, warnings
import xml.etree.ElementTree as ET
import numpy as np
warnings.filterwarnings("ignore")

def log(msg): print(msg, flush=True)
def run(cmd, *, check=True):
    p = subprocess.run(cmd, shell=isinstance(cmd,str),
                       text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"[ERROR]\n{p.stderr[-3000:]}")
    return p

# ══════════════════════════════════════════
# STEP 0: ツールチェック
# ══════════════════════════════════════════
log("==[0/8]== ツール・パッケージの確認中…")

def pip_installed(mod):
    import importlib
    try: importlib.import_module(mod); return True
    except ImportError: return False

def check_cmd(cmd):
    return run(f"which {cmd}", check=False).returncode == 0

py_pkgs = {
    "biopython":    "Bio",
    "rdkit":        "rdkit",
    "matplotlib":   "matplotlib",
    "numpy":        "numpy",
    "scipy":        "scipy",
    "scikit-learn": "sklearn",
    "shapely":      "shapely",
    "Pillow":       "PIL",
    "scikit-image": "skimage",
    "requests":     "requests",
}
for pkg, mod in py_pkgs.items():
    if pip_installed(mod): log(f"  ✓ {pkg}")
    else:
        log(f"  ✗ {pkg} → インストール中")
        run(f"pip -q install {pkg}")

if check_cmd("obabel"): log("  ✓ OpenBabel")
else:
    log("  ✗ OpenBabel → インストール中")
    run("apt-get -qq update && apt-get -qq install -y openbabel > /dev/null")

MAMBA = "/usr/local/bin/micromamba"
if os.path.exists(MAMBA): log("  ✓ micromamba")
else:
    log("  ✗ micromamba → インストール中")
    run("curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest"
        " | tar -xvj bin/micromamba")
    run("mv -f bin/micromamba /usr/local/bin/micromamba"
        " && chmod +x /usr/local/bin/micromamba")

ENV_PREFIX = "/content/plipenv"
def plip_cmd(args):
    return f"{MAMBA} run -p {ENV_PREFIX} plip " + " ".join(args)

plip_ok = (os.path.exists(os.path.join(ENV_PREFIX,"conda-meta")) and
           run(plip_cmd(["--version"]), check=False).returncode == 0)
if plip_ok: log("  ✓ PLIP")
else:
    log("  ✗ PLIP → インストール中")
    os.makedirs(ENV_PREFIX, exist_ok=True)
    if not os.path.exists(os.path.join(ENV_PREFIX,"conda-meta")):
        run(f"{MAMBA} create -y -p {ENV_PREFIX} -c conda-forge plip > /dev/null")
    else:
        run(f"{MAMBA} install -y -p {ENV_PREFIX} -c conda-forge plip > /dev/null")
log("==[0/8]== チェック完了\n")

# ── import ──
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch, Circle
from scipy.spatial import ConvexHull, cKDTree
from sklearn.decomposition import PCA
from skimage import measure
from shapely.geometry import Polygon as SPoly, MultiPolygon, Point
from shapely.ops import unary_union
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image
import requests, textwrap

# ══════════════════════════════════════════
# User knobs
# ══════════════════════════════════════════
PDB_ID            = "1HSG"
LIGAND_KEY        = None
POCKET_CUTOFF_A   = 5.0
SASA_PROBE_RADIUS = 1.40
SASA_N_POINTS     = 960
N_SPHERE          = 80
GRID_RES          = 500

# ══════════════════════════════════════════
# ★ バッジ調整パラメータ（ここだけ変更すれば全体に反映）
# ══════════════════════════════════════════
MAP_BADGE_SCALE   = 0.022   # ポケットマップのバッジ半径（データ座標比）
LIG_BADGE_FRAC    = 0.030   # リガンド画像のバッジ半径（画像幅比）
BADGE_ALPHA       = 0.72    # バッジ背景の透明度
BADGE_EDGE_W      = 1.2     # バッジ枠線の太さ
BADGE_NUM_FS_MAP  = 9.5     # マップ側番号フォントサイズ
BADGE_NUM_FS_LIG  = 11.0    # リガンド側番号フォントサイズ
BADGE_NUM_COLOR   = "#111111"  # 番号テキスト色（黒）

# ══════════════════════════════════════════
# 色・相互作用メタ情報
# ══════════════════════════════════════════
ITYPE_META = {
    "hydrophobic_interactions": {
        "color":"#f4a261","label":"Hydrophobic",
        "desc":"疎水性相互作用\n非極性残基とリガンド疎水部位"},
    "hydrogen_bonds": {
        "color":"#2a9d8f","label":"Hydrogen Bond",
        "desc":"水素結合  N–H···O / O–H···N\n距離<3.5Å, 角度>120°"},
    "water_bridges": {
        "color":"#48cae4","label":"Water Bridge",
        "desc":"水分子を介した水素結合\nネットワーク"},
    "salt_bridges": {
        "color":"#e63946","label":"Salt Bridge",
        "desc":"静電的相互作用\n荷電残基とリガンド荷電基"},
    "pi_stacks": {
        "color":"#8338ec","label":"π–π Stack",
        "desc":"芳香環スタッキング\n平行/T字型  距離<5.5Å"},
    "pi_cation_interactions": {
        "color":"#6d597a","label":"π–Cation",
        "desc":"芳香環と陽イオンの\n静電的相互作用"},
    "halogen_bonds": {
        "color":"#457b9d","label":"Halogen Bond",
        "desc":"ハロゲン結合  C–X···O/N/S\n(X=F,Cl,Br,I)"},
    "metal_complexes": {
        "color":"#b08968","label":"Metal Complex",
        "desc":"金属配位結合\n金属イオンを介した配位"},
    "none": {
        "color":"#dddddd","label":"Pocket Wall",
        "desc":"直接相互作用なし\nポケット形成残基"},
}
PRIORITY = ["salt_bridges","metal_complexes","hydrogen_bonds","water_bridges",
            "pi_cation_interactions","pi_stacks","halogen_bonds",
            "hydrophobic_interactions"]
AA3D = {"ALA":"Ala","ARG":"Arg","ASN":"Asn","ASP":"Asp","CYS":"Cys",
        "GLN":"Gln","GLU":"Glu","GLY":"Gly","HIS":"His","ILE":"Ile",
        "LEU":"Leu","LYS":"Lys","MET":"Met","PHE":"Phe","PRO":"Pro",
        "SER":"Ser","THR":"Thr","TRP":"Trp","TYR":"Tyr","VAL":"Val"}
def aa_label(rn,nr):
    return f"{AA3D.get(rn.upper(),rn.capitalize())}{nr}"

# ══════════════════════════════════════════
# STEP 1: PDB
# ══════════════════════════════════════════
log(f"==[1/8]== PDB {PDB_ID} を確認中…")
pdb_path = f"{PDB_ID}.pdb"
if os.path.exists(pdb_path) and os.path.getsize(pdb_path)>10240:
    log(f"  ✓ 既存ファイルを使用 ({os.path.getsize(pdb_path)//1024} KB)")
else:
    urllib.request.urlretrieve(
        f"https://files.rcsb.org/download/{PDB_ID}.pdb", pdb_path)
    log(f"  → DL完了: {os.path.getsize(pdb_path)/1024:.1f} KB")

# ══════════════════════════════════════════
# STEP 2: PLIP
# ══════════════════════════════════════════
log("==[2/8]== PLIP 解析中…")
plip_out = "plip_out"
os.makedirs(plip_out, exist_ok=True)
for f in os.listdir(plip_out):
    try: os.remove(os.path.join(plip_out,f))
    except: pass
p2 = subprocess.run(plip_cmd(["--help"]),
                    shell=True,text=True,capture_output=True)
flag = "--json" if "--json" in (p2.stdout+p2.stderr) else "--xml"
run(plip_cmd(["-f",pdb_path,"-o",plip_out,flag]))
xml_files = sorted([os.path.join(plip_out,f)
                    for f in os.listdir(plip_out) if f.endswith(".xml")])
if not xml_files: raise RuntimeError("XML not found")

def _txt(el,*tags):
    for t in tags:
        c=el.find(t)
        if c is not None and c.text and c.text.strip(): return c.text.strip()
    return None

ITYPE_CONFIG = {
    "hydrophobic_interactions":"hydrophobic_interaction",
    "hydrogen_bonds":"hydrogen_bond",
    "water_bridges":"water_bridge",
    "salt_bridges":"salt_bridge",
    "pi_stacks":"pi_stack",
    "pi_cation_interactions":"pi_cation_interaction",
    "halogen_bonds":"halogen_bond",
    "metal_complexes":"metal_complex",
}

root = ET.parse(xml_files[0]).getroot()
ligand_entries = []
for bs in root.findall("bindingsite"):
    ident = bs.find("identifiers")
    key = (f"{_txt(ident,'hetid') or 'UNK'}:"
           f"{_txt(ident,'chain') or '?'}:"
           f"{_txt(ident,'position') or '0'}")
    ligand_entries.append((key,bs))
if not ligand_entries: raise RuntimeError("リガンド未検出")
ligand_key, target_bs = (ligand_entries[0] if LIGAND_KEY is None
    else next(e for e in ligand_entries if e[0]==LIGAND_KEY))
log(f"  - リガンド: {ligand_key}")

interactions = []
iel = target_bs.find("interactions")
if iel is not None:
    for itype,item_tag in ITYPE_CONFIG.items():
        cont = iel.find(itype)
        if cont is None: continue
        for item in cont.findall(item_tag):
            chain   = _txt(item,"reschain","protchain","chain")
            resnr   = _txt(item,"resnr","resnum")
            resname = _txt(item,"restype","resname","aa") or ""
            lig_serial = None
            for tag in ["ligatom","lig_atom","ligcarbonidx",
                        "ligandatom","ligatom_orig_idx"]:
                v = _txt(item,tag)
                if v:
                    try: lig_serial=int(v); break
                    except: pass
            if chain is None or resnr is None: continue
            interactions.append({
                "itype":itype,"chain":chain,"resnr":int(resnr),
                "resname":resname,"res_key":f"{chain}:{int(resnr)}",
                "res_label":aa_label(resname,resnr),
                "lig_serial":lig_serial,
            })

res_itype = {}
for iact in interactions:
    rk=iact["res_key"]; cur=res_itype.get(rk)
    if cur is None or PRIORITY.index(iact["itype"])<PRIORITY.index(cur):
        res_itype[rk]=iact["itype"]
res_label_map = {iact["res_key"]:iact["res_label"] for iact in interactions}
log(f"  - 相互作用残基数: {len(res_itype)}")

ident_el   = target_bs.find("identifiers")
smiles_str = _txt(ident_el,"smiles")
if smiles_str is None: raise RuntimeError("SMILES not found")

smiles_to_pdb={}
mappings=target_bs.find("mappings")
if mappings is not None:
    s2p=mappings.find("smiles_to_pdb")
    if s2p is not None and s2p.text:
        for pair in s2p.text.strip().split(","):
            pair=pair.strip()
            if ":" in pair:
                try: si,pi=pair.split(":"); smiles_to_pdb[int(si)]=int(pi)
                except: pass
pdb_to_smiles={v:k for k,v in smiles_to_pdb.items()}

# ══════════════════════════════════════════
# IUPAC名取得
# ══════════════════════════════════════════
log("  - IUPAC名をPubChemから取得中…")
iupac_name = ""
try:
    resp = requests.post(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        "/compound/smiles/cids/JSON",
        data={"smiles": smiles_str}, timeout=15)
    if resp.status_code == 200:
        cid = resp.json()["IdentifierList"]["CID"][0]
        resp2 = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug"
            f"/compound/cid/{cid}/property/IUPACName/JSON",
            timeout=15)
        if resp2.status_code == 200:
            iupac_name = (resp2.json()["PropertyTable"]
                          ["Properties"][0]["IUPACName"])
            log(f"  - 取得成功: {iupac_name[:70]}…")
except Exception as e:
    log(f"  - 取得失敗 ({e})、SMILESを代用")
    iupac_name = smiles_str

# ══════════════════════════════════════════
# STEP 3: 構造解析 + 表面点群
# ══════════════════════════════════════════
log("==[3/8]== 構造解析・SASA計算中…")
structure = PDBParser(QUIET=True).get_structure("x",pdb_path)
sr = ShrakeRupley(probe_radius=SASA_PROBE_RADIUS,n_points=SASA_N_POINTS)
sr.compute(structure,level="R")

def find_lig_atoms_fn(structure,resname):
    return [a for m in structure for ch in m for r in ch
            if r.id[0]!=" " and r.get_resname().strip()==resname
            for a in r.get_atoms()]

lig_resname = ligand_key.split(":")[0]
lig_atoms   = find_lig_atoms_fn(structure,lig_resname)
if not lig_atoms: raise RuntimeError(f"リガンド '{lig_resname}' 未検出")
lig_coords  = np.array([a.get_coord() for a in lig_atoms])
lig_center  = lig_coords.mean(0)
lig_serial_to_coord = {a.serial_number:a.get_coord() for a in lig_atoms}

def get_pocket(structure,lig_coords,cutoff):
    seen,pocket=[],[]
    for m in structure:
        for ch in m:
            for res in ch:
                if not is_aa(res,standard=False): continue
                coords=np.array([a.get_coord() for a in res.get_atoms()])
                if not coords.size: continue
                d=np.sqrt(((coords[:,None]-lig_coords[None,:])**2).sum(2).min())
                if d<=cutoff:
                    rk=f"{ch.id}:{res.id[1]}"
                    if rk not in seen:
                        pocket.append({"res_key":rk,"res":res,
                            "chain":ch.id,"resnr":res.id[1],
                            "resname":res.get_resname().strip(),
                            "sasa":float(getattr(res,"sasa",0.0) or 0.0)})
                        seen.append(rk)
    return pocket

pocket = get_pocket(structure,lig_coords,POCKET_CUTOFF_A)
log(f"  - ポケット残基数: {len(pocket)}")

VDW = {"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80,
       "H":1.20,"F":1.47,"CL":1.75,"BR":1.85,"I":1.98}
def vdw(a):
    el=(a.element or a.get_name()[0]).strip().upper()
    return VDW.get(el,1.70)

all_pc,all_pr=[],[]
for m in structure:
    for ch in m:
        for res in ch:
            if is_aa(res,standard=False):
                for a in res.get_atoms():
                    all_pc.append(a.get_coord()); all_pr.append(vdw(a))
all_pc=np.array(all_pc); all_pr=np.array(all_pr)
ptree=cKDTree(all_pc)

def fib_sphere(n,r,c):
    pts,g=[],math.pi*(3-math.sqrt(5))
    for i in range(n):
        y=1-(i/(n-1))*2; rr=math.sqrt(max(1-y*y,0)); phi=g*i
        pts.append([c[0]+r*rr*math.cos(phi),c[1]+r*y,c[2]+r*rr*math.sin(phi)])
    return np.array(pts)

def exposed_pts(res,probe=1.40,n=N_SPHERE):
    pts=[]
    for atom in res.get_atoms():
        rp=vdw(atom)+probe; ctr=atom.get_coord()
        sph=fib_sphere(n,rp,ctr)
        idxs=ptree.query_ball_point(ctr,r=rp+all_pr.max()+0.1)
        if not idxs: pts.extend(sph.tolist()); continue
        nc=all_pc[idxs]; nr=all_pr[idxs]
        for pt in sph:
            if np.all(np.linalg.norm(nc-pt,axis=1)>=nr+probe-0.01):
                pts.append(pt.tolist())
    return pts

log("==[4/8]== 露出表面点群を計算中…")
res_surface_pts={}
for info in pocket:
    pts=exposed_pts(info["res"])
    if len(pts)>=3:
        res_surface_pts[info["res_key"]]=np.array(pts)
    else:
        ca=(info["res"]["CA"].get_coord() if "CA" in info["res"]
            else np.array([a.get_coord() for a in info["res"].get_atoms()]).mean(0))
        res_surface_pts[info["res_key"]]=ca.reshape(1,3)
log("  - 完了")

# ══════════════════════════════════════════
# STEP 4: PCA 投影
# ══════════════════════════════════════════
log("==[5/8]== PCA 投影中…")
all_surf   = np.vstack(list(res_surface_pts.values()))
poc_center = all_surf.mean(0)
pca        = PCA(n_components=3).fit(all_surf-poc_center)
l2p        = poc_center-lig_center
l2p       /= np.linalg.norm(l2p)+1e-9
dots       = [abs(np.dot(pca.components_[i],l2p)) for i in range(3)]
depth_ax   = int(np.argmax(dots))
proj_ax    = [i for i in range(3) if i!=depth_ax]
ax1,ax2    = pca.components_[proj_ax[0]], pca.components_[proj_ax[1]]

def proj(pts):
    c=pts-poc_center
    return np.column_stack([c@ax1, c@ax2])

res_pts_2d    = {rk:proj(pts) for rk,pts in res_surface_pts.items()}
res_center_2d = {rk:pts.mean(0) for rk,pts in res_pts_2d.items()}

# ══════════════════════════════════════════
# STEP 5: グリッドラベルマップ → ポリゴン
# ══════════════════════════════════════════
log("==[6/8]== グリッドラベルマップ → ポリゴン生成中…")
rk_list  = list(res_pts_2d.keys())
all_2d   = np.vstack([res_pts_2d[rk] for rk in rk_list])
u_all,v_all = all_2d[:,0],all_2d[:,1]
u_min,u_max = u_all.min(),u_all.max()
v_min,v_max = v_all.min(),v_all.max()
u_pad=(u_max-u_min)*0.02; v_pad=(v_max-v_min)*0.02

ug = np.linspace(u_min-u_pad, u_max+u_pad, GRID_RES)
vg = np.linspace(v_min-v_pad, v_max+v_pad, GRID_RES)
UG,VG = np.meshgrid(ug,vg)
grid_pts = np.column_stack([UG.ravel(),VG.ravel()])

rep_pts  = np.array([res_center_2d[rk] for rk in rk_list])
rep_tree = cKDTree(rep_pts)
_,grid_labels = rep_tree.query(grid_pts)
label_grid = grid_labels.reshape(GRID_RES,GRID_RES)

try:
    hull2d    = ConvexHull(all_2d)
    from matplotlib.path import Path as MPath
    hull_path = MPath(all_2d[hull2d.vertices])
    in_hull   = hull_path.contains_points(grid_pts)
    label_grid_masked = np.where(
        in_hull.reshape(GRID_RES,GRID_RES), label_grid, -1)
except Exception:
    label_grid_masked = label_grid

voronoi_polys = {}
for i,rk in enumerate(rk_list):
    mask = (label_grid_masked==i).astype(np.float32)
    if mask.sum()<4: continue
    contours = measure.find_contours(mask, level=0.5)
    if not contours: continue
    polys=[]
    for contour in contours:
        col_idx=contour[:,1]; row_idx=contour[:,0]
        uc=u_min-u_pad+col_idx*(u_max+u_pad-(u_min-u_pad))/(GRID_RES-1)
        vc=v_min-v_pad+row_idx*(v_max+v_pad-(v_min-v_pad))/(GRID_RES-1)
        verts=np.column_stack([uc,vc])
        if len(verts)<3: continue
        try:
            p=SPoly(verts)
            if not p.is_valid: p=p.buffer(0)
            if not p.is_empty and p.area>1e-6: polys.append(p)
        except: pass
    if not polys: continue
    merged=unary_union(polys)
    if not merged.is_empty: voronoi_polys[rk]=merged

log(f"  - ポリゴン生成完了: {len(voronoi_polys)} / {len(rk_list)} 残基")

u_range=(u_max+u_pad)-(u_min-u_pad)
v_range=(v_max+v_pad)-(v_min-v_pad)
FIG_W=32.0; FIG_H=15.0; MAP_W_FRAC=0.38

def calc_fs(poly, fs_min=9.0, fs_max=18.0):
    if poly is None or poly.is_empty: return fs_min
    frac=poly.area/(u_range*v_range)
    ax_px=FIG_W*MAP_W_FRAC*100
    return float(np.clip(math.sqrt(frac)*ax_px/7.0, fs_min, fs_max))

def get_badge_pos(poly, center):
    """バッジをラベルと重ならない右上寄りに配置"""
    if poly is None or poly.is_empty: return center
    r=math.sqrt(poly.area/math.pi)
    # ラベルは重心にあるので、バッジは右上方向へ少しずらす
    for dx,dy in [(0.55,0.55),(0.55,-0.55),(-0.55,0.55),
                  (-0.55,-0.55),(0.55,0.0),(0.0,0.55)]:
        cand=center+np.array([r*dx,r*dy])
        if poly.contains(Point(cand)): return cand
    return center+np.array([r*0.4, r*0.4])

# ══════════════════════════════════════════
# STEP 6: リガンド 2D 構造 + 番号割り当て
# ══════════════════════════════════════════
log("==[7/8]== リガンド2D構造を生成中…")
mol = Chem.MolFromSmiles(smiles_str)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol,AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol)
mol = Chem.RemoveHs(mol)
AllChem.Compute2DCoords(mol)

conf = mol.GetConformer()
rdkit_atom_pos = {i:(conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y)
                  for i in range(mol.GetNumAtoms())}

# ── 番号インデックス割り当て ──
interaction_index = []
seen_rk = {}

for iact in interactions:
    rk = iact["res_key"]
    atom_idx = None
    if iact["lig_serial"] is not None:
        atom_idx = pdb_to_smiles.get(iact["lig_serial"])
    if atom_idx is None or atom_idx >= mol.GetNumAtoms():
        ri = next((p for p in pocket if p["res_key"]==rk), None)
        if ri:
            l3 = np.array([lig_serial_to_coord.get(
                a.serial_number, lig_center) for a in lig_atoms])
            r3 = np.array([a.get_coord()
                           for a in ri["res"].get_atoms()]).mean(0)
            atom_idx = int(np.argmin(np.linalg.norm(l3-r3,axis=1)))
        else:
            atom_idx = 0

    if rk not in seen_rk:
        interaction_index.append({
            "idx":       len(interaction_index)+1,
            "res_key":   rk,
            "res_label": iact["res_label"],
            "itype":     iact["itype"],
            "atom_idxs": [atom_idx],
        })
        seen_rk[rk] = len(interaction_index)-1
    else:
        ei = seen_rk[rk]
        if atom_idx not in interaction_index[ei]["atom_idxs"]:
            interaction_index[ei]["atom_idxs"].append(atom_idx)
        cur = interaction_index[ei]["itype"]
        if PRIORITY.index(iact["itype"]) < PRIORITY.index(cur):
            interaction_index[ei]["itype"] = iact["itype"]

res_to_num = {e["res_key"]: e["idx"] for e in interaction_index}
log(f"  - 番号インデックス数: {len(interaction_index)}")

# ── リガンド画像生成 ──
LIG_PX = 560
drawer = rdMolDraw2D.MolDraw2DSVG(LIG_PX, LIG_PX)
drawer.drawOptions().addStereoAnnotation = True
drawer.drawOptions().padding = 0.15
drawer.DrawMolecule(mol)
drawer.FinishDrawing()
svg_str = drawer.GetDrawingText()
try:
    import cairosvg
    lig_img = Image.open(BytesIO(
        cairosvg.svg2png(bytestring=svg_str.encode()))).convert("RGBA")
except ImportError:
    from rdkit.Chem.Draw import MolToImage
    lig_img = MolToImage(mol, size=(LIG_PX,LIG_PX))
lig_arr = np.array(lig_img)

rxs=[v[0] for v in rdkit_atom_pos.values()]
rys=[v[1] for v in rdkit_atom_pos.values()]
rxmin,rxmax=min(rxs),max(rxs); rymin,rymax=min(rys),max(rys)
iw,ih=lig_arr.shape[1],lig_arr.shape[0]
PAD_F=0.15

def r2p(rx,ry):
    uw=iw*(1-2*PAD_F); uh=ih*(1-2*PAD_F)
    px=(rx-rxmin)/(rxmax-rxmin+1e-9)*uw+iw*PAD_F
    py=(1-(ry-rymin)/(rymax-rymin+1e-9))*uh+ih*PAD_F
    return px,py

# ══════════════════════════════════════════
# STEP 7: 描画
# ══════════════════════════════════════════
log("==[8/8]== 描画中…")

fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")

ax_map = fig.add_axes([0.01, 0.05, 0.37, 0.90])
ax_lig = fig.add_axes([0.40, 0.10, 0.27, 0.80])
ax_tbl = fig.add_axes([0.69, 0.05, 0.30, 0.90])
for ax in [ax_map,ax_lig,ax_tbl]:
    ax.set_facecolor("white")

# ══ ポケットマップ ══════════════════════════
ax_map.set_xlim(u_min-u_pad*3, u_max+u_pad*3)
ax_map.set_ylim(v_min-v_pad*3, v_max+v_pad*3)
ax_map.set_aspect("equal"); ax_map.axis("off")

for gv in np.linspace(u_min-u_pad*3,u_max+u_pad*3,18):
    ax_map.axvline(gv,color="#eeeeee",lw=0.4,zorder=0)
for gv in np.linspace(v_min-v_pad*3,v_max+v_pad*3,18):
    ax_map.axhline(gv,color="#eeeeee",lw=0.4,zorder=0)

def draw_shapely(ax,poly,fc,ec,lw,ls,alpha,z):
    if poly is None or poly.is_empty: return
    if poly.geom_type=="Polygon":
        coords=np.array(poly.exterior.coords)
        if len(coords)<3: return
        ax.add_patch(MplPolygon(coords,closed=True,
            facecolor=fc,edgecolor=ec,linewidth=lw,
            linestyle=ls,alpha=alpha,zorder=z))
    elif poly.geom_type in ("MultiPolygon","GeometryCollection"):
        for g in poly.geoms: draw_shapely(ax,g,fc,ec,lw,ls,alpha,z)

# ── バッジ描画ヘルパー ──────────────────────
def draw_badge_map(ax, cx, cy, num, color, radius):
    """
    ポケットマップ用バッジ
    - 小さめの円（MAP_BADGE_SCALE × データ座標範囲）
    - 枠線あり（白）
    - 番号は黒字
    """
    # 塗り円（色付き・半透明）
    circ = Circle((cx,cy), radius=radius,
                  facecolor=color, edgecolor="white",
                  linewidth=BADGE_EDGE_W,
                  alpha=BADGE_ALPHA, zorder=9,
                  transform=ax.transData)
    ax.add_patch(circ)
    # 番号テキスト（黒・白縁取り）
    ax.text(cx, cy, str(num),
            fontsize=BADGE_NUM_FS_MAP,
            fontweight="bold",
            ha="center", va="center",
            color=BADGE_NUM_COLOR,
            zorder=10,
            path_effects=[
                pe.withStroke(linewidth=2.0, foreground="white")
            ])

def draw_badge_lig(ax, px, py, num, color, radius):
    """
    リガンド2D用バッジ
    - 小さめの円（LIG_BADGE_FRAC × 画像幅）
    - 枠線あり（白）
    - 番号は黒字
    """
    circ = Circle((px,py), radius=radius,
                  facecolor=color, edgecolor="white",
                  linewidth=BADGE_EDGE_W,
                  alpha=BADGE_ALPHA, zorder=5,
                  transform=ax.transData)
    ax.add_patch(circ)
    ax.text(px, py, str(num),
            fontsize=BADGE_NUM_FS_LIG,
            fontweight="bold",
            ha="center", va="center",
            color=BADGE_NUM_COLOR,
            zorder=6,
            path_effects=[
                pe.withStroke(linewidth=2.0, foreground="white")
            ])

# マップのバッジ半径をデータ座標で計算
map_data_range = max(u_max-u_min, v_max-v_min)
MAP_BADGE_R    = map_data_range * MAP_BADGE_SCALE
LIG_BADGE_R    = iw * LIG_BADGE_FRAC

# ── 残基ポリゴン + ラベル + バッジ ──
for info in pocket:
    rk    = info["res_key"]
    poly  = voronoi_polys.get(rk)
    if poly is None or poly.is_empty: continue
    itype = res_itype.get(rk,"none")
    color = ITYPE_META[itype]["color"]
    alpha = 0.62 if itype!="none" else 0.22
    draw_shapely(ax_map,poly,color,"#777777",0.9,(0,(5,3)),alpha,2)

    cx,cy  = res_center_2d[rk]
    label  = res_label_map.get(rk) or aa_label(info["resname"],info["resnr"])
    fs     = calc_fs(poly)

    # 残基名ラベル（重心・最前面）
    ax_map.text(cx, cy, label,
                fontsize=fs, fontweight="bold",
                ha="center", va="center",
                color="#111111", zorder=8,
                path_effects=[
                    pe.withStroke(linewidth=fs*0.35, foreground="white")
                ])

    # 番号バッジ（相互作用残基のみ・ラベルと重ならない位置）
    if rk in res_to_num:
        num   = res_to_num[rk]
        bpos  = get_badge_pos(poly, np.array([cx,cy]))
        draw_badge_map(ax_map, bpos[0], bpos[1],
                       num, color, MAP_BADGE_R)

ax_map.set_title(
    f"Binding Pocket Surface Map  —  {PDB_ID} / {ligand_key}\n"
    "3D→2D PCA projection  |  Area ∝ solvent-exposed surface  "
    "|  Numbers = interaction index",
    fontsize=12, color="#111111", pad=10, fontweight="bold")

# ══ リガンド2D + バッジ ══════════════════════
ax_lig.imshow(lig_arr, aspect="equal", zorder=1)
ax_lig.axis("off")
ax_lig.set_xlim(0,iw); ax_lig.set_ylim(ih,0)

for entry in interaction_index:
    num   = entry["idx"]
    itype = entry["itype"]
    color = ITYPE_META[itype]["color"]
    for ai in entry["atom_idxs"]:
        rx,ry = rdkit_atom_pos.get(ai,
            ((rxmin+rxmax)/2,(rymin+rymax)/2))
        px,py = r2p(rx,ry)
        draw_badge_lig(ax_lig, px, py, num, color, LIG_BADGE_R)

# IUPAC名（構造の下・大きめ・斜体）
wrapped_iupac = "\n".join(textwrap.wrap(iupac_name, width=38))
ax_lig.text(iw/2, ih*1.02, wrapped_iupac,
            fontsize=13,          # ← 大きめに設定
            ha="center", va="top",
            color="#111111", style="italic",
            transform=ax_lig.transData,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#f5f5f5",
                      edgecolor="#bbbbbb", lw=1.0))

ax_lig.set_xlim(-iw*0.05, iw*1.05)
ax_lig.set_ylim(ih*1.28, -ih*0.05)
ax_lig.set_title(f"Ligand: {lig_resname}  —  2D Structure",
                 fontsize=13, color="#111111",
                 pad=10, fontweight="bold")

# ══ インデックス対応表 ══════════════════════
ax_tbl.axis("off")
ax_tbl.set_xlim(0,1); ax_tbl.set_ylim(0,1)

ax_tbl.text(0.5, 0.980, "Interaction Index Table",
            fontsize=14, fontweight="bold",
            ha="center", va="top", color="#111111")
ax_tbl.axhline(0.958, color="#333333", lw=1.5,
               xmin=0.01, xmax=0.99)

HDR_Y  = 0.942
col_x  = [0.03, 0.15, 0.44, 0.74]
for x,hdr in zip(col_x,["#","Residue","Interaction Type","Color"]):
    ax_tbl.text(x, HDR_Y, hdr,
                fontsize=11, fontweight="bold",
                ha="left", va="top", color="#333333")
ax_tbl.axhline(HDR_Y-0.022, color="#aaaaaa", lw=0.8,
               xmin=0.01, xmax=0.99)

n_entries = len(interaction_index)
row_h     = min(0.880/max(n_entries,1), 0.056)
y_start   = HDR_Y - 0.030

for i,entry in enumerate(interaction_index):
    y     = y_start - i*row_h
    num   = entry["idx"]
    label = entry["res_label"]
    itype = entry["itype"]
    color = ITYPE_META[itype]["color"]
    iname = ITYPE_META[itype]["label"]
    fs_r  = float(np.clip(row_h*52, 9.0, 13.0))

    # 交互背景
    if i%2==0:
        ax_tbl.add_patch(FancyBboxPatch(
            (0.01, y-row_h*0.88), 0.97, row_h*0.94,
            boxstyle="round,pad=0.002",
            facecolor="#f7f7f7", edgecolor="none", zorder=1))

    # ── 番号バッジ（小さめ・黒字）──
    badge_r_tbl = min(row_h*0.32, 0.018)
    bx = col_x[0] + 0.04
    by = y - row_h*0.40
    ax_tbl.add_patch(Circle(
        (bx, by), radius=badge_r_tbl,
        facecolor=color, edgecolor="white",
        linewidth=0.8, alpha=BADGE_ALPHA, zorder=3))
    ax_tbl.text(bx, by, str(num),
                fontsize=fs_r, fontweight="bold",
                ha="center", va="center",
                color=BADGE_NUM_COLOR, zorder=4,
                path_effects=[
                    pe.withStroke(linewidth=1.5, foreground="white")
                ])

    # 残基名
    ax_tbl.text(col_x[1], y-row_h*0.38, label,
                fontsize=fs_r, fontweight="bold",
                ha="left", va="center", color="#111111", zorder=3)

    # 相互作用タイプ
    ax_tbl.text(col_x[2], y-row_h*0.38, iname,
                fontsize=fs_r,
                ha="left", va="center", color="#333333", zorder=3)

    # 色見本バー
    ax_tbl.add_patch(FancyBboxPatch(
        (col_x[3], y-row_h*0.68),
        0.22, row_h*0.60,
        boxstyle="round,pad=0.003",
        facecolor=color, edgecolor="#666",
        linewidth=0.6, alpha=0.85, zorder=3))

    ax_tbl.axhline(y-row_h*0.94, color="#dddddd", lw=0.5,
                   xmin=0.01, xmax=0.99)

# ── 相互作用タイプ凡例 ──
legend_y = y_start - n_entries*row_h - 0.020
ax_tbl.axhline(legend_y, color="#aaaaaa", lw=1.0,
               xmin=0.01, xmax=0.99)
ax_tbl.text(0.5, legend_y-0.010,
            "Interaction Type Reference",
            fontsize=11, fontweight="bold",
            ha="center", va="top", color="#333333")

present   = [t for t in PRIORITY if t in set(res_itype.values())]
avail_h   = max(legend_y-0.025, 0.01)
leg_row_h = min(avail_h/max(len(present),1), 0.048)

for j,itype in enumerate(present):
    meta  = ITYPE_META[itype]
    color = meta["color"]
    ly    = legend_y - 0.035 - j*leg_row_h
    fs_l  = float(np.clip(leg_row_h*48, 8.0, 11.0))

    ax_tbl.add_patch(FancyBboxPatch(
        (0.02, ly-leg_row_h*0.55), 0.10, leg_row_h*0.72,
        boxstyle="round,pad=0.003",
        facecolor=color, edgecolor="#666",
        linewidth=0.6, alpha=0.85, zorder=3))
    ax_tbl.text(0.15, ly-leg_row_h*0.18, meta["label"],
                fontsize=fs_l, fontweight="bold",
                ha="left", va="center", color="#111111")
    ax_tbl.text(0.15, ly-leg_row_h*0.60, meta["desc"],
                fontsize=fs_l*0.80,
                ha="left", va="center",
                color="#555555", linespacing=1.2)

out_png = f"pocket_surface_map_{PDB_ID}_v10.png"
plt.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
plt.show()
log(f"\n✅ 完了: {out_png} を保存しました")
