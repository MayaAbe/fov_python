"""
spherecap.py  —  WGS-84 ellipsoid + LOD Earth texture (offline) + FOV visuals

依存: numpy, pillow, pyvista
    pip install numpy pillow pyvista
"""

from __future__ import annotations
import os, pa"""
spherecap.py  —  WGS-84 ellipsoid, correct N-up Earth texture,
                 FOV cap, frustum, and XYZ axes.

依存 : numpy, pyvista
        pip install numpy pyvista
"""

from __future__ import annotations
import numpy as np
import pyvista as pv
from typing import Tuple, Optional
from pyvista import examples                         # texture provider

# ---------------------------------------------------------------------
# WGS-84 回転楕円体パラメータ [km]
# ---------------------------------------------------------------------
A_EQUATOR = 6378.137            # semi-major axis a
B_POLAR   = 6356.752            # semi-minor axis b
_SCALE_TO_UNIT   = np.array([1.0 / A_EQUATOR, 1.0 / A_EQUATOR, 1.0 / B_POLAR])
_SCALE_FROM_UNIT = np.array([A_EQUATOR, A_EQUATOR, B_POLAR])

EARTH_TEXTURE = examples.load_globe_texture()        # 北が上の Blue-Marble

# ---------------------------------------------------------------------
# 変換ユーティリティ
# ---------------------------------------------------------------------
def _to_unit(v: np.ndarray) -> np.ndarray:
    return v * _SCALE_TO_UNIT

def _from_unit(v: np.ndarray) -> np.ndarray:
    return v * _SCALE_FROM_UNIT

def _safe_norm(v: np.ndarray) -> float:
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("零ベクトルは方向を表せません。")
    return n

def _orthonormal_basis(forward: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    forward から Right, Up の直交単位ベクトル (右手系) を得る
    """
    z = np.array([0.0, 0.0, 1.0])
    f = forward / _safe_norm(forward)
    up0 = np.array([0.0, 1.0, 0.0]) if abs(np.dot(f, z)) > 0.9 else z
    r = np.cross(f, up0);  r /= _safe_norm(r)
    u = np.cross(r, f)
    return r, u

# ---------------------------------------------------------------------
# 四角錐視野で“手前側”楕円体パッチを取得（近似：単位球でクリップ）
# ---------------------------------------------------------------------
def box_cap_mesh(view_pos: np.ndarray,
                 hfov: float,
                 vfov: float,
                 look_dir: Optional[np.ndarray] = None,
                 resolution: int = 400) -> pv.PolyData:
    # 視点を単位球空間へ
    view_u = _to_unit(np.asarray(view_pos, dtype=float))
    if _safe_norm(view_u) <= 1.0:
        raise ValueError("視点は地表外に置いてください。")

    look = -view_u if look_dir is None else _to_unit(np.asarray(look_dir, dtype=float))
    f = look / _safe_norm(look)
    r, u = _orthonormal_basis(f)
    tan_h, tan_v = np.tan(hfov / 2.0), np.tan(vfov / 2.0)

    sphere = pv.Sphere(radius=1.0,
                       theta_resolution=resolution,
                       phi_resolution=resolution)
    V = sphere.points
    dir_vecs = V - view_u
    dir_unit = dir_vecs / np.linalg.norm(dir_vecs, axis=1, keepdims=True)

    f_comp = dir_unit @ f
    inside_fov = (
        (f_comp > 0.0) &
        (np.abs(dir_unit @ r) <= f_comp * tan_h) &
        (np.abs(dir_unit @ u) <= f_comp * tan_v)
    )
    front_side = (V @ f) < 0.0           # V·f < 0 → 視点に近い半球
    sphere["mask"] = (inside_fov & front_side).astype(int)
    cap = sphere.threshold(value=0.5, scalars="mask", preference="point")
    cap.points *= _SCALE_FROM_UNIT       # km へ拡大
    return cap

# ---------------------------------------------------------------------
# 四角錐フレーム（km）生成
# ---------------------------------------------------------------------
def _make_frustum_mesh(view_pos: np.ndarray,
                       hfov: float,
                       vfov: float,
                       look_dir: Optional[np.ndarray] = None) -> pv.PolyData:
    view_u = _to_unit(np.asarray(view_pos, dtype=float))
    look = -view_u if look_dir is None else _to_unit(np.asarray(look_dir, dtype=float))
    f = look / _safe_norm(look)
    r, u = _orthonormal_basis(f)
    tan_h, tan_v = np.tan(hfov / 2.0), np.tan(vfov / 2.0)

    corners_u = []
    for sh, sv in [(1,1),(1,-1),(-1,-1),(-1,1)]:
        d = f + sh*tan_h*r + sv*tan_v*u
        d /= _safe_norm(d)
        b = view_u @ d
        t_near = -b - np.sqrt(max(b*b - (view_u @ view_u - 1.0), 0.0))
        corners_u.append(view_u + t_near * d)

    pts_km = _from_unit(np.vstack([view_u, np.array(corners_u)]))
    faces = []
    for i in range(1,5):
        j = i+1 if i<4 else 1
        faces.extend([3,0,i,j])
    faces.extend([4,1,2,3,4])
    return pv.PolyData(pts_km, np.array(faces))

# ---------------------------------------------------------------------
# XYZ 座標軸を追加（X=赤, Y=緑, Z=青）
# ---------------------------------------------------------------------
def _add_axes(pl: pv.Plotter,
              length: float = 8000.0) -> None:
    origin = np.zeros(3)
    axes = {
        "X": (np.array([ length, 0.0,   0.0]), "red"),
        "Y": (np.array([  0.0,  length, 0.0]), "green"),
        "Z": (np.array([  0.0,   0.0,  length]), "blue")
    }
    for lbl, (end_pt, color) in axes.items():
        pl.add_mesh(pv.Line(origin, end_pt),
                    color=color, line_width=4)
        pl.add_point_labels(end_pt.reshape(1,3), [lbl],
                            font_size=24, text_color="black",
                            point_size=0)

# ---------------------------------------------------------------------
# 可視化メイン
# ---------------------------------------------------------------------
def show_box_cap(view_pos: np.ndarray,
                 hfov: float,
                 vfov: float,
                 look_dir: Optional[np.ndarray] = None,
                 resolution: int = 400,
                 show_axes: bool = True) -> None:
    cap   = box_cap_mesh(view_pos, hfov, vfov, look_dir, resolution)
    frust = _make_frustum_mesh(view_pos, hfov, vfov, look_dir)

    # 楕円体メッシュ + テクスチャ
    globe = pv.Sphere(radius=1.0, theta_resolution=360, phi_resolution=180)
    globe.texture_map_to_sphere(inplace=True)
    globe.points *= _SCALE_FROM_UNIT       # 楕円体へ拡大
    globe.rotate_x(180.0, inplace=True)    # ← 南北反転で +Z が北極になる

    pl = pv.Plotter()
    pl.set_background("white")
    pl.add_mesh(globe, texture=EARTH_TEXTURE, smooth_shading=True)
    pl.add_mesh(frust, color="cyan", opacity=0.25,
                show_edges=True, edge_color="navy",
                line_width=2.0)
    pl.add_mesh(cap, color="orange", opacity=0.9, smooth_shading=True)

    # 視線矢印
    look_vec = (-_to_unit(view_pos) if look_dir is None
                else _to_unit(np.asarray(look_dir, dtype=float)))
    pl.add_arrows(view_pos, look_vec/_safe_norm(look_vec),
                  mag=500.0, color="blue")

    _add_axes(pl)          # XYZ ラベル付き軸
    if show_axes:
        pl.show_axes()
    pl.show()

# ---------------------------------------------------------------------
# デモ
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 高度 400 km、北緯 0°, 経度 0° 直上を例
    view = np.array([A_EQUATOR + 400.0, 0.0, 0.0])  # km (グリニッジ上空)
    HFOV, VFOV = np.deg2rad(50.0), np.deg2rad(35.0)
    show_box_cap(view, HFOV, VFOV)
thlib, urllib.request
import numpy as np
import pyvista as pv
from typing import Tuple, Optional
from PIL import Image

# ---------------------------------------------------------------------
# データディレクトリ & テクスチャ準備（初回のみ DL）
# ---------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DL_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74192/world.topo.bathy.200412.3x5400x2700.png"
DL_FILE = DATA_DIR / "blue_marble_src.png"   # 8640×4320

def _ensure_textures() -> None:
    lods = {2048, 4096, 8192}
    if all((DATA_DIR / f"earth_{n}.png").exists() for n in lods):
        return  # 既に存在
    print("▶ Earth textures not found. Downloading & generating LODs ...")
    # 1) ダウンロード（約 8 MB）
    if not DL_FILE.exists():
        print(f"  ↳ downloading {DL_URL}")
        urllib.request.urlretrieve(DL_URL, DL_FILE)
    img = Image.open(DL_FILE).convert("RGB")        # 8640×4320

    # 2) 回転 (南北反転) & サイズ調整 & 保存
    img = img.transpose(Image.FLIP_TOP_BOTTOM)      # 北を上 (+Z)
    for w in sorted(lods):
        img_resized = img.resize((w, w//2), Image.LANCZOS)
        img_resized.save(DATA_DIR / f"earth_{w}.png")
        print(f"  ↳ save LOD {w}×{w//2}")
    print("◎ Earth textures ready.")

_ensure_textures()

# ---------------------------------------------------------------------
# LOD テクスチャ管理
# ---------------------------------------------------------------------
class LODTexture:
    """カメラ距離に応じテクスチャ解像度を切替える簡易クラス"""
    LEVELS = [
        ( 25000.0, DATA_DIR / "earth_2048.png"),   # d > 25 000 km
        ( 12000.0, DATA_DIR / "earth_4096.png"),   # 12k < d ≤ 25k
        (-np.inf,  DATA_DIR / "earth_8192.png"),   # d ≤ 12k
    ]
    _cache: dict[pathlib.Path, pv.Texture] = {}

    @classmethod
    def select(cls, distance_km: float) -> pv.Texture:
        for thresh, path in cls.LEVELS:
            if distance_km > thresh:
                return cls._load(path)
        return cls._load(cls.LEVELS[-1][1])

    @classmethod
    def _load(cls, path: pathlib.Path) -> pv.Texture:
        if path not in cls._cache:
            cls._cache[path] = pv.read_texture(str(path))
        return cls._cache[path]

# ---------------------------------------------------------------------
# WGS-84 パラメータ [km]
# ---------------------------------------------------------------------
A_EQUATOR = 6378.137
B_POLAR   = 6356.752
_SCALE_TO_UNIT   = np.array([1/A_EQUATOR, 1/A_EQUATOR, 1/B_POLAR])
_SCALE_FROM_UNIT = np.array([A_EQUATOR,   A_EQUATOR,   B_POLAR])

# ---------------------------------------------------------------------
# 基本ユーティリティ
# ---------------------------------------------------------------------
def _to_unit(v: np.ndarray) -> np.ndarray: return v * _SCALE_TO_UNIT
def _from_unit(v: np.ndarray) -> np.ndarray: return v * _SCALE_FROM_UNIT
def _safe_norm(v): n = np.linalg.norm(v);  return n if n else 1.0

def _orthonormal_basis(fwd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = np.array([0,0,1.0]); f = fwd / _safe_norm(fwd)
    up0 = np.array([0,1,0]) if abs(np.dot(f,z))>0.9 else z
    r = np.cross(f, up0); r/= _safe_norm(r); u = np.cross(r,f); return r,u

# ---------------------------------------------------------------------
# 視野内パッチ (単位球→楕円体拡大)
# ---------------------------------------------------------------------
def box_cap_mesh(view: np.ndarray, hfov: float, vfov: float,
                 look: Optional[np.ndarray]=None,
                 res: int=400) -> pv.PolyData:
    view_u=_to_unit(view.astype(float))
    look = -view_u if look is None else _to_unit(look.astype(float))
    f = look/_safe_norm(look); r,u=_orthonormal_basis(f)
    tanh,tanv=np.tan(hfov/2),np.tan(vfov/2)

    sph = pv.Sphere(radius=1, theta_resolution=res, phi_resolution=res)
    V=sph.points; diru=(V-view_u)/np.linalg.norm(V-view_u,axis=1,keepdims=True)
    inside=((diru@f>0)&(np.abs(diru@r)<=diru@f*tanh)&(np.abs(diru@u)<=diru@f*tanv))
    front=(V@f)<0
    sph["mask"]=(inside&front).astype(int)
    cap=sph.threshold(value=0.5, scalars="mask", preference="point")
    cap.points*=_SCALE_FROM_UNIT
    return cap

# ---------------------------------------------------------------------
# 四角錐フレーム
# ---------------------------------------------------------------------
def _make_frustum_mesh(view: np.ndarray, hfov: float, vfov: float,
                       look: Optional[np.ndarray]=None)->pv.PolyData:
    vu=_to_unit(view.astype(float))
    look = -vu if look is None else _to_unit(look.astype(float))
    f=look/_safe_norm(look); r,u=_orthonormal_basis(f)
    th,tv=np.tan(hfov/2),np.tan(vfov/2)

    corners=[]
    for sh,sv in [(1,1),(1,-1),(-1,-1),(-1,1)]:
        d=f+sh*th*r+sv*tv*u; d/= _safe_norm(d)
        b=vu@d; t=-b-np.sqrt(max(b*b-(vu@vu-1),0)); corners.append(vu+t*d)
    pts=_from_unit(np.vstack([vu,*corners]))
    faces=[]; [faces.extend([3,0,i,i%4+1]) for i in range(1,5)]
    faces.extend([4,1,2,3,4])
    return pv.PolyData(pts, np.array(faces))

# ---------------------------------------------------------------------
# XYZ 軸追加
# ---------------------------------------------------------------------
def _add_axes(pl: pv.Plotter, length=8_000.0):
    O=np.zeros(3)
    axes={"X":([ length,0,0],"red"),
          "Y":([0, length,0],"green"),
          "Z":([0,0, length],"blue")}
    for lbl,(end,color) in axes.items():
        pl.add_mesh(pv.Line(O,end), color=color, line_width=4)
        pl.add_point_labels(np.array(end).reshape(1,3),[lbl],text_color="black",
                            font_size=24, point_size=0)

# ---------------------------------------------------------------------
# メイン表示
# ---------------------------------------------------------------------
def show_box_cap(view: np.ndarray, hfov: float, vfov: float,
                 look: Optional[np.ndarray]=None,
                 res:int=400, show_axes=True):
    cap=box_cap_mesh(view,hfov,vfov,look,res)
    fr=_make_frustum_mesh(view,hfov,vfov,look)
    dist=np.linalg.norm(view)

    # テクスチャ選択
    tex=LODTexture.select(dist)

    globe=pv.Sphere(radius=1, theta_resolution=360, phi_resolution=180)
    globe.texture_map_to_sphere(inplace=True)
    globe.points*=_SCALE_FROM_UNIT
    globe.rotate_x(180,inplace=True)   # Z+ を北極へ

    pl=pv.Plotter()
    pl.set_background("white")
    pl.add_mesh(globe, texture=tex, smooth_shading=True)
    pl.add_mesh(fr, color="cyan", opacity=0.25, show_edges=True,
                edge_color="navy", line_width=2)
    pl.add_mesh(cap, color="orange", opacity=0.9, smooth_shading=True)

    look_vec=(-_to_unit(view) if look is None else _to_unit(look.astype(float)))
    pl.add_arrows(view, look_vec/_safe_norm(look_vec), mag=500, color="blue")

    _add_axes(pl)
    if show_axes: pl.show_axes()
    pl.show()

# ---------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------
if __name__=="__main__":
    # ISS 程度の高度
    view=np.array([A_EQUATOR+400, 0, 0])  # 経度0°, 緯度0°, 高度400 km
    show_box_cap(view, np.deg2rad(50), np.deg2rad(35))
