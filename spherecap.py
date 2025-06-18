"""
spherecap.py  —  WGS-84 ellipsoid + Earth texture + FOV cap + axes

依存 : numpy, pyvista      pip install numpy pyvista
"""

from __future__ import annotations
import numpy as np
import pyvista as pv
from typing import Tuple, Optional
from pyvista import examples   # texture provider

# ---------------------------------------------------------------------
# WGS-84 回転楕円体パラメータ [km]
# ---------------------------------------------------------------------
A_EQUATOR = 6378.137          # semi-major axis a
B_POLAR   = 6356.752          # semi-minor axis b
_SCALE_TO_UNIT   = np.array([1.0 / A_EQUATOR, 1.0 / A_EQUATOR, 1.0 / B_POLAR])
_SCALE_FROM_UNIT = np.array([A_EQUATOR, A_EQUATOR, B_POLAR])

EARTH_TEXTURE = examples.load_globe_texture()   # 北が上の Blue-Marble

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
    z = np.array([0.0, 0.0, 1.0])
    f = forward / _safe_norm(forward)
    up0 = np.array([0.0, 1.0, 0.0]) if abs(np.dot(f, z)) > 0.9 else z
    r = np.cross(f, up0);  r /= _safe_norm(r)
    u = np.cross(r, f)
    return r, u

# ---------------------------------------------------------------------
# 四角錐視野で手前側パッチを取得（近似：単位球→楕円体拡大）
# ---------------------------------------------------------------------
def box_cap_mesh(view_pos: np.ndarray,
                 hfov: float,
                 vfov: float,
                 look_dir: Optional[np.ndarray] = None,
                 resolution: int = 400) -> pv.PolyData:
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
    front_side = (V @ f) < 0.0            # 手前半球
    sphere["mask"] = (inside_fov & front_side).astype(int)
    cap = sphere.threshold(value=0.5, scalars="mask", preference="point")
    cap.points *= _SCALE_FROM_UNIT        # km へ拡大
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
# 座標軸を追加
# ---------------------------------------------------------------------
def _add_axes(pl: pv.Plotter,
              length: float = 8000.0) -> None:
    origin = np.array([0.0, 0.0, 0.0])
    axes = {
        "X": (np.array([ length, 0.0, 0.0]), "red"),
        "Y": (np.array([ 0.0,  length, 0.0]), "green"),
        "Z": (np.array([ 0.0, 0.0,  length]), "blue")
    }
    for label, (end_pt, color) in axes.items():
        pl.add_mesh(pv.Line(origin, end_pt), color=color,
                    line_width=4.0, opacity=1.0)
        pl.add_point_labels(end_pt.reshape(1,3),
                            [label], font_size=24,
                            text_color="black", point_size=0)

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

    # 楕円体テクスチャ球
    globe = pv.Sphere(radius=1.0, theta_resolution=360, phi_resolution=180)
    globe.texture_map_to_sphere(inplace=True)
    globe.points *= _SCALE_FROM_UNIT

    pl = pv.Plotter()
    pl.set_background("white")
    pl.add_mesh(globe, texture=EARTH_TEXTURE, smooth_shading=True)
    pl.add_mesh(frust, color="cyan", opacity=0.25, show_edges=True,
                edge_color="navy", line_width=2.0)
    pl.add_mesh(cap, color="orange", opacity=0.9, smooth_shading=True)

    # 視線矢印
    look_vec = (-_to_unit(view_pos) if look_dir is None
                else _to_unit(np.asarray(look_dir, dtype=float)))
    pl.add_arrows(view_pos, look_vec/_safe_norm(look_vec),
                  mag=500.0, color="blue")

    # 座標軸 (X,Y,Z)
    _add_axes(pl)

    if show_axes:
        pl.show_axes()
    pl.show()

# ---------------------------------------------------------------------
# デモ
# ---------------------------------------------------------------------
if __name__ == "__main__":
    view = np.array([0.0, 0.0, B_POLAR + 4000.0])   # 高度 400 km
    HFOV, VFOV = np.deg2rad(50.0), np.deg2rad(35.0)
    show_box_cap(view, HFOV, VFOV)
