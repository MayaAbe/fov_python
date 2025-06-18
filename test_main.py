# spherecap.py
"""
spherecap.py
Unit sphere × Rectangular FOV (hfov × vfov)  →  視野内球面パッチを返す
依存: numpy, pyvista
"""

from __future__ import annotations
import numpy as np
import pyvista as pv
from typing import Tuple, Optional


# ---------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------
def _safe_norm(v: np.ndarray) -> float:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("零ベクトルは方向を表せません。")
    return n


def _orthonormal_basis(forward: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    forward から Right, Up の単位ベクトルを構成（任意の up を自動選択）
    """
    z = np.array([0.0, 0.0, 1.0])
    f = forward / _safe_norm(forward)

    # forward が +/-Z と平行に近い場合は Y 軸を基準にする
    if abs(np.dot(f, z)) > 0.9:
        up0 = np.array([0.0, 1.0, 0.0])
    else:
        up0 = z

    r = np.cross(f, up0)
    r /= _safe_norm(r)
    u = np.cross(r, f)
    return r, u


# ---------------------------------------------------------------------
# 円錐 (旧)
# ---------------------------------------------------------------------
def cap_mesh(view_pos: np.ndarray, theta: float, resolution: int = 200) -> pv.PolyData:
    """（旧）円錐 FOV の球冠を返す"""
    sphere = pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution)
    d = _safe_norm(view_pos)
    if d < 1.0:
        raise ValueError("視点は球体外に置いてください。")
    n = view_pos / d
    point_on_plane = -n * np.cos(theta)
    cap = sphere.clip(origin=point_on_plane, normal=n, invert=False)
    return cap


# ---------------------------------------------------------------------
# 長方形 FOV (New!!)
# ---------------------------------------------------------------------
def box_cap_mesh(
    view_pos: np.ndarray,
    hfov: float,
    vfov: float,
    look_dir: Optional[np.ndarray] = None,
    resolution: int = 400,
) -> pv.PolyData:
    """
    四角錐（左右 hfov, 上下 vfov [rad]）で切り取った球冠を返す。

    Parameters
    ----------
    view_pos : (3,) array
        視点（球の外）座標
    hfov, vfov : float
        水平／垂直の視野角 [radians]
    look_dir : (3,) array or None
        視線方向ベクトル（None→-view_pos を自動使用）
    resolution : int
        pyvista.Sphere の分割数

    Returns
    -------
    pv.PolyData
        球面パッチ（可視領域のみ）
    """
    view_pos = np.asarray(view_pos, dtype=float)
    if _safe_norm(view_pos) < 1.0:
        raise ValueError("視点は球体外に置いてください。")

    look = -view_pos if look_dir is None else np.asarray(look_dir, dtype=float)
    f = look / _safe_norm(look)                    # forward
    r, u = _orthonormal_basis(f)                   # right, up

    tan_h = np.tan(hfov / 2.0)
    tan_v = np.tan(vfov / 2.0)

    sphere = pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution)

    V = sphere.points
    dir_vecs = V - view_pos
    dir_norm = np.linalg.norm(dir_vecs, axis=1, keepdims=True)
    dir_unit = dir_vecs / dir_norm

    f_comp = dir_unit @ f
    r_comp = dir_unit @ r
    u_comp = dir_unit @ u

    inside = (
        (f_comp > 0.0) &
        (np.abs(r_comp) <= f_comp * tan_h) &
        (np.abs(u_comp) <= f_comp * tan_v)
    )

    sphere["mask"] = inside.astype(int)
    cap = sphere.threshold(value=0.5, scalars="mask", preference="point")
    return cap


def show_box_cap(
    view_pos: np.ndarray,
    hfov: float,
    vfov: float,
    look_dir: Optional[np.ndarray] = None,
    resolution: int = 400,
    show_axes: bool = True,
) -> None:
    """四角錐 FOV の球冠を可視化（プレゼン／デバッグ用）"""
    cap = box_cap_mesh(view_pos, hfov, vfov, look_dir, resolution)

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60),
                style="wireframe", color="lightgray", opacity=0.3)
    pl.add_mesh(cap, color="orange", opacity=0.9, smooth_shading=True)

    look = -view_pos if look_dir is None else look_dir
    pl.add_arrows(view_pos, look / _safe_norm(look), mag=0.5, color="blue")

    if show_axes:
        pl.show_axes()
    pl.show()


# デモ用
if __name__ == "__main__":
    vp = np.array([2.0, 0.5, 1.5])
    hfov_deg, vfov_deg = 40, 25
    show_box_cap(vp, np.deg2rad(hfov_deg), np.deg2rad(vfov_deg))
