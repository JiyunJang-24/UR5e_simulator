"""Fit checks for print/wrist_cam_mount_v2_mm.stl (run after build_mount_v2.py).

1. STL is one closed (watertight) body after a save/load round trip
2. UR / quick-changer holes are through holes at the right positions
3. D405 pocket inner size, M3 holes + screw/hex-key path from behind
4. quick-changer release tab can be reached from outside, for the gripper in each of its 4 possible 90 deg yaws
5. clearance to the real RG2 + quick-changer meshes over the whole gripper stroke, for each yaw (needs MuJoCo)
Usage: python verify_mount_v2.py   (from anywhere)
"""
import os, json
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
t = trimesh.load(os.path.join(ROOT, "print", "wrist_cam_mount_v2_mm.stl"))
p = json.load(open(os.path.join(ROOT, "wrist_cam_v2_params.json")))
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name} {info}")


check("closed single body after STL round trip", t.is_watertight and len(t.split()) == 1,
      f"(volume {t.volume / 1000:.1f} cm3, bed face z min = {t.bounds[0][2]:.2f})")

def through(x, y):
    return len(t.ray.intersects_location([[x, y, -20]], [[0, 0, 1]])[0]) == 0

for name, (x, y) in {"bolt +x+y": (17.68, 17.68), "bolt +x-y": (17.68, -17.68), "bolt -x+y": (-17.68, 17.68),
                     "bolt -x-y": (-17.68, -17.68), "dowel 0deg (UR pin)": (0, -25), "dowel 90deg": (25, 0),
                     "dowel 180deg": (0, 25), "dowel 270deg": (-25, 0)}.items():
    check(f"{name} hole through", through(x, y))
def hole_d(x, y, z=3.5):
    c = np.array([x, y, z]); ds = []
    for a in np.linspace(0, np.pi, 12, endpoint=False):
        d = np.array([np.cos(a), np.sin(a), 0])
        hits = [np.min(np.abs((t.ray.intersects_location([c], [s * d])[0] - c) @ d)) for s in (1, -1)]
        ds.append(sum(hits))
    return np.mean(ds)
check("bolt hole diameter ~6.6", abs(hole_d(17.68, 17.68) - 6.6) < 0.1, f"({hole_d(17.68, 17.68):.2f} mm)")
check("dowel hole diameter ~6.2", abs(hole_d(0, -25) - 6.2) < 0.1, f"({hole_d(0, -25):.2f} mm)")
r_boss = max(np.hypot(*v[:2]) for v in t.vertices if 3.7 < v[2] < 7.1 and np.hypot(*v[:2]) < 36)  # boss above the collar
check("plate boss fits the changer recess (D62.96 at its floor, CAD) with >= 0.15 mm/side for FDM", 62.96 - 2 * r_boss >= 0.3, f"(boss D{2 * r_boss:.2f})")

fwd = np.array(p["lens_dir"]); xc = np.array(p["cam_xaxis"]); up = np.array(p["cam_yaxis"])
floor = np.array(p["cam_pos"]) * 1000 - fwd * 23.5
mid = floor + fwd * 22.5      # above the USB slot, below the 24 mm lip
w = []
for ax in (xc, up):
    hits = [np.min(np.abs((t.ray.intersects_location([mid], [s * ax])[0] - mid) @ ax)) for s in (1, -1)]
    w.append(sum(hits))
check("D405 pocket >= 42 x 42 (camera is 42 x 42 x 23)", min(w) >= 42.2, f"({w[0]:.2f} x {w[1]:.2f} mm)")
for su in (-1, 1):
    o = floor + xc * su * 10.0 - fwd * 80
    check(f"M3 screw {'+' if su > 0 else '-'} path from behind is clear", len(t.ray.intersects_location([o], [fwd])[0]) == 0)

# gripper (quick changer + RG2) may sit in any 90 deg step on the mount: yaw k*90 deg relative to the
# pin-aligned CAD orientation (k=0: release tab on flange -y, fingers along x; k=2: tab on +y, the "rotated 180" case)
TAB = dict(x=(-15.9, 15.9), r=(19.2, 39.8), z=(21.0, 28.5))      # release tab in the k=0 orientation (tab along -y)
def Rz(a):
    c, s_ = np.cos(a), np.sin(a); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
for k in range(4):
    R = Rz(np.deg2rad(90 * k)); out = R @ np.array([0, -1, 0]); side = R @ np.array([1, 0, 0])
    blocked = []
    for sx in np.linspace(-14, 14, 8):
        for z in (22, 25, 28):
            o = out * 150 + side * sx + np.array([0, 0, z])
            L = t.ray.intersects_location([o], [-out])[0]
            if len(L) and (((L - np.array([0, 0, z])) @ out) > TAB["r"][1]).any():
                blocked.append((round(sx, 1), z))
    check(f"gripper yaw {90 * k:3d} deg: release tab reachable from outside", not blocked, f"(blocked: {blocked})" if blocked else "")

try:
    import mujoco
    xml = os.path.join(ROOT, "..", "universal_robots_ur5e", "scene_rg2_tabletop.xml")
    m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m); att = m.body("attachment").id
    gf = m.body("gripper_frame").id; q0 = m.body_quat[gf].copy()
    tm = t.copy(); tm.apply_scale(0.001)
    for k in range(4):
        qk = np.zeros(4); mujoco.mju_mulQuat(qk, np.array([np.cos(np.pi / 4 * k), 0, 0, np.sin(np.pi / 4 * k)]), q0)
        m.body_quat[gf] = qk
        worst, worst_fit = 1e9, 1e9
        for q in np.linspace(0, 1.5, 4):
            d.qpos[:6] = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
            for n in ["right_arm", "left_arm", "right_inner", "left_inner"]: d.joint(f"rg2_{n}_joint").qpos = q
            for n in ["right_tip", "left_tip"]: d.joint(f"rg2_{n}_joint").qpos = -q
            mujoco.mj_forward(m, d)
            Ra, pa = d.xmat[att].reshape(3, 3), d.xpos[att]; P = []
            for g in range(m.ngeom):
                if m.geom_group[g] == 2 and m.body(m.geom_bodyid[g]).name.startswith("rg2_"):
                    mid_ = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid_], m.mesh_vertnum[mid_]
                    P.append(((m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - pa) @ Ra)
            P = np.vstack(P)
            lo, hi = tm.bounds; P = P[np.all((P > lo - 0.006) & (P < hi + 0.006), axis=1)]   # only points near the mount
            r = np.hypot(P[:, 0], P[:, 1])
            # (a) designed fit: plate boss in the changer recess (0.2 mm radial, top on the bolt bosses) and the changer
            #     rim 0.9 mm above the collar. Sim meshes are decimated (~0.4 mm), so only "not intersecting" is checked.
            fit = (r < 0.0365) & (P[:, 2] < 0.0080)
            boss_contact = (r < 0.0318) & (P[:, 2] > 0.0068) & (P[:, 2] < 0.0076)   # intended face contact on the bosses
            Pf = P[fit & ~boss_contact]
            if len(Pf): worst_fit = min(worst_fit, -trimesh.proximity.signed_distance(tm, Pf).max() * 1000)
            # (b) everything else: real clearance
            Po = P[~fit]
            if len(Po): worst = min(worst, -trimesh.proximity.signed_distance(tm, Po).max() * 1000)
        check(f"gripper yaw {90 * k:3d} deg: changer recess / rim zone, no gross overlap (sim mesh, +-0.4 mm)", worst_fit > -0.45, f"({worst_fit:.2f} mm; exact fit checked on the changer CAD)")
        check(f"gripper yaw {90 * k:3d} deg: clearance to RG2 + changer elsewhere, full stroke >= 2 mm", worst >= 2.0, f"({worst:.2f} mm)")
    m.body_quat[gf] = q0
except ImportError:
    print("[skip] MuJoCo not installed: RG2 clearance not checked")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
