"""Build the MuJoCo model of OnRobot Quick Changer for I/O (robot side) + RG2.

Input : per-part STL dumps (metres, CAD frame) produced by step_explode.py from
        Quick_Changer_for_IO_v2.STEP and RG2_v2.STEP (OnRobot CAD, SolidWorks 2019).
Output: ../assets/*.stl and ../rg2.xml

Model frame ("base_mount") = UR tool flange frame (attachment_site):
  origin on the flange face, +z out of the flange along the tool axis,
  fingers open/close along y.

RG2 finger = parallelogram linkage:
  outer arm (motor-driven, gear hub)  -- pivots on housing
  inner link                          -- pivots on housing
  fingertip                           -- pinned to both, stays parallel
In MuJoCo every joint of the linkage is tied to the right outer-arm joint with
<equality joint>, so one actuator drives the whole gripper.
qpos = 0 -> fully open, qpos increases while closing.

Usage: python build_rg2.py <qc_dump_dir> <rg2_dump_dir>
"""
import sys, os, json
import numpy as np
import trimesh
import fast_simplification

QC_DIR, RG_DIR = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSET = os.path.join(ROOT, "assets")
os.makedirs(ASSET, exist_ok=True)


def load_manifest(d):
    P = json.load(open(os.path.join(d, "manifest.json")))
    for p in P:
        p["top"] = int(p["path"].split("/")[1][4:])  # NAUO<k> of the top-level occurrence
        p["mesh"] = trimesh.load(os.path.join(d, p["file"]), process=False)
    return P


QC = load_manifest(QC_DIR)
RG = load_manifest(RG_DIR)


def bbc(top):  # bbox centre (metres, rg2 CAD frame) of a top-level RG2 occurrence
    ps = [p for p in RG if p["top"] == top]
    lo = np.min([p["bmin"] for p in ps], 0); hi = np.max([p["bmax"] for p in ps], 0)
    return (lo + hi) / 2


# ---------------------------------------------------------------- frames
# Quick changer (CAD): tool axis = -y, robot-side face at y = ymax, tool-side face at y = ymin
qc_lo = np.min([p["bmin"] for p in QC], 0); qc_hi = np.max([p["bmax"] for p in QC], 0)
QC_ROBOT_FACE = qc_hi[1]
QC_THICK = qc_hi[1] - qc_lo[1]
# RG2 (CAD): tool axis = +x. Its blue ring (NAUO3) is the face that seats on the quick changer.
ring = [p for p in RG if p["top"] == 3][0]
RG_FACE_X = ring["bmin"][0]
RG_AXIS = np.array([RG_FACE_X, (ring["bmin"][1] + ring["bmax"][1]) / 2, (ring["bmin"][2] + ring["bmax"][2]) / 2])
QC_AXIS = np.array([(qc_lo[0] + qc_hi[0]) / 2, QC_ROBOT_FACE, (qc_lo[2] + qc_hi[2]) / 2])

# rows = model axes expressed in CAD axes
R_QC = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)   # x->x, z->y, -y->z
R_RG = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)   # -z->x, y->y, x->z


def qc_to_model(v):
    return (R_QC @ (v - QC_AXIS).T).T


def rg_to_model(v):
    return (R_RG @ (v - RG_AXIS).T).T + np.array([0, 0, QC_THICK])


# ---------------------------------------------------------------- linkage geometry (rg2 CAD frame)
# lower finger in CAD (y < axis) = model -y = "right"; upper = "left"
P_INNER_R = bbc(24)          # inner-link pivot on housing
P_INNER_TIP_R = bbc(25)      # inner-link pivot on fingertip
P_ARM_TIP_R = bbc(18)        # outer-arm pivot on fingertip
P_ARM_R = P_ARM_TIP_R - (P_INNER_TIP_R - P_INNER_R)   # outer-arm hub (parallelogram)
P_INNER_L = bbc(38); P_INNER_TIP_L = bbc(39); P_ARM_TIP_L = bbc(20)
P_ARM_L = P_ARM_TIP_L - (P_INNER_TIP_L - P_INNER_L)
LINK = P_ARM_TIP_R - P_ARM_R
L = np.linalg.norm(LINK[:2])
PHI0 = np.arctan2(LINK[1], LINK[0])          # CAD link angle of the right finger (rad)

GROUPS = {
    "right_arm":   [91, 98, 92, 99, 100, 101, 18, 19],
    "right_inner": [22, 35, 23, 25, 26],
    "right_tip":   [51, 60],
    "left_arm":    [90, 94, 93, 96, 95, 97, 20, 21],
    "left_inner":  [36, 49, 37, 39, 40],
    "left_tip":    [50, 61],
}
top2body = {t: b for b, ts in GROUPS.items() for t in ts}

# pad inner surfaces (CAD y) at the CAD pose -> opening width
tipR = np.vstack([p["mesh"].vertices for p in RG if p["top"] == 51])
tipL = np.vstack([p["mesh"].vertices for p in RG if p["top"] == 50])
GAP0 = tipL[:, 1].min() - tipR[:, 1].max()


def gap(alpha):  # alpha: rotation of the right arm about +z_cad from CAD pose
    return GAP0 - 2 * L * (np.sin(PHI0 + alpha) - np.sin(PHI0))


PHI_OPEN = -np.pi / 2 + np.deg2rad(2.0)      # arms almost perpendicular to tool axis
ALPHA_OPEN = PHI_OPEN - PHI0
# closing limit: pads touch
from scipy.optimize import brentq
ALPHA_CLOSE = brentq(gap, 0.0, np.pi / 2 - PHI0 - 1e-3)
Q_CLOSE = ALPHA_CLOSE - ALPHA_OPEN
print(f"L={L*1000:.2f}mm phi0={np.rad2deg(PHI0):.2f}deg gap_cad={GAP0*1000:.1f}mm "
      f"gap_open={gap(ALPHA_OPEN)*1000:.1f}mm q_close={Q_CLOSE:.4f}rad")


def rotz_about(v, c, a):
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return (R @ (v - c).T).T + c


def to_open_pose(v, body):
    """Move CAD-pose vertices of a moving part to the fully-open pose (CAD frame)."""
    s = 1.0 if body.startswith("right") else -1.0
    a = s * ALPHA_OPEN
    side = body.split("_")[0]
    H = P_ARM_R if side == "right" else P_ARM_L
    I = P_INNER_R if side == "right" else P_INNER_L
    B = P_ARM_TIP_R if side == "right" else P_ARM_TIP_L
    if body.endswith("_arm"):
        return rotz_about(v, H, a)
    if body.endswith("_inner"):
        return rotz_about(v, I, a)
    return v + (rotz_about(B[None], H, a)[0] - B)


# body frames in the model (identity orientation), all in open pose
def pivot_model(p_cad, body):
    return rg_to_model(to_open_pose(p_cad[None], body))[0]


BODY_POS = {
    "base": np.zeros(3),
    "right_arm": rg_to_model(P_ARM_R[None])[0],
    "right_inner": rg_to_model(P_INNER_R[None])[0],
    "right_tip": pivot_model(P_ARM_TIP_R, "right_tip"),
    "left_arm": rg_to_model(P_ARM_L[None])[0],
    "left_inner": rg_to_model(P_INNER_L[None])[0],
    "left_tip": pivot_model(P_ARM_TIP_L, "left_tip"),
}
BODY_PARENT = {"right_tip": "right_arm", "left_tip": "left_arm"}


def world_pos(b):  # BODY_POS holds model-frame (world) positions; tips are made parent-relative in finger()
    return BODY_POS[b]

# ---------------------------------------------------------------- collect meshes per (body, colour)
def colour_key(c):
    c = c or [0.7, 0.7, 0.7]
    return tuple(np.round(c, 2))


buckets = {}   # (body, colour) -> list of (V_model, F)
hull_pts = {}  # body -> vertices for collision hull
for p in QC:
    V = qc_to_model(p["mesh"].vertices)
    buckets.setdefault(("base", colour_key(p["color"])), []).append((V, p["mesh"].faces))
    hull_pts.setdefault("base", []).append(V)
for p in RG:
    body = top2body.get(p["top"], "base")
    if p["top"] == 4:   # 52-pin I/O contact block hidden inside the tool changer: drop
        continue
    V = p["mesh"].vertices
    if body != "base":
        V = to_open_pose(V, body)
    V = rg_to_model(V)
    buckets.setdefault((body, colour_key(p["color"])), []).append((V, p["mesh"].faces))
    hull_pts.setdefault(body, []).append(V)

# pad collision: fingertip parts beyond the tip pivot
mesh_xml, geom_xml, mats = [], {b: [] for b in BODY_POS}, {}
for (body, col), parts in sorted(buckets.items()):
    V = np.vstack([v for v, _ in parts]); off = np.cumsum([0] + [len(v) for v, _ in parts])
    F = np.vstack([f + o for (_, f), o in zip(parts, off)])
    m = trimesh.Trimesh(V, F, process=True)
    target = 20000 if body == "base" else 6000
    if len(m.faces) > target:
        v2, f2 = fast_simplification.simplify(m.vertices, m.faces, 1 - target / len(m.faces))
        m = trimesh.Trimesh(v2, f2, process=True)
    m.vertices -= world_pos(body)
    cname = "c%02d%02d%02d" % tuple(int(x * 99) for x in col)
    mats[cname] = col
    name = f"{body}_{cname}"
    m.export(os.path.join(ASSET, name + ".stl"))
    mesh_xml.append(f'    <mesh name="{name}" file="{name}.stl"/>')
    geom_xml[body].append(f'<geom class="rg2_visual" mesh="{name}" material="{cname}"/>')

# collision hulls + inertia
MASS = {"base": 0.72, "right_arm": 0.035, "left_arm": 0.035, "right_inner": 0.015, "left_inner": 0.015,
        "right_tip": 0.03, "left_tip": 0.03}
inertial_xml = {}
for body, pts in hull_pts.items():
    V = np.vstack(pts) - world_pos(body)
    hull = trimesh.Trimesh(V).convex_hull
    hull.density = MASS[body] / hull.volume
    com = hull.center_mass; I = hull.moment_inertia
    inertial_xml[body] = (f'<inertial pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}" mass="{MASS[body]}" '
                          f'fullinertia="{I[0,0]:.3e} {I[1,1]:.3e} {I[2,2]:.3e} {I[0,1]:.3e} {I[0,2]:.3e} {I[1,2]:.3e}"/>')
    if body.endswith("_tip"):
        # finger pad = tip vertices beyond the pivots (towards the object)
        pad = V[V[:, 2] > V[:, 2].max() - 0.032]
        h = trimesh.Trimesh(pad).convex_hull
        h.export(os.path.join(ASSET, f"{body}_pad_col.stl"))
        mesh_xml.append(f'    <mesh name="{body}_pad_col" file="{body}_pad_col.stl"/>')
        geom_xml[body].append(f'<geom name="{body}_pad" class="rg2_pad" mesh="{body}_pad_col"/>')
        rest = V[V[:, 2] <= V[:, 2].max() - 0.032]
        h = trimesh.Trimesh(rest).convex_hull
        h.export(os.path.join(ASSET, f"{body}_col.stl"))
        mesh_xml.append(f'    <mesh name="{body}_col" file="{body}_col.stl"/>')
        geom_xml[body].append(f'<geom class="rg2_collision" mesh="{body}_col"/>')
        padc = (pad.min(0) + pad.max(0)) / 2
        inner_y = pad[:, 1].max() if body.startswith("right") else pad[:, 1].min()
        geom_xml[body].append(f'<site name="{body}_pad_center" pos="{padc[0]:.5f} {inner_y:.5f} {padc[2]:.5f}"/>')
    else:
        hull.export(os.path.join(ASSET, f"{body}_col.stl"))
        mesh_xml.append(f'    <mesh name="{body}_col" file="{body}_col.stl"/>')
        geom_xml[body].append(f'<geom class="rg2_collision" mesh="{body}_col"/>')

# TCP: centre between the pads when closed (z of pad centre is nearly constant over the stroke? no ->
# report closed-pose value; pads move along z by L*(cos) while closing)
padR = np.vstack(hull_pts["right_tip"]); padR = padR[padR[:, 2] > padR[:, 2].max() - 0.032]
pad_center_open = (padR.min(0) + padR.max(0)) / 2
dz_close = L * (np.cos(PHI_OPEN + Q_CLOSE) - np.cos(PHI_OPEN))  # CAD x == model z
tcp = np.array([0.0, 0.0, pad_center_open[2] + dz_close])
print("TCP (closed) from flange [m]:", np.round(tcp, 4), " pad centre z when open:", round(pad_center_open[2], 4))

def v3(a):
    return " ".join(f"{x:.5f}" for x in a)

def finger(side):
    ax = "-1 0 0" if side == "right" else "1 0 0"   # positive = closing
    tip_rel = BODY_POS[f"{side}_tip"] - BODY_POS[f"{side}_arm"]
    g = lambda b: "\n          ".join(geom_xml[b])
    return f"""
        <body name="{side}_arm" pos="{v3(BODY_POS[f'{side}_arm'])}">
          {inertial_xml[f'{side}_arm']}
          <joint name="{side}_arm_joint" class="rg2_driver" axis="{ax}"/>
          {g(f'{side}_arm')}
          <body name="{side}_tip" pos="{v3(tip_rel)}">
            {inertial_xml[f'{side}_tip']}
            <joint name="{side}_tip_joint" class="rg2_follower" axis="{ax}" range="{-Q_CLOSE:.4f} 0"/>
            {g(f'{side}_tip')}
          </body>
        </body>
        <body name="{side}_inner" pos="{v3(BODY_POS[f'{side}_inner'])}">
          {inertial_xml[f'{side}_inner']}
          <joint name="{side}_inner_joint" class="rg2_follower" axis="{ax}"/>
          {g(f'{side}_inner')}
        </body>"""


mat_xml = "\n".join(f'    <material name="{n}" rgba="{c[0]} {c[1]} {c[2]} 1" specular="0.3" shininess="0.3"/>'
                    for n, c in sorted(mats.items()))
xml = f"""<mujoco model="onrobot_rg2">
  <!-- OnRobot Quick Changer for I/O (robot side) + RG2, generated by tools/build_rg2.py from OnRobot STEP files.
       base_mount frame = UR tool flange (attachment_site): z out of the flange, fingers move along y.
       Control: actuator "fingers" target = right_arm_joint angle [rad], 0 = open, {Q_CLOSE:.4f} = closed.
       Use rg2_utils.width_to_ctrl() / ctrl_to_width() to convert to/from opening width in metres. -->
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <option cone="elliptic" impratio="10"/>

  <default>
    <default class="rg2">
      <joint damping="0.05" armature="0.005" frictionloss="0.01" range="0 {Q_CLOSE:.4f}"/>
      <site size="0.002" rgba="1 0 0 1" group="4"/>
      <default class="rg2_driver">
        <joint damping="0.5"/>
      </default>
      <default class="rg2_follower"/>
      <default class="rg2_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/>
      </default>
      <default class="rg2_collision">
        <geom type="mesh" group="3" density="0" rgba="0.5 0.5 0.9 0.4"/>
        <default class="rg2_pad">
          <geom friction="1.0 0.01 0.001" solref="0.004 1" solimp="0.95 0.99 0.001" priority="1"
                rgba="0.9 0.5 0.5 0.4"/>
        </default>
      </default>
    </default>
  </default>

  <asset>
{mat_xml}
{chr(10).join(mesh_xml)}
  </asset>

  <worldbody>
    <body name="base_mount" childclass="rg2">
      <body name="base">
        {inertial_xml['base']}
        {chr(10).join('        ' + x for x in geom_xml['base']).strip()}
        <site name="tcp" pos="{v3(tcp)}"/>{finger('right')}{finger('left')}
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="base" body2="right_tip"/>
    <exclude body1="base" body2="left_tip"/>
    <exclude body1="right_inner" body2="right_tip"/>
    <exclude body1="left_inner" body2="left_tip"/>
    <exclude body1="right_inner" body2="right_arm"/>
    <exclude body1="left_inner" body2="left_arm"/>
    <exclude body1="right_arm" body2="left_arm"/>
    <exclude body1="right_inner" body2="left_inner"/>
    <exclude body1="right_arm" body2="left_inner"/>
    <exclude body1="left_arm" body2="right_inner"/>
  </contact>

  <equality>
    <joint joint1="left_arm_joint" joint2="right_arm_joint" polycoef="0 1 0 0 0" solref="0.005 1"/>
    <joint joint1="right_inner_joint" joint2="right_arm_joint" polycoef="0 1 0 0 0" solref="0.005 1"/>
    <joint joint1="left_inner_joint" joint2="left_arm_joint" polycoef="0 1 0 0 0" solref="0.005 1"/>
    <joint joint1="right_tip_joint" joint2="right_arm_joint" polycoef="0 -1 0 0 0" solref="0.005 1"/>
    <joint joint1="left_tip_joint" joint2="left_arm_joint" polycoef="0 -1 0 0 0" solref="0.005 1"/>
  </equality>

  <actuator>
    <!-- position servo on the arm angle; forcerange chosen so pad force is ~40 N (RG2 max) mid-stroke -->
    <position name="fingers" joint="right_arm_joint" kp="20" kv="0.5" ctrlrange="0 {Q_CLOSE:.4f}"
              forcerange="-{40*L*2:.3f} {40*L*2:.3f}"/>
  </actuator>
</mujoco>
"""
open(os.path.join(ROOT, "rg2.xml"), "w").write(xml)

# constants for the python helper
json.dump(dict(L=L, PHI_OPEN=PHI_OPEN, GAP_OPEN=float(gap(ALPHA_OPEN)), Q_CLOSE=float(Q_CLOSE),
               TCP_CLOSED=tcp.tolist(), PAD_Z_OPEN=float(pad_center_open[2])),
          open(os.path.join(ROOT, "rg2_params.json"), "w"), indent=1)
print("wrote", os.path.join(ROOT, "rg2.xml"))
