"""Wrist-camera mount v2: DROID/LIBERO-style view for UR5e + OnRobot Quick Changer + RG2 + RealSense D405.
Designed to be FDM-printed in one piece, flange face down on the bed, without support material.

Everything is built in the UR tool-flange frame F [mm] (= print frame, bed at z = 0):
  z = 0 on the flange face, +z along the tool, UR dowel pin at (0, -25), RG2 fingers move along x,
  quick-changer release tab on -y (x +-15.9, r <= 39.8, z 21.1..28.6) when the changer is pin-aligned.
  At the UR5e home pose flange -y points AWAY from the robot base, so the camera goes on -y (front of the
  gripper): image bottom = fingers, image top = the workspace in front of the gripper.

Flange interface (dimensions from the v1 mount, Onshape "Part Studio 1", proven on the robot):
  plate      D62.6 x 7 mm (z 0..7, 0.6 mm lead-in chamfer), goes into the quick changer's centring recess
             (D63.2 at the mouth, D62.96 at the floor); its top lands on the
             changer's bolt bosses 2.5 mm deep (from the changer CAD), so the changer rim face sits at z = 4.5
  collar     OD 77.8, z 0..3.6 (0.9 mm under the changer rim)
  outer ring r 38.6..47, z 0..8, 150 deg arc on the camera side: outside the changer (max r 36.2), so it can be
             thick; the legs grow out of it (stiff root instead of a thin hinge under the changer rim)
  holes      4 x D6.6 through on PCD 50 (ISO 9409-1-50-4-M6), dowel holes D6.2 at all four 90 deg positions
             (UR pin is at (0,-25)); the mount and the quick changer can each be fitted in any 90 deg step
  v1's 2.4 mm centring skirt around the UR flange is NOT reproduced: it hangs below the flange face, so the
  part could not be printed flange-face-down. Location = dowel pin + 4 bolts (as on the quick changer itself).
D405 cradle (measured on v1 by ray casting): pocket 42.3 x 42.6 x 24 mm, wall 1.85, floor 2,
  2 x M3 holes 20.0 apart (D3.4 clearance here, v1: D3.15), USB slot 28.3 wide x 16 high through the floor edge
  on the image-left wall (same side as v1 relative to the camera image).

Camera pose (tools/cam_search.py grid search): lens 65 mm off-axis (-y, front), 80 mm down the tool axis,
tilted 10 deg towards the axis; image up = away from the tool axis (fingers at the bottom).

Print-driven geometry:
  * two legs straddle the release tab (window ~32 mm) and join the cradle on its tool-axis wall
  * wedge under the cradle floor -> horizontal underside; two webs under it -> one flat ~33 mm bridge
  * USB-slot ceiling horizontal in the print frame -> flat 28 mm bridge
  * keep-out for the quick changer = cylinder r 43 up to z 50, then a 45 deg cone (no flat ceiling)
  * the script slices the part every 0.2 mm and lists every region not supported by the layer below

Outputs (in ../): print/wrist_cam_mount_v2_mm.stl (mm, print this), assets/wrist_cam_mount_v2.stl (m, MuJoCo),
assets/wrist_cam_mount_v2_col.stl, wrist_cam_v2_params.json.
"""
import os, json
import numpy as np
import trimesh
import manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True); os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)

# ---------------------------------------------------------------- parameters [mm, deg]
CAM_D, CAM_H, CAM_TILT = 65.0, 80.0, 10.0
SIDE = -1                         # -1: front (-y, away from the robot base at home), +1: back (+y)
PLATE_R, PLATE_T = 31.3, 7.0         # D62.6: changer recess narrows to D62.96 at its floor (CAD) -> 0.18 mm/side + FDM growth
PLATE_CHAMFER = 0.6                # lead-in chamfer on the boss top edge
COLLAR_OR, COLLAR_T = 38.9, 3.6
QC_RECESS = 2.5                   # changer recess depth to its bolt bosses (CAD) -> changer rim at PLATE_T - QC_RECESS = 4.5
RING_IR, RING_OR, RING_T, RING_ARC = 38.6, 47.0, 8.0, 150.0   # top 8 mm: >= 3.6 mm under the changer's side M8 port (z 11.6..21.6)
BOLT_R, PIN_R, PCD_R = 3.3, 3.1, 25.0
POCKET = np.array([42.3, 42.6]); WALL, FLOOR, WALL_H = 1.85, 2.0, 24.0
D405_DEPTH = 23.0
HOLE_R, HOLE_PITCH = 1.7, 20.0
SLOT_W, SLOT_H = 28.3, 16.0
LEG_X_ROOT = (19.0, 31.0)         # |x| span of each leg at the collar (release tab is |x| <= 15.9)
LEG_U_CRADLE = 14.5               # inner |u| of each leg where it meets the cradle
LEG_V_MAX = -15.0                 # legs join the floor only for v < this (USB slot is |v| < 14.15)
WEB_U = (16.3, 20.3)              # |u| of the two webs under the cradle floor (inside the slot's floor notch)
QC_KEEPOUT_R, QC_KEEPOUT_Z = 38.5, 45.0   # changer (max r 36.2) + RG2 tool ring (r 35.5) up to z ~43; release tab passes between the legs
SEG = 96

# ---------------------------------------------------------------- camera frame in F
th = np.deg2rad(CAM_TILT)
FWD = np.array([0, -np.sin(th), np.cos(th)])          # optical axis
UP = np.array([0, np.cos(th), np.sin(th)])            # image up (away from the tool axis)
XC = np.cross(UP, -FWD)                                # image right
LENS = np.array([0, CAM_D, CAM_H])
RZ = np.diag([SIDE, SIDE, 1.0])                        # side selection = rotation about the tool axis
FWD, UP, XC, LENS = RZ @ FWD, RZ @ UP, RZ @ XC, RZ @ LENS
FLOOR_C = LENS - FWD * (D405_DEPTH + 0.5)             # cradle floor (inner face) centre, D405 back sits here
Rc = np.stack([XC, UP, FWD], 1)                        # cradle local (u, v, w) -> F
pu, pv = POCKET / 2
ou, ov = pu + WALL, pv + WALL


def M(x):
    return np.asarray(x, float)[:3, :4]


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return mf.Manifold.cube(hi - lo).translate(lo)


def cyl(r, z0, z1, xy=(0, 0), r1=None):
    return mf.Manifold.cylinder(z1 - z0, r, r if r1 is None else r1, SEG).translate([xy[0], xy[1], z0])


def to_F(man):
    T = np.eye(4); T[:3, :3] = Rc; T[:3, 3] = FLOOR_C
    return man.transform(M(T))


def loc2F(pts):
    return [FLOOR_C + Rc @ np.asarray(p, float) for p in pts]


def to_trimesh(man):
    mesh = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def hull(pts):
    t = trimesh.Trimesh(np.asarray(pts, float)).convex_hull
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))


# ---------------------------------------------------------------- flange interface
base = (cyl(PLATE_R, 0, PLATE_T - PLATE_CHAMFER) + cyl(PLATE_R, PLATE_T - PLATE_CHAMFER, PLATE_T, r1=PLATE_R - PLATE_CHAMFER)
        + cyl(COLLAR_OR, 0, COLLAR_T))
cam_dir = np.arctan2(SIDE, 0.0)            # camera side (+-y)
sector = hull([[0, 0, z] for z in (-1, RING_T + 1)] +
              [[80 * np.cos(cam_dir + a), 80 * np.sin(cam_dir + a), z]
               for a in np.deg2rad(np.linspace(-RING_ARC / 2, RING_ARC / 2, 31)) for z in (-1, RING_T + 1)])
ring = (cyl(RING_OR, 0, RING_T) - cyl(RING_IR, -1, RING_T + 1)) ^ sector
base = base + ring

# ---------------------------------------------------------------- D405 cradle (local u,v,w; w = optical axis)
cradle = box([-ou, -ov, -FLOOR], [ou, ov, WALL_H]) - box([-pu, -pv, 0], [pu, pv, WALL_H + 1])
cradle_F = to_F(cradle)
# Wedge under the (10 deg tilted) floor so its underside is horizontal in the print frame: prints as ONE flat
# bridge between the webs instead of an ~80 deg staircase overhang.
floor_bot = loc2F([[u, v, -FLOOR + 0.3] for u in (-ou, ou) for v in (-ov, ov)])
Z_FLAT = min(p[2] for p in loc2F([[0, v, -FLOOR] for v in (-ov, ov)]))   # lowest point of the floor underside
wedge = hull(floor_bot + [[p[0], p[1], Z_FLAT] for p in floor_bot])
cradle_F = cradle_F + wedge
# USB slot (image-left wall), through the floor edge; its ceiling is made horizontal in the print frame
# (the v1 height of 16 mm is kept at the high end, so the slot is never lower than v1) -> flat 28 mm bridge
z_ceil = max(p[2] for p in loc2F([[0, v, SLOT_H] for v in (-SLOT_W / 2, SLOT_W / 2)]))
slot = to_F(box([-ou - 1, -SLOT_W / 2, -FLOOR - 1], [-pu + 0.5, SLOT_W / 2, WALL_H + 5])) ^ box([-200, -200, -1], [200, 200, z_ceil])
cradle_F -= slot

# ---------------------------------------------------------------- legs + webs
w_in = -FLOOR + 0.6                    # hull points sit 0.6 mm inside the floor/wall: no coplanar faces
legs_webs = None
for sx in (-1, 1):
    root = [[sx * x, SIDE * y, z] for x in LEG_X_ROOT for y in (39.0, 58.0) for z in (0.0, RING_T - 1.0)]   # sits on the bed, inside the ring
    floor_strip = loc2F([[sx * u, v, w_in] for u in (LEG_U_CRADLE, ou - 0.3) for v in (-ov + 0.3, LEG_V_MAX)])
    wall_strip = loc2F([[sx * u, -ov + 0.3, w] for u in (LEG_U_CRADLE, ou - 0.3) for w in (w_in, 20.0)])
    leg = hull(root + floor_strip + wall_strip)
    web_root = [[sx * x, SIDE * y, z] for x in (WEB_U[0] + 2.7, WEB_U[1] + 2.7) for y in (39.0, 58.0) for z in (0.0, RING_T - 1.0)]
    web_top = [[q[0], q[1], Z_FLAT + 0.6] for q in loc2F([[sx * u, v, -FLOOR] for u in WEB_U for v in (-ov + 0.3, ov - 0.3)])]
    web = hull(web_root + web_top)
    piece = leg + web
    legs_webs = piece if legs_webs is None else legs_webs + piece
# quick changer keep-out: cylinder, then a 45 deg cone so the cut never leaves a flat ceiling
keepout = cyl(QC_KEEPOUT_R, COLLAR_T + 0.3, QC_KEEPOUT_Z) + cyl(QC_KEEPOUT_R, QC_KEEPOUT_Z, QC_KEEPOUT_Z + QC_KEEPOUT_R, r1=0.01)
legs_webs -= keepout

part = base + legs_webs + cradle_F

# ---------------------------------------------------------------- holes
for sx in (-1, 1):
    for sy in (-1, 1):
        part -= cyl(BOLT_R, -10, 20, (sx * PCD_R / np.sqrt(2), sy * PCD_R / np.sqrt(2)))
for px, py in ((0, -PCD_R), (PCD_R, 0), (0, PCD_R), (-PCD_R, 0)):   # dowel hole at all 4 positions:
    part -= cyl(PIN_R, -10, 20, (px, py))                              # mount / changer can go on in any 90 deg step
for su in (-1, 1):  # D405 M3 holes through the floor + screw-head / hex-key clearance behind it
    part -= to_F(mf.Manifold.cylinder(FLOOR + 30, HOLE_R, HOLE_R, 32).translate([su * HOLE_PITCH / 2, 0, -FLOOR - 29]))
    part -= to_F(mf.Manifold.cylinder(60, 3.2, 3.2, 32).translate([su * HOLE_PITCH / 2, 0, -FLOOR - 60]))

part = part.simplify(0.0005)
mesh_mm = to_trimesh(part)


# ---------------------------------------------------------------- printability check (flange face on the bed)
def printability(mesh, layer=0.2):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    prev, islands, z = None, [], layer / 2   # islands: unsupported area per layer (bridges show up here too)
    while z < mesh.bounds[1][2]:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        polys = []
        if sec is not None:
            p2, _ = sec.to_2D(to_2D=np.eye(4))
            polys = [p for p in p2.polygons_full if p.area > 0.05]
        cur = unary_union(polys) if polys else None
        if prev is not None and cur is not None:
            free = cur.difference(prev.buffer(0.3))
            for p in (free.geoms if hasattr(free, "geoms") else [free]):
                if p.area > 0.5:
                    islands.append((round(z, 2), round(p.area, 1), np.round(p.centroid.coords[0], 1).tolist()))
        if cur is not None:
            prev = cur
        z += layer
    n = mesh.face_normals; c = mesh.triangles_center; a = mesh.area_faces
    down = (n[:, 2] < -np.cos(np.deg2rad(45))) & (c[:, 2] > 0.3)   # overhang steeper than 45 deg
    return islands, float(a[down].sum()), c[down], a[down]


islands, over_area, over_c, over_a = printability(mesh_mm)
print("watertight", mesh_mm.is_watertight, "bodies", len(mesh_mm.split()), "volume cm3", round(mesh_mm.volume / 1000, 1),
      "bounds mm", np.round(mesh_mm.bounds, 1).tolist())
print(f"print check: {len(islands)} layer regions not supported by the layer below (> 0.5 mm2):")
for it in islands:
    print("   z=%.1f area=%.1f mm2 at %s" % it)
print(f"downward faces steeper than 45 deg: {over_area:.0f} mm2")

mesh_mm.export(os.path.join(ROOT, "print", "wrist_cam_mount_v2_mm.stl"))
mesh_m = mesh_mm.copy(); mesh_m.apply_scale(0.001)
mesh_m.export(os.path.join(ROOT, "assets", "wrist_cam_mount_v2.stl"))
col = to_trimesh(legs_webs + cradle_F).convex_hull; col.apply_scale(0.001)
col.export(os.path.join(ROOT, "assets", "wrist_cam_mount_v2_col.stl"))
np.save(os.path.join(HERE, ".overhangs.npy"), np.c_[over_c, over_a])

params = dict(
    version=2, flange_to_qc_rim=(PLATE_T - QC_RECESS) / 1000,
    cam_pos=(LENS / 1000).tolist(), cam_xaxis=XC.tolist(), cam_yaxis=UP.tolist(), lens_dir=FWD.tolist(),
    cam_body_pos=((FLOOR_C + FWD * D405_DEPTH / 2) / 1000).tolist(), cam_body_size=[0.021, 0.021, 0.0115],
    cam_d_mm=CAM_D, cam_h_mm=CAM_H, tilt_deg=CAM_TILT,
)
json.dump(params, open(os.path.join(ROOT, "wrist_cam_v2_params.json"), "w"), indent=1)
print(json.dumps(params))
