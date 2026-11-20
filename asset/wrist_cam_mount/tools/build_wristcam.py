"""Convert the wrist-camera mount STL (Onshape "Part Studio 1.stl", mm) for MuJoCo.

Mount CAD frame: robot-side face of the adapter plate at z = 0, plate 10 mm thick (z 0..10),
dowel-pin hole at (0, +25) mm, camera cradle (RealSense D405 size: 42x42x23 mm) on an arm.
Output frame = UR tool flange frame (attachment_site): z out of the flange, UR dowel pin at
(0, -24.2) mm  ->  mount is rotated 180 deg about z.

Writes ../assets/*.stl and ../wrist_cam_params.json (camera pose in the flange frame).
Usage: python build_wristcam.py "Part Studio 1.stl"
"""
import sys, os, json
import numpy as np
import trimesh
import fast_simplification

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)

m = trimesh.load(sys.argv[1])
R = np.diag([-1.0, -1.0, 1.0])            # 180 deg about z: mount pin (0,25) -> flange pin (0,-25)
PLATE_T = 0.010

parts = m.split(only_watertight=False)
cradle_arm = max(parts, key=lambda p: p.extents[0] * p.extents[1] * p.extents[2] if p.bounds[0][1] < -40 else 0)

# ---- camera cradle geometry (mount CAD frame, mm)
c, n, a = cradle_arm.triangles_center, cradle_arm.face_normals, cradle_arm.area_faces
box = c[:, 1] < -60
guess = np.array([0.4, 0.3, 0.9]); guess /= np.linalg.norm(guess)
k = box & ((n @ guess) > 0.95)
N = (n[k] * a[k, None]).sum(0); N /= np.linalg.norm(N)          # lens direction (out of the cradle)
h = c @ N
floor_h = np.median(h[k][a[k] > np.percentile(a[k], 50)])
fk = k & (np.abs(h - h[k].min()) < 1.0)                          # inner floor face
floor_c = (c[fk] * a[fk, None]).sum(0) / a[fk].sum()
D405 = np.array([42.0, 42.0, 23.0])
cam_body_c = floor_c + N * D405[2] / 2
cam_lens = floor_c + N * (D405[2] + 0.5)
# image "up" = towards the tool axis (matches the real D405 image orientation in the lab)
radial = np.array([cam_lens[0], cam_lens[1], 0.0]); radial /= np.linalg.norm(radial)
y_cam = -(radial - (radial @ N) * N); y_cam /= np.linalg.norm(y_cam)
z_cam = -N                                                       # MuJoCo cameras look along -z
x_cam = np.cross(y_cam, z_cam)


def to_flange(p_mm):
    return (R @ (np.asarray(p_mm) / 1000.0).T).T


def vec(v):
    return R @ v


# ---- meshes
merged = trimesh.util.concatenate(parts)
merged.vertices = to_flange(merged.vertices)
merged.invert() if np.linalg.det(R) < 0 else None
v2, f2 = fast_simplification.simplify(merged.vertices, merged.faces, 0.6)
trimesh.Trimesh(v2, f2).export(os.path.join(ROOT, "assets", "wrist_cam_mount.stl"))
# collision: cradle hull only (the plate is sandwiched between flange and quick changer)
cr = trimesh.Trimesh(to_flange(cradle_arm.vertices[cradle_arm.vertices[:, 1] < -55]))
cr.convex_hull.export(os.path.join(ROOT, "assets", "wrist_cam_cradle_col.stl"))

params = dict(
    plate_thickness=PLATE_T,
    lens_dir=vec(N).tolist(),
    cam_body_pos=to_flange(cam_body_c).tolist(),
    cam_body_size=(D405 / 2000).tolist(),
    cam_pos=to_flange(cam_lens).tolist(),
    cam_xaxis=vec(x_cam).tolist(),
    cam_yaxis=vec(y_cam).tolist(),
    tilt_from_tool_axis_deg=float(np.degrees(np.arccos(N[2]))),
)
json.dump(params, open(os.path.join(ROOT, "wrist_cam_params.json"), "w"), indent=1)
print(json.dumps(params, indent=1))
