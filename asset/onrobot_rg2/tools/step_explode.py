"""Read a STEP assembly via XCAF and export each leaf part instance as an STL in
assembly coordinates (metres), plus a JSON manifest with name, path, colour, bbox."""
import sys, os, json
import numpy as np
import trimesh
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS
from OCP.Quantity import Quantity_Color
from OCP.IFSelect import IFSelect_RetDone

step, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

doc = TDocStd_Document(TCollection_ExtendedString("doc"))
r = STEPCAFControl_Reader()
r.SetColorMode(True); r.SetNameMode(True)
assert r.ReadFile(step) == IFSelect_RetDone
r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())


def name_of(lab):
    a = TDataStd_Name()
    if lab.FindAttribute(TDataStd_Name.GetID_s(), a):
        return a.Get().ToExtString()
    return "?"


def color_of(lab, shape):
    c = Quantity_Color()
    for t in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
        if ct.GetColor_s(lab, t, c) or ct.GetColor(shape, t, c):
            return [c.Red(), c.Green(), c.Blue()]
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        for t in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
            if ct.GetColor(exp.Current(), t, c):
                return [c.Red(), c.Green(), c.Blue()]
        exp.Next()
    return None


def mesh_shape(shape, loc):
    BRepMesh_IncrementalMesh(shape, 0.05, False, 0.3, True)  # mm deflection
    V, F = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        L = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, L)
        if tri is not None:
            T = L.Transformation()
            off = len(V)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(T)
                V.append((p.X(), p.Y(), p.Z()))
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                F.append((off + a - 1, off + c - 1, off + b - 1) if rev else (off + a - 1, off + b - 1, off + c - 1))
        exp.Next()
    V = np.array(V, float)
    if len(V) == 0:
        return None
    # apply instance location
    trsf = loc.Transformation()
    M = np.eye(4)
    for i in range(3):
        for j in range(4):
            M[i, j] = trsf.Value(i + 1, j + 1)
    V = (M[:3, :3] @ V.T).T + M[:3, 3]
    return trimesh.Trimesh(V / 1000.0, np.array(F), process=True)


parts = []


def walk(lab, loc, path, inherited_color):
    if st.IsReference_s(lab):
        ref = TDF_Label(); st.GetReferredShape_s(lab, ref)
        loc = loc * st.GetLocation_s(lab)
        col = color_of(lab, st.GetShape_s(lab)) or inherited_color
        walk(ref, loc, path + [name_of(lab)], col)
        return
    if st.IsAssembly_s(lab):
        kids = TDF_LabelSequence(); st.GetComponents_s(lab, kids)
        for i in range(1, kids.Length() + 1):
            walk(kids.Value(i), loc, path, inherited_color)
        return
    shape = st.GetShape_s(lab)
    col = color_of(lab, shape) or inherited_color
    m = mesh_shape(shape, loc)
    if m is None:
        return
    idx = len(parts)
    fn = f"p{idx:03d}.stl"
    m.export(os.path.join(outdir, fn))
    parts.append(dict(idx=idx, file=fn, name=name_of(lab), path="/".join(path), color=col,
                      bmin=m.bounds[0].tolist(), bmax=m.bounds[1].tolist(),
                      volume=float(abs(m.volume)) if m.is_watertight else None, nfaces=len(m.faces)))


roots = TDF_LabelSequence(); st.GetFreeShapes(roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), TopLoc_Location(), [name_of(roots.Value(i))], None)
json.dump(parts, open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
print(len(parts), "parts")
