"""Extract actual z=0 cell polygons and cell-associated fields with pvpython.

pvpython scripts/extract_bubble_animation.py --case CASE/simulation --output DIR
No cell-to-point interpolation, scalar smoothing, or mesh simplification occurs.
The boundary curve is recovered from the slice's actual boundary edges.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from paraview.simple import OpenFOAMReader, Slice
from paraview import servermanager
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkFeatureEdges

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--max-time', type=float)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
reader = OpenFOAMReader(FileName=str(args.case/'post.foam'))
reader.MeshRegions = ['internalMesh']
reader.CellArrays = ['s1', 's2', 's3', 'U']
reader.SkipZeroTime = 0
reader.UpdatePipelineInformation()
times = [t for t in reader.TimestepValues if args.max_time is None or t<=args.max_time]
plane = Slice(Input=reader)
plane.SliceType = 'Plane'
plane.SliceType.Origin = [0, 0, 0]
plane.SliceType.Normal = [0, 0, 1]
plane.Triangulatetheslice = 0
checks = []
for index, time in enumerate(times):
    plane.UpdatePipeline(time=time)
    block = servermanager.Fetch(plane)
    assert block.GetNumberOfBlocks() == 1
    mesh = block.GetBlock(0)
    points = vtk_to_numpy(mesh.GetPoints().GetData())[:, :2]
    polygons = mesh.GetPolys()
    offsets = vtk_to_numpy(polygons.GetOffsetsArray())
    connectivity = vtk_to_numpy(polygons.GetConnectivityArray())
    sizes = np.diff(offsets)
    assert sizes.max() <= 4
    # Pad triangles with a repeated final vertex; this preserves their geometry.
    ids = offsets[:-1, None] + np.minimum(np.arange(4)[None, :], sizes[:, None]-1)
    vertices = points[connectivity[ids]]
    concentration = np.column_stack([vtk_to_numpy(mesh.GetCellData().GetArray(f's{i}'))
                                     for i in (1, 2, 3)])
    velocity = vtk_to_numpy(mesh.GetCellData().GetArray('U'))
    assert np.isfinite(vertices).all() and np.isfinite(concentration).all()
    assert np.isfinite(velocity).all()
    edges = vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn(); edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff(); edges.NonManifoldEdgesOff(); edges.Update()
    boundary = edges.GetOutput()
    bubble_points = []
    for cell_i in range(boundary.GetNumberOfCells()):
        cell = boundary.GetCell(cell_i)
        pair = np.array([cell.GetPoints().GetPoint(k)[:2] for k in range(2)])
        if np.linalg.norm(pair, axis=1).max() < 1.5 and pair[:, 0].max() > 1e-8:
            bubble_points.extend(pair)
    interface = np.unique(np.round(bubble_points, 12), axis=0)
    # Snap roundoff-level signed zeros to the axis before angular sorting.
    interface[:, 0] = np.maximum(interface[:, 0], 0.)
    interface = interface[np.argsort(np.arctan2(interface[:, 0], interface[:, 1]))]
    assert len(interface) > 10 and interface[:, 0].min() >= -1e-7
    np.savez_compressed(args.output/f'frame-{index:04d}.npz',
                        vertices=vertices, concentration=concentration,
                        interface=interface, time=time)
    checks.append({'time':time, 'cells':len(vertices),
                   'concentration_min': concentration.min(axis=0).tolist(),
                   'concentration_max': concentration.max(axis=0).tolist(),
                   'maximum_speed':float(np.linalg.norm(velocity, axis=1).max()),
                   'interface_r_min':float(np.linalg.norm(interface, axis=1).min()),
                   'interface_r_max':float(np.linalg.norm(interface, axis=1).max())})
    print(index, time, checks[-1], flush=True)
(args.output/'frames.json').write_text(json.dumps(checks, indent=2)+'\n')
