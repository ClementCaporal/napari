"""
Add nD Graph
===================

Display a time-varying Möbius strip as a graph in napari. Each node is a point on a Möbius
strip, and edges connect neighbouring points. Over time the strip's
twist travels around the loop, giving a looping animation.

.. tags:: visualization-nD
"""

import networkx as nx
import numpy as np

import napari

n_u = 60       # nodes along the loop
n_v = 5        # nodes across the strip width
n_t = 48       # number of time frames
radius = 15    # radius of the loop
width = 20     # width of the strip

u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)  # position along the loop
v = np.linspace(-width / 2, width / 2, n_v)         # position across the strip
U, V = np.meshgrid(u, v, indexing="ij")

# Build the graph: each node must have a 'coordinates' attribute
graph = nx.Graph()
nodes_per_frame = n_u * n_v

for t in range(n_t):
    # Half a turn of the strip along the loop makes it a Möbius strip;
    # adding a time-dependent phase makes the twist travel around it.
    angle = 0.5 * U + 2 * np.pi * t / n_t

    X = (radius + V * np.cos(angle)) * np.cos(U)
    Y = (radius + V * np.cos(angle)) * np.sin(U)
    Z = V * np.sin(angle)

    # napari coordinates are (t, z, y, x)
    coords = np.column_stack([np.full(X.size, t), Z.ravel(), Y.ravel(), X.ravel()])

    # Node ids of frame t are offset so that they are unique across frames
    offset = t * nodes_per_frame
    for k, coord in enumerate(coords):
        graph.add_node(offset + k, coordinates=coord)

    # Connect each node to its neighbours
    for i in range(n_u):
        for j in range(n_v):
            node = offset + i * n_v + j

            # Across the width
            if j < n_v - 1:
                graph.add_edge(node, node + 1)

            # Along the loop
            if i < n_u - 1:
                graph.add_edge(node, node + n_v)
            else:
                # Closing the loop: the half twist flips the strip,
                # so the last row connects to the first row mirrored.
                graph.add_edge(node, offset + (n_v - 1 - j))

# add the graph to napari
viewer = napari.Viewer(ndisplay=3)
viewer._add_graph(graph, name="Möbius strip")

viewer.dims.axis_labels = ("t", "z", "y", "x")
viewer.scene.overlays.coordinate_grid.visible = True
viewer.scene.camera.angles = (-30, 0, -70)

if __name__ == "__main__":
    napari.run()
