from dataclasses import dataclass
import numpy as np
from networkx import Graph

from napari.layers.base import _LayerSlicingState


@dataclass
class _GraphSlice:
    node_coordinates: np.ndarray
    edge_coordinates: np.ndarray


class _GraphSlicingState(_LayerSlicingState):
    def __init__(self, layer, data, cache):
        super().__init__(layer, data, cache)
        self._view = None

    def _set_view_slice(self) -> None:
        pass


def _slice_graph(graph: Graph, slice_input, data_slice) -> _GraphSlice:
    # Choose the visible nodes and edges, then return their coordinates.
    ...
