from networkx import Graph
from napari.layers.base import Layer


class _Graph(Layer):
    def __init__(self, graph: Graph, ndim: int | None = None):
        nodes = list(graph.nodes(data=True))

        if nodes and not all(
            'coordinates' in node_attrs for _, node_attrs in nodes
        ):
            raise ValueError('All nodes need to have coordinates')

        ndim = len(nodes[0][1]['coordinates']) if nodes else (ndim or 2)

        super().__init__(data=graph, ndim=ndim)
        return
