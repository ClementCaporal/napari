from networkx import Graph
from napari.layers.base import Layer


class _Graph(Layer):
    _data: Graph

    def __init__(self, graph: Graph, ndim: int | None = None):
        nodes = list(graph.nodes(data=True))

        if nodes and not all(
            'coordinates' in node_attrs for _, node_attrs in nodes
        ):
            raise ValueError('All nodes need to have coordinates')

        ndim = (
            len(nodes[0][1]['coordinates'])
            if nodes
            else (2 if ndim is None else ndim)
        )

        self._data = graph
        super().__init__(data=graph, ndim=ndim)
        return

    @property
    def data(self) -> Graph:
        return self._data

    @data.setter
    def data(self, data: Graph) -> None:
        self._data = data
