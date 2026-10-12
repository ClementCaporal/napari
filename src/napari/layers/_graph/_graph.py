from networkx import Graph
from napari.layers.base import Layer


class _Graph(Layer):
    _data: Graph

    def __init__(self, graph: Graph, ndim: int | None = None):
        inferred_ndim: int | None = self._get_graph_ndim(graph)

        if (
            ndim is not None
            and inferred_ndim is not None
            and inferred_ndim != ndim
        ):
            raise ValueError(
                f'The graph nodes dimensions {inferred_ndim} do not match the provided ndim argument {ndim}'
            )

        ndim = (
            inferred_ndim
            if inferred_ndim is not None
            else (2 if ndim is None else ndim)
        )

        self._data = graph
        super().__init__(data=graph, ndim=ndim)

    @property
    def data(self) -> Graph:
        return self._data

    @data.setter
    def data(self, data: Graph) -> None:
        self._data = data

    def _get_ndim(self) -> int:
        inferred_ndim: int | None = self._get_graph_ndim(self._data)
        return inferred_ndim if inferred_ndim is not None else self._ndim

    @staticmethod
    def _get_graph_ndim(graph: Graph) -> int | None:
        nodes = list(graph.nodes(data=True))

        if nodes and not all(
            'coordinates' in node_attr for _, node_attr in nodes
        ):
            raise ValueError('All nodes need to have coordinates')

        return len(nodes[0][1]['coordinates']) if nodes else None
