import numpy as np
import pytest
from networkx import Graph as nxGraph

from napari._tests.utils import (
    assert_layer_state_equal,
    check_layer_world_data_extent,
)
from napari.components.dims import Dims
from napari.layers._graph._graph import _Graph
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_docstring_parent_class_consistency,
    validate_kwargs_sorted,
)


def test_none_graph():
    layer = _Graph(data=None, ndim=None)
    assert layer.data.number_of_nodes() == 0
    assert layer.ndim == 2


def test_empty_graph():
    graph = nxGraph()
    layer = _Graph(data=graph, ndim=None)
    assert layer.data.number_of_nodes() == 0
    assert layer.ndim == 2


def test_ndim_on_empty_and_none_graph():
    empty_graph = nxGraph()
    layer_empty = _Graph(data=empty_graph, ndim=3)
    assert layer_empty.ndim == 3

    none_layer = _Graph(data=None, ndim=4)
    assert none_layer.ndim == 4


def test_type_error_on_invalid_graph():

    with pytest.raises(TypeError):
        _Graph(data='not_a_graph')


def test_inconsistent_node_coordinates():
    graph = nxGraph()
    graph.add_node(1, coordinates=[0, 0])
    graph.add_node(2, coordinates=[0, 0, 0])
    with pytest.raises(
        ValueError, match='All node coordinates need to have the same ndim'
    ):
        _Graph(data=graph)


def test_node_missing_coordinates():
    graph = nxGraph()
    graph.add_node(1)  # no coordinates attribute
    with pytest.raises(ValueError, match='All nodes need to have coordinates'):
        _Graph(data=graph)


def test_4D_graph():
    graph = nxGraph()
    graph.add_node(1, coordinates=[1, 2, 3, 4])
    graph.add_node(2, coordinates=[5, 6, 7, 8])
    graph.add_edge(1, 2)

    layer = _Graph(data=graph)
    assert layer.ndim == 4


def test_visibility():
    """Test setting layer visibility."""
    graph = nxGraph()

    layer = _Graph(data=graph)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    graph = nxGraph()

    layer = _Graph(data=graph)
    assert layer.opacity == 1.0

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer.opacity = 0.75
    assert layer.opacity == 0.75


def test_blending():
    """Test setting layer blending mode."""
    graph = nxGraph()

    layer = _Graph(data=graph)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_thick_slice():
    """Test setting layer thick_slice mode."""
    graph = nxGraph()
    graph.add_node(1, coordinates=[0, 0, 0])
    graph.add_node(2, coordinates=[10, 1, 0])
    graph.add_edge(1, 2)

    layer = _Graph(data=graph)
    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0)))

    # Test thick slice with a single slice, only the first node should be visible
    np.testing.assert_array_equal(
        layer._slicing_state._view.node_coordinates, np.array([[0, 0]])
    )

    # if margin is thick enough, both nodes should be visible
    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0), margin_right=(10, 0, 0)))
    np.testing.assert_array_equal(
        layer._slicing_state._view.node_coordinates, np.array([[0, 0], [1, 0]])
    )

    # when the slice does not intersect the edge nor the nodes, it should not be visible
    layer._slice_dims(
        Dims(
            ndim=3,
            point=(100, 100, 100),
            range=((100, 200, 1), (100, 200, 1), (100, 200, 1)),
        )
    )
    assert layer._slicing_state._view.node_coordinates.size == 0
    assert layer._slicing_state._view.edge_coordinates.size == 0

    # when the slice intersects the edge, it should be clipped to the slice
    layer._slice_dims(
        Dims(ndim=3, point=(5, 0, 0), range=((0, 10, 1), (0, 2, 1), (0, 2, 1)))
    )
    np.testing.assert_array_equal(
        layer._slicing_state._view.edge_coordinates,
        np.array([[[0.45, 0.0], [0.55, 0.0]]]),
    )


def test_world_data_extent():
    """Test that the world data extent is computed correctly."""
    graph = nxGraph()
    graph.add_node(1, coordinates=[7, -5, 0])
    graph.add_node(2, coordinates=[-2, 0, 15])
    graph.add_node(3, coordinates=[4, 30, 12])
    layer = _Graph(data=graph)

    min_val = (-2, -5, 0)
    max_val = (7, 30, 15)

    extent = np.array((min_val, max_val))
    check_layer_world_data_extent(layer, extent, (3, 1, 1), (10, 20, 5))


def test_extent_data_empty_returns_nan():
    layer = _Graph(data=nxGraph())
    assert layer.extent.data.shape == (2, 2)
    np.testing.assert_array_equal(layer.extent.data, np.full((2, 2), np.nan))


def test_extent_single_node_returns_coordinates():
    """Test that the extent of a single node is its coordinates."""
    graph = nxGraph()
    graph.add_node(1, coordinates=[3, 4])
    layer = _Graph(data=graph)
    extent = layer.extent
    assert extent.data.shape == (2, 2)
    np.testing.assert_array_equal(extent.data, np.array([[3, 4], [3, 4]]))


def test_get_state_round_trip():  # currently fails
    graph = nxGraph()
    graph.add_node(1, coordinates=[1, 2])
    graph.add_node(2, coordinates=[3, 4])
    graph.add_edge(1, 2)
    layer = _Graph(data=graph)
    state = layer._get_state()
    new_layer = _Graph(**state)
    assert_layer_state_equal(new_layer._get_state(), state)


def test_thick_slice_excludes_all():
    graph = nxGraph()
    graph.add_node(1, coordinates=[0, 0, 0])
    layer = _Graph(data=graph)
    layer._slice_dims(
        Dims(
            ndim=3,
            point=(100, 0, 0),
            range=((100, 200, 1), (0, 1, 1), (0, 1, 1)),
        )
    )
    assert layer._slicing_state._view.node_coordinates.size == 0


def test_data_setter():
    graph = nxGraph()
    graph.add_node(1, coordinates=[0, 0])
    layer = _Graph(data=graph)
    assert layer.ndim == 2

    graph_2d = nxGraph()
    graph_2d.add_node(1, coordinates=[1, 2])
    layer.data = graph_2d
    assert layer.data is graph_2d


def test_docstring():  # currently fails
    validate_all_params_in_docstring(_Graph)
    validate_kwargs_sorted(_Graph)
    validate_docstring_parent_class_consistency(_Graph)
