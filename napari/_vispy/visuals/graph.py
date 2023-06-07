from vispy.visuals import LineVisual

from napari._vispy.visuals.points import PointsVisual


class GraphVisual(PointsVisual):
    def __init__(self):
        super().__init__()
        self.add_subvisual(LineVisual(connect='segments'))