# inspired by add_geometry.py from Open3D

import numpy as np

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class ShowClouds:

    def __init__(self, warnings=False, projection='perspective'):

        if warnings is False:  # disable warnings
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        self._id = 0
        self.window = gui.Application.instance.create_window("Open3D", 1024, 768)

        # 3D widget
        self.widget = gui.SceneWidget()
        # self.widget.set_on_mouse(self.on_mouse)  # SHOULD NOT BE NECESSARY
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.widget.scene.set_background([255, 255, 255, 1])
        # self.widget.scene.show_skybox(True)
        self.widget.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.577, -0.577, -0.577))

        self.widget.scene.scene.set_sun_light([-1, -1, -1],  # direction
                                              [1, 1, 1],  # color
                                              1000)  # intensity
        self.widget.scene.scene.enable_sun_light(True)

        # Modify the projection [NOT IMPLEMENTED]
        if projection == 'perspective':
            # Set projection mode to perspective
            pass
        elif projection == 'orthographic':
            # To switch to orthographic mode, you can use:
            pass
        else:
            raise ValueError('projection must be either perspective or orthographic')

        self.fov = 60

        self.window.add_child(self.widget)

        self.geometries = {}

    def update_camera(self):

        bbox = o3d.geometry.AxisAlignedBoundingBox()

        for name, geometry in self.geometries.items():
            print(f'add geometry to the bounding box: {name}')
            bbox += o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(geometry)

        self.widget.setup_camera(self.fov, bbox, bbox.get_center())

    def on_mouse(self, event):

        if event.type != gui.MouseEvent.Type.DRAG:
            return gui.SceneWidget.IGNORED  # possible values: CONSUMED HANDLED IGNORED

        self.update_camera()

        return gui.SceneWidget.HANDLED

    def add_cloud(self, name, cloud, color, size):
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"  # defaultUnlit defaultLitTransparency normals
        if color is not None:
            material.base_color = color
        material.point_size = size
        self.widget.scene.add_geometry(name, cloud, material)
        self.geometries[name] = cloud


def show_clouds(clouds, warnings=False, projection='perspective'):
    gui.Application.instance.initialize()
    showClouds = ShowClouds(warnings=warnings, projection=projection)
    for cloud in clouds:
        showClouds.add_cloud(*cloud)
    showClouds.update_camera()
    gui.Application.instance.run()
