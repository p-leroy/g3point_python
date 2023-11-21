# inspired by add_geometry.py from Open3D

import numpy as np

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class ShowClouds:

    def __init__(self):
        self._id = 0
        self.window = gui.Application.instance.create_window("Open3D", 1024, 768)

        # 3D widget
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)
        self.widget.scene.set_background([255, 255, 255, 1])
        # self.widget.scene.show_skybox(True)
        self.widget.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.577, -0.577, -0.577))

        self.widget.scene.scene.set_sun_light([-1, -1, -1],  # direction
                                              [1, 1, 1],  # color
                                              75000)  # intensity
        self.widget.scene.scene.enable_sun_light(True)

        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                   [10, 10, 10])
        self.fov = 60
        self.widget.setup_camera(self.fov, bbox, [0, 0, 0])  # verticalFoV, scene_bounds, center_of_rotation

        self.window.add_child(self.widget)

    def add_cloud(self, name, cloud, color, size):
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"  # defaultUnlit defaultLitTransparency normals
        if color is not None:
            material.base_color = color
        material.point_size = size
        self.widget.scene.add_geometry(name, cloud, material)
        xyz = np.asarray(cloud.points)
        xmin, ymin, zmin = np.amin(xyz, axis=0)
        xmax, ymax, zmax = np.amax(xyz, axis=0)
        bbox = o3d.geometry.AxisAlignedBoundingBox([xmin, ymin, zmin],
                                                   [xmax, ymax, zmax])
        self.widget.setup_camera(self.fov, bbox, [0, 0, 0])



def show_clouds(clouds):
    gui.Application.instance.initialize()
    showClouds = ShowClouds()
    for cloud in clouds:
        showClouds.add_cloud(*cloud)
    gui.Application.instance.run()
