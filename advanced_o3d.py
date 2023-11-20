# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import platform
import random
import threading
import time

class ShowClouds:

    def __init__(self):
        self._id = 0
        self.window = gui.Application.instance.create_window(
            "Add Spheres Example", 1024, 768)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([1, 1, 1, 1])
        self.widget.scene.scene.set_sun_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.widget.scene.scene.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                   [10, 10, 10])
        self.widget.setup_camera(60, bbox, [0, 0, 0])

        self.window.add_child(self.widget)

    def add_cloud(self, name, cloud, color, size):
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        if color is not None:
            material.base_color = color
        material.point_size = size
        self.widget.scene.add_geometry(name, cloud, material)



def show_clouds(clouds):
    gui.Application.instance.initialize()
    showClouds = ShowClouds()
    for cloud in clouds:
        showClouds.add_cloud(*cloud)
    gui.Application.instance.run()
