#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 hawkey
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import cv2
import time


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # translate
        res[:3, 3] -= self.center
        
        ## Convert pose
        res[..., 1] *= -1
        res[..., 2] *= -1

        #pose_pre = np.eye(4)
        #pose_pre[1, 1] *= -1
        #pose_pre[2, 2] *= -1
        #res = pose_pre @ res @ pose_pre

        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        #self.radius *= 1.1 ** (-delta)
        self.radius += delta * 0.25

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])


class NeRFGUI:
    def __init__(
        self,
        system,
        W=800,
        H=800,
        radius=1.0,
        fovy=30,
        debug=True
    ):
        self.system = system
        self.system.render_fn = self.system.render_fn.cuda()
        self.system.render_fn.eval()
        self.train_dataset = system.dm.train_dataset

        if 'tarot' in self.system.cfg.dataset.collection and self.system.cfg.dataset.use_ndc:
            fovy = 75
            radius = 0.0
            self.pan_factor = 0.25
        elif 'tarot' in self.system.cfg.dataset.collection and not self.system.cfg.dataset.use_ndc:
            fovy = 60
            radius = -1.0
            self.pan_factor = 0.1
        else:
            self.pan_factor = 1.0

        self.W = W
        self.H = H

        self.cam = OrbitCamera(W, H, r=radius, fovy=fovy)
        self.debug = debug

        self.training = False
        self.step = 0 # training step 

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.mode = 'image' # choose from ['image', 'depth']

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16

        self.flip = 'fabien' in self.system.cfg.dataset.collection or 'tarot' in self.system.cfg.dataset.collection
        self.transpose = 'fabien' in self.system.cfg.dataset.collection

        self.pan_dx = 0
        self.pan_dy = 0

        self.rot_dx = 0
        self.rot_dy = 0

        dpg.create_context() # TODO: enable again
        self.register_dpg()
        self.test_step()
        

    def __del__(self):
        #return
        dpg.destroy_context() # TODO: enable again


    def prepare_buffer(self, outputs):
        return outputs['image']
    
    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?
        self.need_update = True

        loop_length = 2.0
        t = (time.time() % 2.0) / 2.0

        if self.need_update:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            # Width, height, intrinsics
            W = min(int(self.W * self.downscale), self.W)
            H = min(int(self.H * self.downscale), self.H)

            K = np.eye(3)
            K[0, 0] = self.cam.intrinsics[0] * self.downscale
            K[1, 1] = self.cam.intrinsics[1] * self.downscale
            K[0, 2] = self.cam.intrinsics[2] * self.downscale
            K[1, 2] = self.cam.intrinsics[3] * self.downscale

            # Get coords
            num_frames = self.system.cfg.dataset.num_frames if 'num_frames' in self.system.cfg.dataset else 2
            coords = self.train_dataset.get_coords_from_camera(
                self.cam.pose,
                np.round(t * num_frames - 1) / (num_frames - 1),
                0,
                K,
                W,
                H,
                'cuda'
            )
            #coords = self.train_dataset.get_coords(0).to('cuda')

            # Run forward
            rgb_output = self.system(coords)['rgb'].view(H, W, 3).cpu().numpy()
            #rgb_output = self.system(coords)['rgb'].view(567, 1008, 3).cpu().numpy()
            #rgb_output = self.system(coords)['rgb'].view(512, 512, 3).cpu().numpy()

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # Write out image (temporary)
            #rgb_output = cv2.cvtColor(rgb_output, cv2.COLOR_BGR2RGB)
            #cv2.imwrite('tmp.png', np.uint8(rgb_output * 255))

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))

                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:

                if self.transpose:
                    rgb_output = rgb_output.transpose(1, 0, 2)

                if self.flip:
                    rgb_output = np.flip(rgb_output, axis=0)

                self.render_buffer = np.ascontiguousarray(rgb_output).astype(np.float32)

                self.need_update = False

            print(f'{t:.4f}ms ({int(1000/t)} FPS)')
            #return

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):
        #return

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True

                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution, callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)
                

            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1] * 0.5
            dy = app_data[2] * 0.5

            if self.transpose:
                tmp = dx
                dx = dy
                dy = tmp

            if self.flip:
                dy = -dy
            
            rot_dx = dx - self.rot_dx
            rot_dy = dy - self.rot_dy

            self.cam.orbit(rot_dx, rot_dy)
            self.need_update = True

            self.rot_dx = dx
            self.rot_dy = dy


            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            #dx = app_data[1] * 4.5
            #dy = app_data[2] * 4.5
            dx = app_data[1] * 15.0 * self.pan_factor
            dy = app_data[2] * 15.0 * self.pan_factor

            if self.transpose:
                tmp = dx
                dx = dy
                dy = tmp

            if self.flip:
                dx = -dx


            pan_dx = dx - self.pan_dx
            pan_dy = dy - self.pan_dy

            self.cam.pan(pan_dx, pan_dy)
            self.need_update = True

            self.pan_dx = dx
            self.pan_dy = dy

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_mouse_down(sender, app_data):
            self.pan_dx = 0
            self.pan_dy = 0

            self.rot_dx = 0
            self.rot_dy = 0

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=callback_mouse_down)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        
        dpg.create_viewport(title='HyperReel', width=self.W, height=self.H, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):
        #while True:
        #    self.test_step()

        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()