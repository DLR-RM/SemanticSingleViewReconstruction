import threading
import random
from pathlib import Path

from PIL import Image
import numpy as np

import glfw

from OpenGL.GL import *
from OpenGL.GL.ARB import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.special import *
from OpenGL.GL.shaders import *
from engine.math_gl import *
import engine.common as common
import math as mathf

from svr.all_in_one_client import AllInOneClient
from svr.tsdf_renderer.engine.RenderObject import TSDFRenderObject
from svr.tsdf_renderer.engine.plane_object import PlaneObject


def windows_focus_callback(window, focus):
    print("focus")

class WindowManager(object):

    def __init__(self, window_size, title=None):
        self.window = None
        self._window_size = window_size
        if title:
            self._title = title
        else:
            self._title = 'TSDF renderer'
        self._mapping = {}
        self._render_objects = []
        self._org_camera_position = vec3( 0.5, -0.5, 3 )
        self._camera_position = self._org_camera_position.copy()
        self._org_angle = (3.14, 0.0)
        self._horizontalAngle = self._org_angle[0]
        self._verticalAngle = self._org_angle[1]
        self._focus = False
        self._lastTime = None
        self._last_key_pressed = glfw.get_time()
        self._last_any_key_pressed = glfw.get_time()
        self._all_in_one_client = None
        self._used_tsdf_object = None
        self._used_plane_object = None
        self._time_at_last_tsdf_refresh = 0.0
        self._never_updated_tsdf_before = True
        self._list_of_test_imgs = list((Path(__file__).parent.parent.parent.parent / "demo").glob("*.jpg"))
        if not self._list_of_test_imgs:
            raise FileNotFoundError("No test images were found in the demo folder!")
        random.shuffle(self._list_of_test_imgs)
        self._index_in_list_of_imgs = 0

    def init(self):
        assert glfw.init(), 'Glfw Init failed!'
        self.window = glfw.create_window(self._window_size[0], self._window_size[1], self._title, None, None)
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.make_context_current(self.window)
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS, True)

        # Enable depth test
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_3D)
        # Accept fragment if it closer to the camera than the former one
        glDepthFunc(GL_LESS)
        # disable vsync
        glfw.swap_interval(0)
        self._init_shaders()

    def set_all_in_one_client(self, client: AllInOneClient):
        self._all_in_one_client = client
        self._used_tsdf_object = TSDFRenderObject()
        self._used_plane_object = PlaneObject("Image")

    def _init_shaders(self):
        self._mapping['vertex_array_id'] = glGenVertexArrays(1)
        glBindVertexArray(self._mapping['vertex_array_id'])
        vertex_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "VertexShader.glsl")
        fragment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "FragmentShader.glsl")
        self._mapping['program_id'] = common.LoadShaders(vertex_path, fragment_path)
        program_id = self._mapping['program_id']
        self._mapping['texture_id'] = glGetUniformLocation(program_id, "myTextureSampler")
        use_obj = True
        if use_obj:
            self._mapping['obj_class_id'] = glGetUniformLocation(program_id, "objClassSampler")


        # Get a handle for our "MVP" uniform
        self._mapping['MVP'] = glGetUniformLocation(program_id, "MVP")
        self._mapping['unproj_mat'] = glGetUniformLocation(program_id, "unproj_mat")
        self._mapping['V'] = glGetUniformLocation(program_id, "V")
        self._mapping['M'] = glGetUniformLocation(program_id, "M")
        self._mapping['light_id'] = glGetUniformLocation(program_id, "LightPosition_worldspace")
        self._mapping['UseTexture'] = glGetUniformLocation(program_id, "UseTexture")
        self._mapping['use_unproject'] = glGetUniformLocation(program_id, "use_unproject")

    def _windows_focus_callback(self, focus):
        self._focus = focus

    def add_render_object(self, object):
        self._render_objects.append(object)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self._mapping['program_id'])
        vp = self._compute_matrices_from_input()
        lightPos = vec3(2,2,6)
        glUniform3f(self._mapping['light_id'], lightPos.x, lightPos.y, lightPos.z)
        if self._used_tsdf_object:
            self._used_tsdf_object.render(vp, self._mapping)
        if self._used_plane_object:
            self._used_plane_object.render(vp, self._mapping)

        glfw.swap_buffers(self.window)

        # Poll for and process events
        glfw.poll_events()
        if glfw.get_time() - self._time_at_last_tsdf_refresh > 5.0 \
                and glfw.get_time() - self._last_any_key_pressed > 3 or self._never_updated_tsdf_before:

            thread = threading.Thread(target=self.update_voxel_grid)
            thread.run()
            self._never_updated_tsdf_before = False

    def time_until_next_update(self):
        return max(0.0, np.max([5 - (glfw.get_time() - self._time_at_last_tsdf_refresh), 3-(glfw.get_time() - self._last_any_key_pressed)]))

    def get_next_color_img(self) -> np.ndarray:
        img = np.array(Image.open(self._list_of_test_imgs[self._index_in_list_of_imgs]).resize((512, 512))).astype(np.float32)
        self._index_in_list_of_imgs = (self._index_in_list_of_imgs + 1) % len(self._list_of_test_imgs)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def update_voxel_grid(self):
        print("Update voxel grid")
        img = self.get_next_color_img()
        voxel_output, class_output = self._all_in_one_client.get_scene(img)

        self._used_tsdf_object.update(voxel_output, class_output, img)
        self._time_at_last_tsdf_refresh = glfw.get_time()

    def run_window(self):
        last_time = glfw.get_time()
        frames = 0
        while glfw.get_key(self.window, glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(self.window):

            current_time = glfw.get_time()
            if current_time - last_time >= 0.5:
                glfw.set_window_title(self.window, f"FPS: {frames}, Time until updated: {self.time_until_next_update():.1f}s")
                frames = 0
                last_time = current_time
            self.render()
            frames += 1

        for object in self._render_objects:
            object.delete()
        glDeleteProgram(self._mapping['program_id'])
        glDeleteVertexArrays(1, [self._mapping['vertex_array_id']])
        glfw.terminate()

    def _is_key_pressed(self, key):
        if isinstance(key, list):
            for ele in key:
                if self._is_key_pressed(ele):
                    self._last_any_key_pressed = glfw.get_time()
                    return True
        else:
            if glfw.get_key( self.window, key) == glfw.PRESS:
                self._last_any_key_pressed = glfw.get_time()
                return True
        return False

    def _compute_matrices_from_input(self):
        old_focus = self._focus
        self._focus = glfw.get_window_attrib(self.window, glfw.FOCUSED)
        if old_focus != self._focus and self._focus:
            glfw.set_cursor_pos(self.window, self._window_size[0]/2, self._window_size[1]/2)

        FoV = 45
        mouse_speed =  0.001
        speed = 5.0
        # glfwget_time is called only once, the first time this function is called
        if self._lastTime is None:
            self._lastTime = glfw.get_time()

        currentTime = glfw.get_time()
        if self._focus:
            if self._is_key_pressed(glfw.KEY_O):
                self._camera_position = self._org_camera_position.copy()
                self._horizontalAngle = self._org_angle[0]
                self._verticalAngle = self._org_angle[1]

            deltaTime = currentTime - self._lastTime
            #if deltaTime > 0.01:
            xpos,ypos = glfw.get_cursor_pos(self.window)

            # Reset mouse position for next frame
            if xpos != self._window_size[0]/2 or ypos != self._window_size[1]/2:
                glfw.set_cursor_pos(self.window, self._window_size[0]/2, self._window_size[1]/2)

            horizontal_diff = float(self._window_size[0]/2.0 - xpos )
            vertical_diff = float( self._window_size[1]/2.0 - ypos )
            if horizontal_diff > 0 or vertical_diff > 0:
                self._last_any_key_pressed = glfw.get_time()
            # Compute new orientation
            self._horizontalAngle += mouse_speed * horizontal_diff
            self._verticalAngle   += mouse_speed * vertical_diff

        # Direction : Spherical coordinates to Cartesian coordinates conversion
        direction = vec3(
            mathf.cos(self._verticalAngle) * mathf.sin(self._horizontalAngle),
            mathf.sin(self._verticalAngle),
            mathf.cos(self._verticalAngle) * mathf.cos(self._horizontalAngle)
        )

        # Right vector
        right = vec3(
            mathf.sin(self._horizontalAngle - 3.14/2.0),
            0.0,
            mathf.cos(self._horizontalAngle - 3.14/2.0)
        )

        # Up vector
        up = vec3.cross( right, direction )

        if self._focus:
            # Move forward
            if self._is_key_pressed([glfw.KEY_W, glfw.KEY_UP]):
                self._camera_position += direction * deltaTime * speed

            # Move backward
            if self._is_key_pressed([glfw.KEY_S, glfw.KEY_DOWN]):
                self._camera_position -= direction * deltaTime * speed

            # Strafe right
            if self._is_key_pressed([glfw.KEY_D, glfw.KEY_RIGHT]):
                self._camera_position += right * deltaTime * speed

            # Strafe left
            if self._is_key_pressed([glfw.KEY_A, glfw.KEY_LEFT]):
                self._camera_position -= right * deltaTime * speed

            if self._is_key_pressed(glfw.KEY_C):
                new_press =  glfw.get_time()
                if new_press - self._last_key_pressed > 0.15:
                    if self._used_tsdf_object:
                        self._used_tsdf_object.flip_collapse_mode()
                    self._last_key_pressed = new_press

            if self._is_key_pressed(glfw.KEY_T):
                new_press =  glfw.get_time()
                if new_press - self._last_key_pressed > 0.15:
                    if self._used_tsdf_object:
                        self._used_tsdf_object.flip_color_mode()
                    self._last_key_pressed = new_press

            if self._is_key_pressed(glfw.KEY_U):
                new_press =  glfw.get_time()
                if new_press - self._last_key_pressed > 0.15:
                    if self._used_tsdf_object:
                        self._used_tsdf_object.flip_unproject()
                    self._last_key_pressed = new_press


        ProjectionMatrix = mat4.perspective(FoV, self._window_size[0] / float(self._window_size[1]), 0.1, 100.0)
        ViewMatrix       = mat4.lookat(self._camera_position, self._camera_position+direction, up)


        vp = ProjectionMatrix * ViewMatrix

        glUniformMatrix4fv(self._mapping['V'], 1, GL_FALSE, ViewMatrix.data)
        self._lastTime = currentTime

        return vp



