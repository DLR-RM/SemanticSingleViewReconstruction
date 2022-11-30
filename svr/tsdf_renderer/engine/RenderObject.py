import sys

from OpenGL.GL import *
from OpenGL.GLUT import *

import numpy as np

main_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(main_path)

from engine.math_gl import *
from threading import Lock
from svr.implicit_tsdf_decoder import marching_cubes


def opengl_error_check():
    error = glGetError()
    if error != GL_NO_ERROR:
        print("OPENGL_ERROR: ", gluErrorString(error))

class RenderObject(object):

    def __init__(self, name):
        self._name = name
        self._mapping = {}
        self._model_matrix = mat4.identity()
        self._inited = False

    def render(self, vp, mapping):
        print("This fct. should be overwritten")

    def delete(self):
        print("This fct. should be overwritten")

    def flip_collapse_mode(self):
        pass

class TSDFRenderObject(RenderObject):
    bind_to_card_lock = Lock()
    texture_id = None

    def __init__(self, name=None):
        if name:
            super(TSDFRenderObject, self).__init__(name)
        else:
            super(TSDFRenderObject, self).__init__('TSDF_Volume')
        self._use_texture = True
        self.use_unproject = 1
        self._flip_plane_mode = False
        self._model_matrix[0][0] = 0
        self._model_matrix[1][0] = 1
        self._model_matrix[0][1] = -1
        self._model_matrix[1][1] = 0
        scale = 1.0
        xFov = 0.5
        near = 1.0
        far = 4.0
        yFov = 0.388863
        width = 1. / np.tan(xFov)
        height = 1. / np.tan(yFov)
        proj_mat = np.array([[width/scale, 0, 0, 0],[0, height/scale, 0, 0],[0, 0, (near + far) / (near - far)/scale, (2*near*far) / (near- far)/scale], [0, 0, 1, 0]])
        result = np.linalg.inv(proj_mat)
        result = result.astype(GLfloat)
        self.unproj_mat = mat4.identity()
        for i in range(4):
            for j in range(4):
                if abs(result[i,j]) < 1e-5:
                    self.unproj_mat[i][j] = 0
                else:
                    self.unproj_mat[i][j] = result[i,j]
        self._loaded_once = False
        self._used_texture_value = 0


    def update(self, voxel: np.ndarray, class_output: np.ndarray, image: np.ndarray):
        if image is not None:
            new_used_texture_value = 0 if np.mean(np.var(image)) < 1e-3 else 1
        else:
            new_used_texture_value = 0
        if self._used_texture_value == 0 and new_used_texture_value == 1:
            self._used_texture_value = 1

        # voxel set up
        verts, _, faces, normals, vertex_class_coordinates = TSDFRenderObject.perform_marching_cubes(voxel)
        verts[:, 1] *= -1
        verts = verts[:, [1, 0, 2]]
        verts[:, 2] *= -1
        print("Verts: {}, faces: {}".format(len(verts), len(faces)))
        vertex_class_coordinates = vertex_class_coordinates.astype(np.int32)
        used_classes = class_output[vertex_class_coordinates[:, 0], vertex_class_coordinates[:, 1], vertex_class_coordinates[:, 2]]
        used_classes = used_classes.astype(GLfloat)

        verts = verts.astype(GLfloat)
        normals = normals.astype(GLfloat)
        faces = faces.flatten()

        self._indexed_vertices = verts[faces]
        self._indexed_classes = used_classes[faces]
        self._indexed_normals = normals[faces]
        self._indices = np.arange(self._indexed_vertices.shape[0], dtype=GLuint)

        # image set up
        if image is not None and len(image.shape) == 3 and image.shape[0] == 512 and image.shape[1] == 512 and image.shape[2] == 4:
            self._image = image.astype(np.uint8)
        else:
            if image is not None:
                if len(image.shape) == 3 and image.shape[0] == 512 and image.shape[1] == 512 and image.shape[2] == 3:
                    self._image = np.concatenate([image.astype(np.uint8), np.ones((512,512,1), dtype=np.uint8) * 255], axis=2).astype(np.uint8)
                else:
                    print("The image has not the right form: " + str(image.shape) + ", for " + str(self._name))
                    self._image = np.zeros((512, 512, 4), dtype=np.uint8)
            else:
                self._image = np.zeros((512, 512, 4), dtype=np.uint8)
        self._inited = False
        self._loaded_once = True


    @staticmethod
    def perform_marching_cubes(voxel):
        threshold = np.max([np.min(voxel) + 1e-8, 0])
        print('Use threshold: {}, min: {}, max: {}'.format(threshold, np.min(voxel), np.max(voxel)))
        #from skimage import measure
        #return measure.marching_cubes(voxel, 0)
        return marching_cubes(voxel, correct_depth=True, unproject=False, is_curved_space=True)

    def init(self, use_preloaded_img_id=None):
        if use_preloaded_img_id is not None:
            self._use_preloaded_img = use_preloaded_img_id
        self._bind_to_card()
        self._inited = True

    def flip_color_mode(self):
        self._used_texture_value = int((self._used_texture_value + 1) % 3)

    def flip_plane_mode(self):
        self._flip_plane_mode = not self._flip_plane_mode

    def flip_unproject(self):
        self.use_unproject = int((self.use_unproject + 1) % 3)

    def _bind_to_card(self):
        with TSDFRenderObject.bind_to_card_lock:
            if "texture" in self._mapping:
                glDeleteBuffers(1, [self._mapping['texture']])
            self._mapping['texture'] = glGenTextures(1)
            TSDFRenderObject.texture_id = self._mapping["texture"]
            glBindTexture(GL_TEXTURE_2D, self._mapping['texture'])
            glPixelStorei(GL_UNPACK_ALIGNMENT,1)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

            glTexImage2D(GL_TEXTURE_2D, 0, 3, 512, 512, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, self._image)
            opengl_error_check()

            if "vertex_buffer" in self._mapping:
                glDeleteBuffers(1, [self._mapping['vertex_buffer']])
            self._mapping['vertex_buffer'] = glGenBuffers(1)

            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['vertex_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._indexed_vertices) * 4 * 3, self._indexed_vertices, GL_STATIC_DRAW)

            if "normal_buffer"in self._mapping:
                glDeleteBuffers(1, [self._mapping['normal_buffer']])
            self._mapping['normal_buffer'] = glGenBuffers(1)

            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['normal_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._indexed_normals) * 4 * 3, self._indexed_normals, GL_STATIC_DRAW)

            if "class_buffer" in self._mapping:
                glDeleteBuffers(1, [self._mapping['class_buffer']])
            self._mapping['class_buffer'] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['class_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._indexed_classes) * 4, self._indexed_classes, GL_STATIC_DRAW)

            # Generate a buffer for the indices as well
            if "element_buffer" in self._mapping:
                glDeleteBuffers(1, [self._mapping['elementbuffer']])
            self._mapping['elementbuffer'] = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mapping['elementbuffer'])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self._indices) * 4, self._indices, GL_STATIC_DRAW)

    def render(self, vp, mapping):
        if not self._loaded_once:
            return

        if not self._inited:
            self.init()
        null = c_void_p(0)
        mvp = vp * self._model_matrix
        glUniformMatrix4fv(mapping['unproj_mat'], 1, GL_TRUE, self.unproj_mat.data)

        glUniformMatrix4fv(mapping['MVP'], 1, GL_FALSE, mvp.data)
        glUniform1i(mapping['use_unproject'], self.use_unproject)
        glUniformMatrix4fv(mapping['M'], 1, GL_FALSE, self._model_matrix.data)

        if self._used_texture_value == 1:
            # activate the texture
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._mapping['texture'])
            # Set our "myTextureSampler" sampler to user Texture Unit 0
            glUniform1i(mapping['texture_id'], 0)

        glUniform1i(mapping['UseTexture'], self._used_texture_value)

        # activate vertex data
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self._mapping['vertex_buffer'])
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, null)

        # activate normal data
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._mapping['normal_buffer'])
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, null)

        # activate class data
        glEnableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, self._mapping['class_buffer'])
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, null)

        # activate indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mapping['elementbuffer'])

        # Draw the triangles
        glDrawElements(GL_TRIANGLES, len(self._indices), GL_UNSIGNED_INT,	null)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)

    def delete(self):
        glDeleteBuffers(1, [self._mapping['vertex_buffer']])
        glDeleteBuffers(1, [self._mapping['normal_buffer']])
        glDeleteBuffers(1, [self._mapping['class_buffer']])
        glDeleteBuffers(1, [self._mapping['elementbuffer']])
        glDeleteBuffers(1, [self._mapping['texture']])

    def get_texture_id(self) -> int:
        if 'texture' in self._mapping:
            return self._mapping['texture']
        else:
            # this has not been set
            return 0


