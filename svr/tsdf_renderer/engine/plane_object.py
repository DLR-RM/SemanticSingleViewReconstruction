import numpy as np


from OpenGL.GL import *
from OpenGL.GLUT import *


from engine.math_gl import *
from svr.tsdf_renderer.engine.RenderObject import RenderObject, TSDFRenderObject


class PlaneObject(RenderObject):

    def __init__(self, name: str):
        super().__init__(name)
        self._plane_verts = np.array([np.array(ele) for ele in [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]]).astype(GLfloat)
        self._plane_verts *= 2
        self._plane_verts -= 1
        self._plane_normals = np.array([[0,0,1]] * 4).astype(GLfloat)
        self._plane_indices = np.array([0,1,2,0,2,3]).astype(GLuint)
        self._class_numbers = np.ones(self._plane_verts.shape[0], dtype=GLfloat)
        self._model_matrix[0][0] = 0
        self._model_matrix[1][0] = -1
        self._model_matrix[0][1] = 16. / 9. * 3. / 4.0
        self._model_matrix[1][1] = 0
        self._model_matrix[3][0] = -3
        self._model_matrix[3][1] = -3.895
        self._model_matrix[2][3] = 2.5

    def render(self, vp, mapping):
        print("This fct. should be overwritten")

    def delete(self):
        print("This fct. should be overwritten")

    def flip_collapse_mode(self):
        pass

    def render(self, vp, mapping):
        if TSDFRenderObject.texture_id is None:
            return
        if not self._inited:
            self._bind_to_card()
        null = c_void_p(0)
        mvp = self._model_matrix

        unproj_mat = mat4.identity()
        glUniformMatrix4fv(mapping['MVP'], 1, GL_FALSE,mvp.data)
        glUniformMatrix4fv(mapping['unproj_mat'], 1, GL_FALSE, unproj_mat.data)
        glUniformMatrix4fv(mapping['M'], 1, GL_FALSE, self._model_matrix.data)

        # activate the texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, TSDFRenderObject.texture_id)
        glUniform1i(mapping['use_unproject'], 1)
        # Set our "myTextureSampler" sampler to user Texture Unit 0
        glUniform1i(mapping['texture_id'], 0)

        glUniform1i(mapping['UseTexture'], 1)

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
        glDrawElements(GL_TRIANGLES, len(self._plane_indices), GL_UNSIGNED_INT,	null)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)

    def _bind_to_card(self):
        with TSDFRenderObject.bind_to_card_lock:
            self._mapping['vertex_buffer'] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['vertex_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._plane_verts) * 4 * 3, self._plane_verts, GL_STATIC_DRAW)

            self._mapping['normal_buffer'] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['normal_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._plane_normals) * 4 * 3, self._plane_normals, GL_STATIC_DRAW)

            if "class_buffer" not in self._mapping:
                self._mapping['class_buffer'] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['class_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._class_numbers) * 4, self._class_numbers, GL_STATIC_DRAW)

            # Generate a buffer for the indices as well
            self._mapping['elementbuffer'] = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mapping['elementbuffer'])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self._plane_indices) * 4, self._plane_indices, GL_STATIC_DRAW)
            self._inited = True

