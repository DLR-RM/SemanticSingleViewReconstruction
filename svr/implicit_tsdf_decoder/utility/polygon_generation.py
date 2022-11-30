import time
from typing import Tuple, Union
from pathlib import Path

import numpy as np
from skimage import measure


def marching_cubes(voxel_grid: np.ndarray, correct_depth: bool = False,
                   unproject: bool = False, is_curved_space: bool = True):
    try:
        verts, faces, normals, _ = measure.marching_cubes(voxel_grid, 0)
    except ValueError as e:
        # the predicted voxel grid does not contain any negative and positive area
        print("No area found!")
        verts, org_verts = np.empty((0, 3)), np.empty((0, 3))
        vertex_class_coordinates, faces, normals = np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
        return verts, vertex_class_coordinates, faces, normals, org_verts
    org_verts = verts.copy()
    vertex_class_coordinates = verts.astype(np.int64)
    side_resolution = voxel_grid.shape[0]
    verts /= side_resolution
    verts -= 0.5
    verts *= 2.0
    if unproject and correct_depth:
        raise Exception("The mesh can not be unprojected and depth corrected at the same time")
    if not is_curved_space and correct_depth:
        raise Exception("Correct depth only works in curved spaces!")
    if unproject:
        xFov, yFov, near, far = 0.5, 0.388863, 1.0, 4.0
        height, width = 1.0 / np.tan(yFov), 1. / np.tan(xFov)
        proj_mat = np.array([[width, 0, 0, 0], [0, height, 0, 0],
                             [0, 0, (near + far) / (near - far), (2 * near * far) / (near - far)],
                             [0, 0, -1, 0]])
        unproject_mat = np.linalg.inv(proj_mat)
        if is_curved_space:
            verts[:, 2] = ((np.square((verts[:, 2] - 1) * -0.5)) - 0.5) * -2
        verts = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
        verts = verts @ unproject_mat.transpose()
        for i in range(3):
            verts[:, i] /= verts[:, 3]
        verts = verts[:, :3]
    elif correct_depth:
        verts[:, 2] = ((np.square((verts[:, 2] - 1) * -0.5)) - 0.5) * -2
    return verts, vertex_class_coordinates, faces, normals, org_verts


def convert_final_block_to_mesh(final_block: np.ndarray, final_class_block: np.ndarray,
                                vertex_offset: int = 1, move_vec: np.ndarray = None,
                                correct_depth: bool = False, unproject: bool = False,
                                add_uv_texture: bool = False,
                                ignored_blocks_z: float = 8.5,
                                verbose: bool = True) -> Tuple[str, int]:
    if verbose:
        start_time = time.time()
        print(f"Perform marching cubes on final_block, min value: {np.min(final_block)}, "
              f"max value: {np.max(final_block)}")
    verts, vertex_class_coordinates, faces, normals, org_verts = marching_cubes(final_block, correct_depth, unproject,
                                                                                is_curved_space=True)
    if verbose:
        print(f"Took {time.time() - start_time}s to perform marching cubes.")
        start_time = time.time()

    used_classes = final_class_block[vertex_class_coordinates[:, 0], vertex_class_coordinates[:, 1],
                                     vertex_class_coordinates[:, 2]]
    used_classes_per_face = used_classes[faces]
    used_classes_per_face = np.unique(used_classes_per_face, axis=-1)[:, 0]
    if move_vec is not None:
        verts += move_vec
    verts = verts.astype(str)
    text = "\n".join("v " + " ".join(v) for v in verts) + "\n"
    last_texture = None
    used_classes_per_face = used_classes[faces]
    used_classes_per_face = np.unique(used_classes_per_face, axis=-1)[:, 0]
    if add_uv_texture:
        uv_coords = org_verts.astype(np.float32) / final_block.shape[0]
        uv_coords = uv_coords[:, :2].astype(str)
        text += "\n".join("vt " + " ".join(u) for u in uv_coords) + "\n"
        normal_z_coord = normals[:, 2]
        # z_faces = np.sum((normal_z_coord[faces] < 0.0).astype(np.int32), axis=-1) > 1

        depth = np.ones((final_block.shape[0], final_block.shape[1])) * 1000
        depth_coords = (org_verts.astype(np.float32) * (depth.shape[0] / final_block.shape[0])).astype(np.int32)[:, :2]
        depth_coords = np.clip(depth_coords, 0, depth.shape[0])
        for z_value, coord in zip(org_verts[:, 2], depth_coords):
            depth[coord[0], coord[1]] = np.min([depth[coord[0], coord[1]], z_value])
        min_coord_values = np.array([depth[coord[0], coord[1]] for coord in depth_coords])
        is_not_correct = np.abs(min_coord_values - org_verts[:, 2]) > ignored_blocks_z
        not_correct_faces = np.sum((is_not_correct[faces]).astype(np.int32), axis=-1) > 1
        # not_correct_faces = np.any(np.array([not_correct_faces, z_faces]), axis=0)
        faces += 1
        pink_faces = faces[not_correct_faces].astype(str)
        image_faces = faces[np.logical_not(not_correct_faces)].astype(str)
        text += f"usemtl img_material\n"
        text += "\n".join("f " + " ".join(f"{e}/{e}" for e in f) for f in image_faces) + "\n"
        text += f"usemtl pink_material\n"
        text += "\n".join("f " + " ".join(f"{e}/{e}" for e in f) for f in pink_faces) + "\n"
    else:
        faces += vertex_offset
        faces = faces.astype(str)
        for f, used_class_per_face in zip(faces, used_classes_per_face):
            texture_name = "mtl_{}".format(used_class_per_face)
            if last_texture is None or last_texture != texture_name:
                text += "usemtl {}\n".format(texture_name)
                last_texture = texture_name
            text += "f " + " ".join(f) + "\n"
    if verbose:
        print(f"Obj file generation took: {time.time() - start_time}")
    return text, verts.shape[0]


def generate_image_mtl(output_file_path: Union[str, Path]):
    with Path(output_file_path).with_suffix(".mtl").open("w") as file:
        mtl_text = f"newmtl img_material\n"
        mtl_text += "Ka 1.000 1.000 1.000\n"
        mtl_text += "Kd 1.000 1.000 1.000\n"
        mtl_text += f"map_Ka {Path(output_file_path).with_suffix('.png').name}\n"
        mtl_text += f"map_Kd {Path(output_file_path).with_suffix('.png').name}\n"
        mtl_text += f"newmtl pink_material\n"
        mtl_text += "Ka 1.000 0.078 0.576\n"
        mtl_text += "Kd 1.000 0.078 0.576\n"
        file.write(mtl_text)


def generate_mtl_file(output_file_path: Union[str, Path], number_of_classes: int = 10):
    used_colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                   (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
                   (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180),
                   (0, 0, 128), (85, 85, 85), (170, 170, 170), (255, 255, 255), (0, 0, 0)]

    with Path(output_file_path).with_suffix(".mtl").open("w") as file:
        mtl_text = ""
        for i in range(number_of_classes):
            mtl_text += f"newmtl mtl_{i}\n"
            color = np.array(used_colors[i], dtype=np.float32)
            color /= 255.0
            mtl_text += f"\tKa " + " ".join(str(e) for e in color) + "\n"
            mtl_text += f"\tKd " + " ".join(str(e) for e in color) + "\n"
        file.write(mtl_text)
