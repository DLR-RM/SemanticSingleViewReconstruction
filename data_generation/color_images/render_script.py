import blenderproc as bproc
bproc.SetupUtility.setup_pip(["numba==0.56.4"])

import argparse
import os
import json
import glob
import random
import time
import sys
from pathlib import Path

import h5py
import numpy as np
import numba
import bpy
from mathutils import Matrix, Euler

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from data_generation.color_images.load_3d_front import load_front3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help="Path to the .json file of the 3d front scene to load")
    parser.add_argument('future_model_path', help="Future model path")
    parser.add_argument('front_3D_texture_path', help="Future model path")
    parser.add_argument('output_dir', help="Path to where the final hdf5 files, will be saved.")
    parser.add_argument('front_3D_obj_mesh_folder', help="Front 3D folder in which the meshes of the objects are saved.")
    parser.add_argument('cc_textures', help="Path to the cctextures folder")
    args = parser.parse_args()

    os.environ["BLENDER_PROC_RANDOM_SEED"] = "1"

    cc_texture_folder = args.cc_textures
    if not os.path.exists(cc_texture_folder):
        raise Exception(f"The given folder path does not exist: {cc_texture_folder}")

    bproc.init()

    amount_of_poses = 25
    # load the objects into the scene
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    label_mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    with bproc.utility.BlockStopWatch("Load 3D objects"):
        loaded_objects = load_front3d(args.json_path, args.future_model_path, args.front_3D_texture_path,
                                      label_mapping, front_3D_obj_mesh_folder=args.front_3D_obj_mesh_folder)

    # set the light bounces
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, ao_bounces_render=200, max_bounces=200,
                                     transmission_bounces=200, transparent_max_bounces=200, volume_bounces=0)

    # define the camera intrinsics
    bproc.camera.set_intrinsics_from_blender_params(1, 512, 512, lens_unit="FOV", pixel_aspect_x=1.33333333333333,
                                                    clip_start=1.0)

    # Init bvh tree containing all mesh objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if
                                                           isinstance(o, bproc.types.MeshObject)])

    # Init sampler for sampling locations inside the loaded front3D house
    point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects, amount_of_objects_needed_per_room=0)

    list_of_used_names = ["tea table", "children cabinet", "lazy sofa", "chaise longue sofa", "appliance",
                          "round end table", "dining chair", "armchair", "bed", "kids bed",
                          "footstool / sofastool / bed end stool / stool", "table", "bath", "sofa", "tv stand",
                          "double bed",
                          "single bed", "shelf", "nightstand", "desk", "dressing table", "wardrobe", "bunk bed",
                          "bed frame", "three-seat / multi-person sofa", "l-shaped sofa"]
    list_of_ids = [label_mapping.id_from_label(id_label) for id_label in list_of_used_names]


    def is_special_object(obj):
        return obj.get_cp("category_id") in list_of_ids


    special_objects = [o for o in loaded_objects if is_special_object(o)]
    print("Special objects: {}".format([o.get_name() for o in special_objects]))
    list_of_used_cam2worlds = []


    def get_point_cloud(cam2world_matrix: np.ndarray, bvh_tree, sqrt_number_of_rays: int = 10):
        cam2world_matrix = Matrix(cam2world_matrix)

        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data
        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]
        # Go in discrete grid-like steps over plane
        position = cam2world_matrix.to_translation()
        dists_result = np.zeros((sqrt_number_of_rays, sqrt_number_of_rays, 3))
        for x in range(0, sqrt_number_of_rays):
            for y in range(0, sqrt_number_of_rays):
                # Compute current point on plane
                end = frame[0] + vec_x * x / float(sqrt_number_of_rays - 1) + vec_y * y / float(sqrt_number_of_rays - 1)
                # Send ray from the camera position through the current point on the plane

                _, _, _, dist = bvh_tree.ray_cast(position, end - position)
                if dist is None:
                    return None
                direction = end - position
                direction /= np.linalg.norm(direction)
                dists_result[x, y] = position + direction * dist
        return dists_result.reshape((-1, 3))


    @numba.jit(nopython=True, fastmath=True)
    def hausdorff(XA, XB):
        nA = XA.shape[0]
        nB = XB.shape[0]
        final_dists = np.zeros(nA)
        for i in range(nA):
            cmin = np.inf
            for j in range(nB):
                d = np.sum(np.square(XA[i, :] - XB[j, :]))
                if d < cmin:
                    cmin = d
            final_dists[i] = cmin
        return np.sqrt(np.mean(final_dists))


    list_of_point_clouds = []
    with bproc.utility.BlockStopWatch("Running camera sampling"):
        poses = 0
        tries = 0
        start_time = time.time()
        used_special_coverage_score = 0.8
        hausdorff_dist = 0.5
        while tries < 100000 and poses < amount_of_poses:
            if tries == 25000:
                used_special_coverage_score = 0.5
            if tries == 50000:
                hausdorff_dist = 0.25
            # Sample point inside house
            height = np.random.uniform(1.45, 1.85)
            location = point_sampler.sample(height)

            # Sample rotation (fix around X and Y axis)
            rotation = np.random.uniform([0.7853, 0, 0], [1.483, 0, 6.283185307])
            cam2world_matrix = bproc.math.build_transformation_mat(location, Euler(rotation).to_matrix())

            # Check that obstacles are at least 1 meter away from the camera and make sure the view interesting enough
            is_obstacle_in_view = bproc.camera.perform_obstacle_in_view_check(cam2world_matrix,
                                                                              {"min": 1.25, "avg": {"min": 1.5, "max": 3.5},
                                                                               "no_background": True}, bvh_tree,
                                                                              sqrt_number_of_rays=20)
            if is_obstacle_in_view and bproc.camera.scene_coverage_score(cam2world_matrix, special_objects=special_objects,
                                                                         special_objects_weight=4.0) > used_special_coverage_score:
                dist_point_cloud = get_point_cloud(cam2world_matrix, bvh_tree, sqrt_number_of_rays=15)
                if dist_point_cloud is None:
                    continue
                to_similar = False
                for used_point_cloud in list_of_point_clouds:
                    # if the average distance between the point clouds is smaller than a meter do not use it
                    if hausdorff(dist_point_cloud, used_point_cloud) < hausdorff_dist:
                        to_similar = True
                        break
                if to_similar:
                    continue
                list_of_used_cam2worlds.append(cam2world_matrix)
                list_of_point_clouds.append(dist_point_cloud)
                bproc.camera.add_camera_pose(cam2world_matrix)
                poses += 1
            tries += 1

    # set the amount of noise, which should be used for the color rendering
    bproc.renderer.set_noise_threshold(0.01)  # this is the default value
    bproc.renderer.set_max_amount_of_samples(512)

    # activate normal and distance rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])
    bproc.material.add_alpha_channel_to_textures(blurry_edges=True)

    modules = bproc.python.utility.Utility.Utility.initialize_modules([{"module": "writer.CameraStateWriter", "config": {
        "attributes_to_write": ["location", "rotation_forward_vec", "rotation_up_vec"],
        "destination_frame": ["X", "Z", "-Y"],
        "output_dir": args.output_dir
    }}])
    camera_state_writer_module = modules[0]
    # render the whole pipeline
    data = bproc.renderer.render()
    data["original_texture"] = [np.string_("True")] * len(data["colors"])

    # get the camera poses
    camera_state_writer_module.run()
    camera_poses_res = bproc.python.writer.WriterUtility._WriterUtility.load_registered_outputs({"campose"})
    data.update(camera_poses_res)

    # write the data to a .hdf5 container
    bproc.writer.write_hdf5(args.output_dir, data)


    def convert_hdf5_to_sdf_format(file_path):
        def to_string(l):
            return " ".join([str(e) for e in l])

        with h5py.File(file_path, "r") as file:
            if "campose" not in file:
                raise Exception("You have to use the CameraStateWriter in your blenderproc config pipeline.")

            cam_pose = json.loads(np.array(file["campose"]).tostring())[0]
            # is fixed for the whole method
            cam_pose_fov_x = 0.5
            cam_pose_fov_y = 0.388863

            rotation_forward_vec = np.array([float(e) for e in cam_pose["rotation_forward_vec"]])
            rotation_up_vec = np.array([float(e) for e in cam_pose["rotation_up_vec"]])

            new_pose = "  ".join([to_string(cam_pose["location"]), to_string(rotation_forward_vec),
                                  to_string(rotation_up_vec), str(cam_pose_fov_x), str(cam_pose_fov_y)])
        return new_pose


    obj_folder = os.path.join(args.front_3D_obj_mesh_folder, os.path.basename(args.json_path.replace(".json", "")))
    camera_pose_file = os.path.join(obj_folder, "camera_positions")
    # generate the corresponding TSDF volumes by first generating the camera poses in the correct format
    imgs_paths = glob.glob(os.path.join(args.output_dir, "*.hdf5"))
    new_poses = [""] * len(imgs_paths)
    for img_path in imgs_paths:
        use_this_file = False
        with h5py.File(img_path, "r") as file:
            if "campose" in file:
                use_this_file = True

        if use_this_file:
            new_pose = convert_hdf5_to_sdf_format(img_path)
            print("new pose: ", new_pose)
            number = int(os.path.basename(img_path)[:os.path.basename(img_path).rfind(".")])
            new_poses[number] = new_pose
        else:
            raise Exception("Something went wrong!")
    with open(camera_pose_file, "w") as file:
        file.write("\n".join(new_poses))

    org_materials = bproc.loader.load_ccmaterials(cc_texture_folder, preload=True)
    # remove alpha materials
    org_materials = [mat for mat in org_materials if
                     mat.get_principled_shader_value("Alpha") is not None and mat.get_principled_shader_value(
                         "Alpha") == 1.0]

    materials = random.sample(org_materials, k=75)

    sphere = bproc.object.create_primitive("SPHERE")
    for m in materials:
        sphere.add_material(m)
    bproc.loader.load_ccmaterials(cc_texture_folder, fill_used_empty_materials=True)
    sphere.delete()

    MeshObject = bproc.types.MeshObject
    for i in range(3):
        objs = bproc.object.get_all_mesh_objects()
        for obj in objs:
            for i in range(len(obj.get_materials())):
                if "emission" in obj.get_materials()[i].get_name().lower():
                    continue
                nodes = obj.get_materials()[i].get_nodes_with_type("ShaderNodeBsdfTransparent")
                if len(nodes) > 0:
                    # replace color of the leave
                    alpha_material = obj.get_materials()[i]
                    material = random.choice(materials)
                    texture_nodes = material.get_nodes_with_type("TexImage")

                    for tex_node in texture_nodes:
                        if "_Color" in str(tex_node.image):
                            used_node = tex_node
                            break

                    new_texture_node = alpha_material.nodes.new("ShaderNodeTexImage")
                    new_texture_node.image = tex_node.image
                    bsdf_nodes = alpha_material.get_nodes_with_type("BsdfPrincipled")
                    if len(bsdf_nodes) == 1:
                        bsdf_node = bsdf_nodes[0]
                    else:
                        raise Exception("There is more than one bsdf node")
                    alpha_material.links.new(new_texture_node.outputs["Color"], bsdf_node.inputs["Base Color"])
                else:
                    obj.set_material(i, random.choice(materials))

        # render the whole pipeline
        data = bproc.renderer.render()
        data.update(camera_poses_res)
        data["original_texture"] = [np.string_("False")] * len(data["colors"])
        del data["normals"]
        del data["depth"]

        # write the data to a .hdf5 container
        bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)
