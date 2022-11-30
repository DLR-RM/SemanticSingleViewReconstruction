from typing import Optional
from pathlib import Path

from PIL import Image
import numpy as np


def map_image(input_image: np.ndarray, current_focal_length: Optional[float] = None,
              opening_angle_in_horz_deg: Optional[float] = None, opening_angle_in_vert_deg: Optional[float] = None,
              current_focal_value: Optional[float] = None):
    """
    Uses the input image and the current focal length for a sensor size of 35.0 mm. For example the main camera
    of an iPhone 14 Pro, has a focal distance of 24.0 mm, while iPhone 12 Pro has 26.0 mm.

    One can also give the opening angle of the camera in degree in horizontal and vertical direction, for example the
    Xtion Pro uses 58.0 degree as horizontal and 45.0 degree as vertical opening angle.

    Lastly, one can also provide the focal value this value can be extracted from the K matrix of your camera.
    The value should be in the same range as your resolution for an image with 640x480, the value might be around 600.
    """
    if current_focal_length is not None:
        sensor_width = 35.0
        sensor_height = sensor_width * input_image.shape[0] / input_image.shape[1]
        field_of_view_x = 2 * np.arctan(sensor_width / (2.0 * current_focal_length))
        field_of_view_y = 2 * np.arctan(sensor_height / (2.0 * current_focal_length))
    elif opening_angle_in_horz_deg is not None and opening_angle_in_vert_deg is not None:
        field_of_view_x = np.deg2rad(opening_angle_in_horz_deg)
        field_of_view_y = np.deg2rad(opening_angle_in_vert_deg)
    elif current_focal_value is not None:
        field_of_view_x = np.deg2rad(2 * np.arctan(input_image.shape[1] / (2 * current_focal_value)))
        field_of_view_y = np.deg2rad(2 * np.arctan(input_image.shape[0] / (2 * current_focal_value)))
    else:
        raise ValueError("One value to calculate the opening angle has to be given!")

    image_fov = np.array([field_of_view_x, field_of_view_y]) * 0.5
    desired_fov = np.array([0.5, 0.388863])
    image_fov_in_tanges = np.tan(image_fov)
    desired_fov_in_tanges = np.tan(desired_fov)
    # flip the axis to be x and y
    new_cropped_size = np.array(
        [input_image.shape[1], input_image.shape[0]]) / image_fov_in_tanges * desired_fov_in_tanges
    new_cropped_size = np.round(new_cropped_size).astype(int)
    # flip the axis to be y and x again as in input image
    new_cropped_size = np.array([new_cropped_size[1], new_cropped_size[0]])

    offset = (input_image.shape[:2] - new_cropped_size) // 2
    cropped_img = input_image[offset[0]: offset[0] + new_cropped_size[0], offset[1]: offset[1] + new_cropped_size[1]]
    return cropped_img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Convert a general image to the correct opening angle of our method. "
                                     "Our method only supports an opening angle of 57.30 x 44.56 degree.")
    parser.add_argument("input_file_path", help="Path to the input image file.")
    parser.add_argument("--focal_length_full_format", help="The focal distance in relation to full format sensor. "
                                                           "For example the main camera of an iPhone 14 Pro, has "
                                                           "a focal distance of 24.0 mm, while iPhone 12 Pro has "
                                                           "26.0 mm. You can google the focal length of your phone.",
                        default=None, type=float)
    parser.add_argument("--opening_angle_in_horizontal_deg", help="You can also set the opening angle of your camera "
                                                                  "in degree. The value must be bigger than 57.30!",
                        default=None, type=float)
    parser.add_argument("--opening_angle_in_vertical_deg", help="You can also set the opening angle of your camera "
                                                                "in degree. The value must be bigger than 44.56!",
                        default=None, type=float)
    parser.add_argument("--focal_value", help="The focal value can be extracted from the K matrix of your camera. "
                                              "You can take the average of K[0,0] and K[1,1]. "
                                              "The value should be in the same range as your resolution for an image "
                                              "with 640x480, the value might be around 600.", default=None, type=float)
    args = parser.parse_args()

    img_path = Path(args.input_file_path)
    if not img_path.exists():
        raise FileNotFoundError("The input image does not exist!")

    img = np.asarray(Image.open(str(img_path)))

    mapped_img = map_image(img, current_focal_length=args.focal_length_full_format,
                           opening_angle_in_horz_deg=args.opening_angle_in_horizontal_deg,
                           opening_angle_in_vert_deg=args.opening_angle_in_vertical_deg,
                           current_focal_value=args.focal_value)

    mapped_img = Image.fromarray(mapped_img)
    mapped_img = mapped_img.resize((512, 512))

    mapped_img.save(img_path.parent / (img_path.with_suffix("").name + "_mapped.jpg"))


