import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from svr.tsdf_renderer.engine.WindowManager import WindowManager
from svr.all_in_one_client import AllInOneClient


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Connect to the all in one server and predict for an image a 3D scene")
    parser.add_argument("--server_ip", help="Name of the node, where the server is running on", default=None)
    args = parser.parse_args()

    all_in_one_client = AllInOneClient(args.server_ip)

    wm = WindowManager((1920, 1024), 'Please wait until the loading is done!')
    wm.init()
    wm.set_all_in_one_client(all_in_one_client)
    print("Done loading!")
    print("##########################")
    help = "Overview over control:\n\tt - changes the texture between image, classification, and the normals of " \
           "the surfaces\n\tu - switching between projected and unprojected view\n\to - brings the camera back " \
           "to the origin\n\twasd and mouse - for game like movement in the scene."
    print(help)
    print("##########################")
    wm.run_window()

