import os
from pathlib import Path

import gdown


def download_all_models():
    link_to_file = "https://drive.google.com/uc?id=101Ddy1cRExSdPKbMcyWgJ9pDGddpJa7M"

    current_folder = Path(__file__).parent.absolute()

    output = current_folder / "trained_models.zip"

    if not output.exists():
        gdown.download(link_to_file, output.name, quiet=False)
    else:
        print("The file is already downloaded!")

    if output.exists():
        print(f"Unzipping the file {output.name}")

        os.chdir(output.parent)
        os.system(f"unzip {output} > /dev/null")

        os.remove(output)
        print("Downloaded the models for the SemanticSingleViewReconstruction, the neural network "
              "for the compression and the UNet for the normal generation.")
    else:
        raise FileNotFoundError("The output does not exist!")

if __name__ == "__main__":
    download_all_models()
