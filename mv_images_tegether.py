import shutil
import os
import argparse

from utils import count_images_in_dir_recursive


def main(project_dir, output_dir):
    # Count the number of images in the project directory
    num_images, images_path = count_images_in_dir_recursive(project_dir)
    print(f"Number of images in {project_dir}: {num_images}")

    for image_path in images_path:
        image_dir = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        new_name = image_dir + "_" + os.path.basename(image_path)
        new_path = os.path.join(output_dir, new_name)
        shutil.move(image_path, new_path)
        print(f"Moved {image_path} to {new_path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Move images from project directory to output directory.")
    parser.add_argument("--project_dir", "-p", type=str, help="Path to the project directory containing images.")
    parser.add_argument("--output_dir", "-o", type=str, help="Path to the output directory where images will be moved.")

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(args.project_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {output_dir}")

    main(args.project_dir, output_dir)