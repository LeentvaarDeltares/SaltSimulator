from PIL import Image
import os


def jpg_giffer(folder, file):
    image_files = sorted([f for f in os.listdir(folder) if f.endswith(f"{file}.jpg")])
    images = [Image.open(os.path.join(folder, img)) for img in image_files]

    images[0].save(
        f"gif_{file}.gif", save_all=True, append_images=images[1:], duration=100, loop=0
    )

    return


if __name__ == "__main__":
    folder = "C:/Users/leentvaa/SaltSimulator/results/"
    file = "contourfull"
    jpg_giffer(folder, file)
