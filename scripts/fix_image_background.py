from PIL import Image
import os


def make_background_transparent(image_path, save_path):
    # Open the image
    img = Image.open(image_path)
    # Ensure the image has an alpha channel
    img = img.convert("RGBA")

    # Load the data of the image
    datas = img.getdata()

    # Create a new data list
    newData = []
    for item in datas:
        # Change all white (also consider almost white) pixels to transparent
        if item[0] == 255 and item[1] == 255 and item[2] == 255:  # You can adjust these values
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    # Update the image data
    img.putdata(newData)
    # Save the image
    img.save(save_path, "PNG")


directory = "/scr/OmniGibson/docs/assets/object_states/"
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        image_path = os.path.join(directory, filename)
        save_path = os.path.join(directory, filename)
        make_background_transparent(image_path, save_path)
        print(f"Processed {filename}")
