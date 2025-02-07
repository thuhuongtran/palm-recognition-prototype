
# Load the image

from PIL import Image

# Open an image
image = Image.open("../dataset/archive/session1/00001.tiff")

# Check the mode
print("Image mode:", image.mode)  # Example output: "RGB" or "RGBA"
