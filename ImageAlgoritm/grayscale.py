from PIL import Image
import numpy as np

def convert_to_grayscale(image):
    """
    Converts an image to grayscale manually.

    Args:
        image: The input image as a PIL.Image or NumPy array.

    Returns:
        np.ndarray: The image converted to grayscale as a NumPy array.
    """
    # Convert NumPy array to PIL.Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Ensure the image is in RGB mode before converting
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Get the dimensions of the image
    cols, rows = image.size

    # Create a new grayscale image
    grayscale_image = Image.new("L", (cols, rows))
    grayscale_pixels = grayscale_image.load()

    # Load the pixels of the original image
    img_pixels = image.load()

    # Loop through all pixels and compute the grayscale value
    for i in range(cols):
        for j in range(rows):
            r, g, b = img_pixels[i, j]
            grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)  # Grayscale conversion formula
            grayscale_pixels[i, j] = grayscale_value

    # Return the result as a NumPy array
    return np.array(grayscale_image)
