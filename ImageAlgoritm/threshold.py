from PIL import Image
from ImageAlgoritm.grayscale import convert_to_grayscale

def apply_threshold(image):

    # تحويل الصورة إلى Grayscale إذا لم تكن بالفعل
    if image.mode != "L":
        grayscale_image = convert_to_grayscale(image)
    else:
        grayscale_image = image  # تعيين الصورة مباشرة إذا كانت Grayscale

    img_array = grayscale_image.load()

    rows, cols = grayscale_image.size


    total = 0
    count = rows * cols

    for i in range(rows):
        for j in range(cols):
            total += img_array[i, j]

    threshold_value = total // count

    thresholded_array = Image.new("L", grayscale_image.size)
    thresholded_pixels = thresholded_array.load()

    for i in range(rows):
        for j in range(cols):
            if img_array[i, j] > threshold_value:
                thresholded_pixels[i, j] = 255
            else:
                thresholded_pixels[i, j] = 0

    return thresholded_array
