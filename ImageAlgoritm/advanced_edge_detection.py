import numpy as np


def homogeneity_operator(image, window_size=3):
    height, width = image.shape
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for i in range(pad_size, height - pad_size):
        for j in range(pad_size, width - pad_size):
            window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            mean_value = np.mean(window)
            result[i, j] = np.abs(image[i, j] - mean_value)

    return result


def difference_operator(image):
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            diff = abs(int(image[i, j]) - int(image[i, j + 1])) + abs(int(image[i, j]) - int(image[i + 1, j]))
            result[i, j] = diff
    return result


import math


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    sum_kernel = 0

    for x in range(-center, center + 1):
        for y in range(-center, center + 1):
            kernel[x + center, y + center] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(
                -(x ** 2 + y ** 2) / (2 * sigma ** 2))
            sum_kernel += kernel[x + center, y + center]

    # Normalizing the kernel so that sum is 1
    kernel /= sum_kernel
    return kernel


import numpy as np


def apply_kernel(image, kernel):

    # Ensure image is a NumPy array
    image = np.array(image)

    # Get the dimensions of the image
    height, width = image.shape
    size = kernel.shape[0]
    pad_size = size // 2

    # Pad the image with zeros
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)

    # Initialize the result array
    result = np.zeros_like(image)

    # Apply the kernel
    for i in range(pad_size, height - pad_size):
        for j in range(pad_size, width - pad_size):
            region = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            result[i, j] = np.sum(region * kernel)

    return result


def difference_of_gaussians(image, sigma1=1.0, sigma2=2.0):
    kernel1 = gaussian_kernel(7, sigma1)  # 7x7 kernel for sigma1
    kernel2 = gaussian_kernel(9, sigma2)  # 9x9 kernel for sigma2

    blurred1 = apply_kernel(image, kernel1)
    blurred2 = apply_kernel(image, kernel2)

    return blurred1 - blurred2


# دالة Contrast-Based Edge Detection التي تعتمد على difference_operator
def contrast_based_edge_detection(image, threshold=20, high=250):
    # تطبيق دالة difference_operator لاستخراج الحواف
    edge_result = difference_operator(image)

    # تطبيع القيم الناتجة لتكون ضمن النطاق [0, 255]
    edge_result = np.clip(edge_result, 0, 255)

    # تطبيق threshold لتحسين الصورة
    result = np.zeros_like(edge_result)
    result[edge_result >= threshold] = high
    result[edge_result < threshold] = 0

    return result


def variance_operator(image, window_size=3):
    height, width = image.shape
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for i in range(pad_size, height - pad_size):
        for j in range(pad_size, width - pad_size):
            window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            variance = np.var(window)
            result[i, j] = variance

    return result


def range_operator(image, window_size=3):
    height, width = image.shape
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for i in range(pad_size, height - pad_size):
        for j in range(pad_size, width - pad_size):
            window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            range_value = np.max(window) - np.min(window)
            result[i, j] = range_value

    return result
