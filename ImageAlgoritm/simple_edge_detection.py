import numpy as np


# Sobel Filter (كشف الحواف)
def sobel_filter(image):
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=np.float32)

    grad_x = apply_convolution(image, sobel_x)
    grad_y = apply_convolution(image, sobel_y)

    # مجموع التدرجات لتحديد الحواف
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.clip(grad_magnitude, 0, 255).astype(np.uint8)


# Prewitt Filter (كشف الحواف)
def prewitt_filter(image):
    prewitt_x = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]], dtype=np.float32)
    prewitt_y = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]], dtype=np.float32)

    grad_x = apply_convolution(image, prewitt_x)
    grad_y = apply_convolution(image, prewitt_y)

    # مجموع التدرجات لتحديد الحواف
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.clip(grad_magnitude, 0, 255).astype(np.uint8)


# تطبيق الالتفاف (Convolution) يدويًا
def apply_kirsch_filter(image, kernel):
    """
    تطبيق مرشح Kirsch على الصورة باستخدام مصفوفة واحدة.
    :param image: الصورة المدخلة كـ numpy array.
    :param kernel: مصفوفة Kirsch kernel.
    :return: الصورة الناتجة بعد الالتفاف.
    """
    kernel = np.array(kernel).reshape(3, 3)
    pad_size = kernel.shape[0] // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            filtered_image[i, j] = np.sum(region * kernel)

    return filtered_image


# دالة Kirsch لتطبيق جميع المرشحات واختيار الحافة الأقوى
def kirsch_operator(image):
    """
    تطبيق مرشح Kirsch لاكتشاف الحواف.
    :param image: الصورة المدخلة كـ numpy array.
    :return: الصورة مع الحواف المكتشفة.
    """
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    ]

    # تطبيق المرشحات وجمع النتائج
    edges = []
    for kernel in kirsch_kernels:
        filtered_image = apply_kirsch_filter(image, kernel)
        edges.append(filtered_image)

    # اختيار الصورة مع أقوى الحواف
    final_edge = np.max(np.array(edges), axis=0)

    # تحسين التباين للصورة النهائية
    final_edge = np.clip(final_edge, 0, 255)  # التأكد من أن القيم بين 0 و 255
    return final_edge.astype(np.uint8)


# تطبيق الالتفاف (Convolution) يدويًا
def apply_convolution(image, kernel):
    pad_size = kernel.shape[0] // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            filtered_image[i, j] = np.sum(region * kernel)

    return filtered_image
