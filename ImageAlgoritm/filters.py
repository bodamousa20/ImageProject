import numpy as np
from PIL import Image

# تطبيق فلتر High-pass
def high_pass_filter(image):
    """
    تطبيق فلتر High-pass على الصورة باستخدام الدالة اليدوية.
    """
    img_array = np.array(image)
    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]])
    filtered_img = apply_filter_manual(img_array, kernel)
    return Image.fromarray(filtered_img)

# تطبيق فلتر Low-pass
def low_pass_filter(image):
    """
    تطبيق فلتر Low-pass على الصورة باستخدام الدالة اليدوية.
    """
    img_array = np.array(image)
    kernel = np.array([[1/6, 1/3, 1/6],
                       [1/3, 1/2, 1/3],
                       [1/6, 1/3, 1/6]])
    filtered_img = apply_filter_manual(img_array, kernel)
    return Image.fromarray(filtered_img)

# تطبيق فلتر Median يدويًا
def apply_median_filter(image):
    """
    تطبيق فلتر Median على الصورة يدويًا.
    """
    img_array = np.array(image)
    rows, cols = img_array.shape
    median_img = np.zeros_like(img_array)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = img_array[i - 1:i + 2, j - 1:j + 2]
            median_val = np.median(window)
            median_img[i, j] = median_val
    return Image.fromarray(np.uint8(median_img))

# تطبيق الالتفاف يدويًا
def apply_filter_manual(image, filter_mask):
    """
    تطبيق الالتفاف يدويًا باستخدام فلتر محدد.
    """
    h, w = image.shape
    fh, fw = filter_mask.shape
    pad_h, pad_w = fh // 2, fw // 2

    # حشو الصورة بالـ 0
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # مصفوفة الإخراج
    output_image = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + fh, j:j + fw]  # نافذة حول البكسل
            output_image[i, j] = np.sum(region * filter_mask)  # تطبيق الالتفاف

    # تقييد القيم بين 0 و 255
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return output_image



