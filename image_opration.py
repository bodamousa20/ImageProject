import numpy as np

# 1. إضافة الصورة ونسختها يدويًا (للصور 2D)
def add_images(image,image2):
    rows, cols = image.shape  # نحصل على حجم الصورة 2D (صفوف وأعمدة)
    result = np.zeros_like(image, dtype=np.uint8)  # مصفوفة فارغة بنفس حجم الصورة

    # إنشاء نسخة من الصورة
    image_copy = image2

    # إضافة الصورة ونسختها
    for i in range(rows):
        for j in range(cols):
            result[i, j] = image[i, j] + image_copy[i, j]

    # التأكد أن القيم ضمن النطاق [0, 255]
    result = np.clip(result, 0, 255)
    return result


# 2. طرح الصورة ونسختها يدويًا (للصور 2D)
def subtract_images(image,image2):
    rows, cols = image.shape  # نحصل على حجم الصورة 2D (صفوف وأعمدة)
    result = np.zeros_like(image, dtype=np.uint8)  # مصفوفة فارغة بنفس حجم الصورة

    # إنشاء نسخة من الصورة
    image_copy = image2

    # طرح الصورة ونسختها
    for i in range(rows):
        for j in range(cols):
            result[i, j] = image[i, j] - image_copy[i, j]
            # التأكد أن القيم ضمن النطاق [0, 255]
    result = np.clip(result, 0, 255)
    return result

# 3. عكس الصورة يدويًا (للصور 2D)
def invert_image(image):
    rows, cols = image.shape  # نحصل على حجم الصورة 2D (صفوف وأعمدة)
    result = np.zeros_like(image, dtype=np.uint8)  # مصفوفة فارغة بنفس حجم الصورة

    # عكس قيم البكسلات
    for i in range(rows):
        for j in range(cols):
            result[i, j] = 255 - image[i, j]
    return result
