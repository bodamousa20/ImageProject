import numpy as np
from PIL import Image

from ImageAlgoritm.grayscale import convert_to_grayscale


def calculate_histogram(image):
    """
    حساب المدرج التكراري للصورة (مطابق لكود الكتاب).
    
    Args:
        image (PIL.Image): صورة Grayscale أو RGB.
        
    Returns:
        list: قائمة تمثل المدرج التكراري (عدد مرات ظهور كل قيمة رمادية من 0 إلى 255).
    """
    # التأكد من أن الصورة Grayscale
    if image.mode != "L":
        image = convert_to_grayscale(image)

    # تحويل الصورة إلى مصفوفة numpy
    img_array = np.array(image)

    # إنشاء مصفوفة المدرج التكراري
    histogram = [0] * 256  # مصفوفة بحجم 256 لتخزين عدد التكرارات لكل قيمة رمادية

    # حساب التكرارات لكل مستوى رمادي باستخدام حلقات متداخلة
    rows, cols = img_array.shape
    for i in range(rows):
        for j in range(cols):
            pixel_value = img_array[i, j]  # قيمة البكسل عند الموضع (i, j)
            histogram[pixel_value] += 1  # زيادة عدد التكرار للقيمة المناسبة

    return histogram


def equalize_histogram(image):
    """
    تطبيق معادلة المدرج التكراري على الصورة.
    :param image: صورة من نوع PIL Image.
    :return: الصورة الناتجة بعد معادلة المدرج التكراري.
    """
    # تحويل الصورة إلى Grayscale إذا كانت ملونة
    if image.mode != "L":
        image = convert_to_grayscale(image)

    img_array = np.array(image)
    histogram = calculate_histogram(image)

    # حساب CDF (Cumulative Distribution Function)
    cdf = np.cumsum(histogram)
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)

    # تطبيق معادلة المدرج التكراري
    equalized_img_array = cdf_normalized[img_array]
    equalized_img = Image.fromarray(equalized_img_array)

    return equalized_img
