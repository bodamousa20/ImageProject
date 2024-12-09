import numpy as np
from PIL import Image
from ImageAlgoritm.grayscale import convert_to_grayscale

# تطبيق Halftone بسيط باستخدام Threshold
def apply_simple_halftone(image):
    """
    تطبيق Halftone بسيط (Threshold) على الصورة.
    
    Args:
        image (PIL.Image): الصورة المدخلة (RGB).
        
    Returns:
        PIL.Image: الصورة بعد تطبيق الـ Halftone البسيط.
    """
    grayscale_image = convert_to_grayscale(image)
    
    # تطبيق العتبة باستخدام الدالة apply_threshold من utils.threshold
    thresholded_image = apply_threshold(grayscale_image)
    
    return thresholded_image


# def apply_advanced_halftone(image, threshold=128, one=255, zero=0):
#     """
#     تطبيق Halftone متقدم باستخدام توزيع الخطأ المخصص في الكتاب.
#     Args:
#         image (PIL.Image): الصورة المدخلة (RGB أو Grayscale).
#         threshold (int): قيمة العتبة.
#         one (int): قيمة البكسل إذا كانت النتيجة أعلى من العتبة.
#         zero (int): قيمة البكسل إذا كانت النتيجة أقل من أو تساوي العتبة.
#     Returns:
#         PIL.Image: الصورة بعد تطبيق Halftone.
#     """
#     # التأكد من تحويل الصورة إلى Grayscale
#     if len(np.array(image).shape) == 3:  # إذا كانت الصورة تحتوي على 3 أبعاد (RGB)
#         image = convert_to_grayscale(image)
    
#     # تحويل الصورة إلى numpy array
#     img_array = np.array(image, dtype=np.float32)
#     rows, cols = img_array.shape  # الآن الصورة يجب أن تحتوي فقط على بُعدين
    
#     # إنشاء مصفوفتين لتخزين الأخطاء الموزعة
#     ep = np.zeros((rows, cols), dtype=np.float32)  # مجموع الأخطاء الموزعة
#     eg = np.zeros((rows, cols), dtype=np.float32)  # الأخطاء الناتجة
    
#     # مصفوفة توزيع الأخطاء (كما في الكتاب)
#     c = np.array([
#         [0.0, 0.2, 0.0],
#         [0.6, 0.1, 0.1]
#     ])
    
#     # مصفوفة الصورة الناتجة
#     out_image = np.zeros((rows, cols), dtype=np.uint8)
    
#     # تطبيق توزيع الأخطاء
#     for m in range(rows):
#         for n in range(cols):
#             sum_p = 0.0
#             for i in range(2):
#                 for j in range(3):
#                     xx = m - i + 1
#                     yy = n - j + 1
#                     if xx < 0:
#                         xx = 0
#                     if xx >= rows:
#                         xx = rows - 1
#                     if yy < 0:
#                         yy = 0
#                     if yy >= cols:
#                         yy = cols - 1
#                     sum_p += c[i, j] * eg[xx, yy]
            
#             ep[m, n] = sum_p
#             t = img_array[m, n] + ep[m, n]
            
#             # تعيين البكسل إلى one أو zero بناءً على العتبة
#             if t > threshold:
#                 eg[m, n] = t - threshold * 2
#                 out_image[m, n] = one
#             else:
#                 eg[m, n] = t
#                 out_image[m, n] = zero
    
#     # تحويل الصورة الناتجة إلى PIL.Image
#     halftoned_image = Image.fromarray(out_image)
#     return halftoned_image


# تطبيق Halftone متقدم (Error Diffusion) باستخدام Floyd-Steinberg
def apply_advanced_halftone(image):
    """
    تطبيق Halftone متقدم (Error Diffusion) باستخدام طريقة Floyd-Steinberg.
    
    Args:
        image (PIL.Image): الصورة المدخلة (RGB).
        
    Returns:
        PIL.Image: الصورة بعد تطبيق الـ Halftone المتقدم.
    """
    # تحويل الصورة إلى تدرجات الرمادي باستخدام دالة convert_to_grayscale من utils.grayscale
    grayscale_image = convert_to_grayscale(image)
    
    # تحويل الصورة إلى مصفوفة
    img_array = np.array(grayscale_image)
    
    # الحصول على الأبعاد
    height, width = img_array.shape
    
    # تنفيذ عملية Error Diffusion باستخدام Floyd-Steinberg
    for y in range(height):
        for x in range(1, width - 1):  # تفادي الحواف
            old_pixel = img_array[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            img_array[y, x] = new_pixel
            error = old_pixel - new_pixel
            
            # توزيع الخطأ على الجيران (التوزيع Floyd-Steinberg)
            img_array[y, x + 1] += error * 7 / 16
            if y + 1 < height:
                img_array[y + 1, x - 1] += error * 3 / 16
                img_array[y + 1, x] += error * 5 / 16
                if x + 1 < width:
                    img_array[y + 1, x + 1] += error * 1 / 16
    
    # تحويل المصفوفة مرة أخرى إلى صورة
    halftoned_image = Image.fromarray(np.uint8(img_array))
    
    return halftoned_image


def apply_threshold(image):
    """
    تطبيق العتبة يدويًا على الصورة بناءً على متوسط قيم البيكسلات.

    Args:
        image (PIL.Image): الصورة المدخلة (RGB).

    Returns:
        PIL.Image: الصورة بعد تطبيق العتبة.
    """
    # تحويل الصورة إلى Grayscale إذا لم تكن بالفعل
    if image.mode != "L":
        grayscale_image = convert_to_grayscale(image)
    else:
        grayscale_image = image  # تعيين الصورة مباشرة إذا كانت Grayscale

    img_array = grayscale_image.load()

    # الحصول على أبعاد الصورة
    rows, cols = grayscale_image.size

    # حساب متوسط قيم البيكسلات لتحديد العتبة
    total = 0
    count = rows * cols

    for i in range(rows):
        for j in range(cols):
            total += img_array[i, j]

    threshold_value = total // count

    # تطبيق العتبة يدويًا
    thresholded_array = Image.new("L", grayscale_image.size)
    thresholded_pixels = thresholded_array.load()

    for i in range(rows):
        for j in range(cols):
            if img_array[i, j] > threshold_value:
                thresholded_pixels[i, j] = 255
            else:
                thresholded_pixels[i, j] = 0

    return thresholded_array
