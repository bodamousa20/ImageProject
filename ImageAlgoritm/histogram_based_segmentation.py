import numpy as np

# 1. تطبيق العتبة يدويًا (Manual Thresholding)
def manual_thresholding(image, threshold):
    """
    تقسيم الصورة باستخدام عتبة ثابتة يدويا.
    """
    # إنشاء مصفوفة فارغة بنفس حجم الصورة
    result = np.zeros_like(image)
    
    # تطبيق العتبة
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # إذا كانت قيمة البكسل أكبر من العتبة، نقوم بتعيينها إلى 255 (أبيض)
            if image[i, j] > threshold:
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result


# 2. تقسيم الصورة بناءً على القمة في الهيستوجرام (Histogram Peak Thresholding)
def histogram_peak_thresholding(image):
    """
    تقسيم الصورة باستخدام القمة في الهيستوجرام.
    """
    # حساب الهيستوجرام للصورة
    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1
    
    # إيجاد القمة في الهيستوجرام
    peak = np.argmax(hist)
    
    # تطبيق العتبة عند القمة
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= peak:
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result


# 3. تقسيم الصورة بناءً على الوادي في الهيستوجرام (Histogram Valley Thresholding)
def histogram_valley_thresholding(image):
    """
    تقسيم الصورة باستخدام الوادي بين القمم في الهيستوجرام.
    """
    # حساب الهيستوجرام
    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1
    
    # إيجاد الوادي بين القمم
    valleys = np.diff(hist)
    valley_position = np.argmin(valleys)
    
    # تطبيق العتبة عند موقع الوادي
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= valley_position:
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result


# 4. تقسيم الصورة باستخدام العتبة التكيفية (Adaptive Histogram Thresholding)
def adaptive_histogram_thresholding(image, block_size=11, C=2):
    """
    تقسيم الصورة باستخدام العتبة التكيفية بناءً على الجوار.
    """
    rows, cols = image.shape
    result = np.zeros_like(image)
    
    # إضافة العتبة التكيفية لكل بكسل في الصورة
    for i in range(rows):
        for j in range(cols):
            # تحديد الحدود المربعة
            min_i = max(i - block_size // 2, 0)
            max_i = min(i + block_size // 2, rows - 1)
            min_j = max(j - block_size // 2, 0)
            max_j = min(j + block_size // 2, cols - 1)
            
            # حساب المتوسط في الجوار
            local_block = image[min_i:max_i+1, min_j:max_j+1]
            local_mean = np.mean(local_block)
            
            # تطبيق العتبة التكيفية
            if image[i, j] > local_mean - C:
                result[i, j] = 255
            else:
                result[i, j] = 0
    
    return result
