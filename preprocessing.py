import pytesseract
from ROI_selection import detect_lines, get_ROI
import numpy as np

import cv2
import numpy as np

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_binary(image):
    (thresh, blackAndWhiteImage) = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage

def invert_area(image, x, y, w, h, display=False):
    ones = np.copy(image)
    ones = 1

    image[ y:y+h , x:x+w ] = ones*255 - image[ y:y+h , x:x+w ]

    if (display):
        cv2.imshow("inverted", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image

def detect(cropped_frame, is_number = False):
    if (is_number):
        text = pytesseract.image_to_string(cropped_frame,
                                           config ='-c tessedit_char_whitelist=0123456789 --psm 10 --oem 2')
    else:
        text = pytesseract.image_to_string(cropped_frame, config='--psm 10')

    return text

def draw_text(src, x, y, w, h, text):
    cFrame = np.copy(src)
    cv2.rectangle(cFrame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(cFrame, "text: " + text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
               2, (0, 0, 0), 5, cv2.LINE_AA)

    return cFrame

def erode(img, kernel_size = 5):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    img_erosion = cv2.dilate(img, kernel, iterations=2)
    return img_erosion


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

def clear_img(img_path):
    # Read image using opencv
    img = img_path
    # Extract the file name without the file extension
    # Create a directory for outputs
    # Rescale the image, if needed.
    # Converting to gray scale
    #Removing Shadows
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((1,1), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 27)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)

    #Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)#increases the white region in the image
    img = cv2.erode(img, kernel, iterations=1) #erodes away the boundaries of foreground object

    #Apply blur to smooth out the edges
    #img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #Save the filtered image in the output directory
    return img

def apply_adaptive_threshold(img, method):
    """
    Apply adaptive thresholding, either Gaussian (threshold value is the weighted sum of neighbourhood values where weights are a Gaussian window) or mean (threshold value is the mean of neighbourhood area). Show result.
    """
    img = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_RGB2GRAY,
    )

    if method == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif method == 'mean':
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

    img_adaptive = cv2.adaptiveThreshold(
        src=img,
        maxValue=255,
        adaptiveMethod=adaptive_method,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    return img_adaptive

def apply_laplacian(img):

    img = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_RGB2GRAY,
    )

    img_laplacian = np.uint8(
        np.absolute(
            cv2.Laplacian(
                src=img,
                ddepth=cv2.CV_64F,
            )
        )
    )

    return img_laplacian

def apply_sobel(img, direction):

    img = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_RGB2GRAY,
    )

    if direction == 'h':
        dx, dy = 0, 1

    elif direction == 'v':
        dx, dy = 1, 0

    img_sobel = cv2.Sobel(
        src=img,
        ddepth=cv2.CV_64F,
        dx=dx,
        dy=dy,
        ksize=5,
    )

    return img_sobel

def removeline(image):
    result = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mh = 0
    for c in cnts:
        for m in c[0]:
            if m[1]> mh:
                mh = m[1]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mv = 0
    for c in cnts:
        for m in c[0]:
            if m[0]> mv:
                mv = m  [0]
    # thresh = thresh[0:mh, 0:mv]

    for i in range(3):

        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(result, [c], -1, (255,255,255),  3)

        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255,255,255), 3)
    result = result[0:mh, 0:mv]
    return result
