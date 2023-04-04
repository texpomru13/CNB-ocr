import cv2 as cv
import numpy as np
from pprint import pprint
filename = 'Python-Custom-Digit-Recognition/testdgt.png'
img = cv.imread(cv.samples.findFile(filename))
cImage = np.copy(img) #image to draw lines

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 250, 250)
cv.imshow("canny", canny)
cv.waitKey(0)
cv.destroyWindow("canny")

rho = 3
theta = np.pi/180
threshold = 20
minLinLength = 80
maxLineGap = 10
linesP = cv.HoughLinesP(canny, rho , theta, threshold, None, minLinLength, maxLineGap)


def is_vertical(line):
    return line[0]==line[2]
def is_horizontal(line):
    return line[1]==line[3]
horizontal_lines = []
vertical_lines = []

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        if (is_vertical(l)):
            vertical_lines.append(l)

        elif (is_horizontal(l)):
            horizontal_lines.append(l)
for i, line in enumerate(horizontal_lines):
    cv.line(cImage, (line[0], line[1]), (line[2], line[3]), (0,255,0), 3, cv.LINE_AA)

for i, line in enumerate(vertical_lines):
    cv.line(cImage, (line[0], line[1]), (line[2], line[3]), (0,0,255), 3, cv.LINE_AA)


def overlapping_filter(lines, sorting_index):
    filtered_lines = []

    lines = sorted(lines, key=lambda lines: lines[sorting_index])
    separation = 5
    for i in range(len(lines)):
        l_curr = lines[i]
        if(i>0):
            l_prev = lines[i-1]
            if ( (l_curr[sorting_index] - l_prev[sorting_index]) > separation):
                filtered_lines.append(l_curr)
        else:
            filtered_lines.append(l_curr)

    return filtered_lines


horizontal_lines = []
vertical_lines = []

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        if (is_vertical(l)):
            vertical_lines.append(l)

        elif (is_horizontal(l)):
            horizontal_lines.append(l)
    horizontal_lines = overlapping_filter(horizontal_lines, 1)
    vertical_lines = overlapping_filter(vertical_lines, 0)
for i, line in enumerate(horizontal_lines):
    cv.line(cImage, (line[0], line[1]), (line[2], line[3]), (0,255,0), 3, cv.LINE_AA)
    cv.putText(cImage, str(i) + "h", (line[0] + 5, line[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
for i, line in enumerate(vertical_lines):
    cv.line(cImage, (line[0], line[1]), (line[2], line[3]), (0,0,255), 3, cv.LINE_AA)
    cv.putText(cImage, str(i) + "v", (line[0], line[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

cv.imshow("with_line", cImage)
cv.waitKey(0)
cv.destroyWindow("with_line") #close the window


def get_cropped_image(image, x, y, w, h):
    cropped_image = image[ y:y+h , x:x+w ]
    return cropped_image
def get_ROI(image, horizontal, vertical, left_line_index, right_line_index, top_line_index, bottom_line_index, offset=4):
    x1 = vertical[left_line_index][2] + offset
    y1 = horizontal[top_line_index][3] + offset
    x2 = vertical[right_line_index][2] - offset
    y2 = horizontal[bottom_line_index][3] - offset

    w = x2 - x1
    h = y2 - y1

    cropped_image = get_cropped_image(image, x1, y1, w, h)

    return cropped_image, (x1, y1, w, h)

import pytesseract
def draw_text(src, x, y, w, h, text):
    cFrame = np.copy(src)
    cv.rectangle(cFrame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv.putText(cFrame, "text: " + text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv.LINE_AA)

    return cFrame
def detect(cropped_frame, is_number = False):
    if (is_number):
        text = pytesseract.image_to_string(cropped_frame,
                                           config ='-c tessedit_char_whitelist=0123456789 --psm 10 --oem 2 -l rus')
    else:
        text = pytesseract.image_to_string(cropped_frame, lang='rus')

    return text


## set counter for image indexing
counter = 0

## set line index
first_line_index = 1
last_line_index = 2

counter = 0
print("Start detecting text...")
(thresh, bw) = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
resdict = {}
for i in range(first_line_index, last_line_index):
    resdict[i] = {}
    for j in range(3):
        counter += 1

        left_line_index = j
        right_line_index = j+1
        top_line_index = i
        bottom_line_index = i+1

        cropped_image, (x,y,w,h) = get_ROI(bw, horizontal_lines, vertical_lines, left_line_index, right_line_index, top_line_index, bottom_line_index)
        cv.imwrite("cropped_img/{}_{}.jpg".format(i,j), cropped_image)
        text = detect(cropped_image,is_number = True)
        # cv.imwrite("cropped_img/{}_{}.jpg".format(i,j), cropped_image)
        # cv.im("cropped_image", cropped_image)
        # cv.waitKey(0)
        # cv.destroyWindow("cropped_image")
        resdict[i][j] = text
        image_with_text = draw_text(img, x, y, w, h, text)

pprint(resdict)

# import cv2
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1]//100))
# eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
#
# vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
# vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
#
# thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
# contours, hierarchy = cv.findContours(vertical_horizontal_lines, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
# boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
# (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x:x[1][1]))
#
# boxes = []
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     if (w<1000 and h<500):
#         image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#         boxes.append([x,y,w,h])
#
# rows=[]
# columns=[]
# heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
# mean = np.mean(heights)
# print(mean)
# columns.append(boxes[0])
# previous=boxes[0]
# for i in range(1,len(boxes)):
#     if(boxes[i][1]<=previous[1]+mean/2):
#         columns.append(boxes[i])
#         previous=boxes[i]
#         if(i==len(boxes)-1):
#             rows.append(columns)
#     else:
#         rows.append(columns)
#         columns=[]
#         previous = boxes[i]
#         columns.append(boxes[i])
# print("Rows")
# for row in rows:
#     print(row)