# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np

def linecrop(path):

    image = cv2.imread(path)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Use canny edge detection
    edges = cv2.Canny(gray,50,150,apertureSize=3)
    lines_list =[]
    lines = cv2.HoughLinesP(
        edges, # Input edge image
        1, # Distance resolution in pixels
        np.pi/180, # Angle resolution in radians
        threshold=100, # Min number of votes for valid line
        minLineLength=50, # Min allowed length of line
        maxLineGap=10 # Max allowed gap between line for joining them
    )

    # Iterate over points
    maxx2 = 0
    maxy2 = 0
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        # Draw the lines joing the points
        # On the original image
        # cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
        # Maintain a simples lookup list for points
        lines_list.append([(x1,y1),(x2,y2)])
        if x2 > maxx2:
            maxx2 = x2

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        if y2 > maxy2:
            maxy2 = y2
    image = image[0:maxy2, 0:maxx2]
    # Save the result image
    cv2.imwrite('detectedLines.png',image)
    return maxy2, maxx2

