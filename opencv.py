import cv2
import numpy as np
import matplotlib.pyplot as plt 

def main():
    image = cv2.imread("Road.jpg")
    lane_image = np.copy(image)
    canny_image = find_edges(image)
    # cv2.imshow("Canny", canny_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 10, np.pi/180, 20, np.array([]), minLineLength = 3, maxLineGap = 1)
    averaged_lines = average_slope_intercept(lane_image, lines)


    print(lines)
    line_image = display_lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv2.imshow("Combo", combo_image)
    cv2.imshow("Cropped", line_image)

    # cv2.imshow("Road.jpg", image)
    cv2.waitKey(0)

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        # The reason we have to reshape is because the line is a 2D array and we need a 1D array
        x1, y1 ,x2 ,y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # This is the slope and the y intercept
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept)) # This is the slope and the y intercept
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)   
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    # y = mx + b
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1 ,x2 ,y2])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1 ,x2 ,y2 = line.reshape(4)
            # The reason we have to reshape is because the line is a 2D array and we need a 1D array
            # BGR is the color of the line and the last parameter is the thickness of the line
            cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 2)
            # The reason we have two sets of coordinates is because we have two points
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([
        # The triangle is the region of interest
        [(0, height), (700, height), (500, 300)]
        ])
    # Each of the following parameters are the coordinates of the triangle and the coordinates are in pixels
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def find_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny_image = cv2.Canny(blur, 50, 100) 
    return canny_image

main()
# Last parameter is the threshold and the higher the threshold the more white pixels we get
# The last paramter is Gaussian blur is the kernel size and the higher the kernel size the more blur we get

# The reason it is 5 is because it is an odd number and it is the kernel size which means the size 
# of the matrix it is going to use to blur the image



