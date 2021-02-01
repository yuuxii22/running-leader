from cv2 import cv2
import numpy as np
import matplotlib.pylab as plt
import math
import logging
import sys
import datetime
import pandas as pd
from sklearn.cluster import KMeans
from operator import itemgetter


def region_interest(photo,vertices):
    mask = np.zeros_like(photo)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(photo, mask)
    return masked_image

def drawline(photo, lines):
    photo = np.copy(photo)
    line_image = np.zeros((photo.shape),dtype = np.uint8)
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),thickness=2)
    return line_image

def do_canny(photo):
    blur = cv2.GaussianBlur(photo, (5,5), 0)
    return cv2.Canny(blur,100,200)

def cluster(list):
    df = pd.DataFrame(list)
    model = KMeans(n_clusters = 2)
    model.fit(df)
    return model.cluster_centers_


def average_slope_intercept(photo,lines):
    left = []
    right = []
    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameter = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameter[0]
        intercept = parameter[1]
        if slope < -0.5:
            if x1 < left_region_boundary and x2 < left_region_boundary:
                left.append((slope,intercept))
        elif slope > 0.5: 
            if x1 > right_region_boundary and x2 > right_region_boundary:
                right.append((slope,intercept))
    try:
        left_lines = cluster(left)
        right_lines = cluster(right)
        left_line_0 = make_coordinate(left_lines[0])
        left_line_1 = make_coordinate(left_lines[1])
        right_line_0 = make_coordinate(right_lines[0])
        right_line_1 = make_coordinate(right_lines[1])
        lines = np.array([left_line_0,left_line_1,right_line_0,right_line_1])
    except:
        try:
            left_lines = cluster(left)
            right_lines = np.average(right,axis=0)
            left_line_0 = make_coordinate(left_lines[0])
            left_line_1 = make_coordinate(left_lines[1])
            right_average = make_coordinate(right_lines)
            lines = np.array([left_line_0,left_line_1,right_average])
        except:
            try:
                left_lines = np.average(left,axis=0)
                right_lines = cluster(right)
                right_line_0 = make_coordinate(right_lines[0])
                right_line_1 = make_coordinate(right_lines[1])
                left_average = make_coordinate(left_lines)
                lines = np.array([left_average,right_line_0,right_line_1])
            except:
                left_average = np.average(left,axis=0)
                right_average = np.average(right,axis=0)
                left_average_coordinates = make_coordinate(left_average)
                right_average_coordinates = make_coordinate(right_average)
                lines = np.array([left_average_coordinates,right_average_coordinates])
    lines = sorted(lines, key=itemgetter(0))
    return lines

def make_coordinate(line_parameter):
    slope,intercept = line_parameter
    y1 = height
    y2 = int(height/2)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def compute_steering_angle(lane_lines):
    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0]
        _, _, right_x2, _ = lane_lines[1]
    if len(lane_lines) == 3:
        _,_,lx2,_ = lane_lines[0]
        x1,_,x2,_ = lane_lines[1]
        _,_,rx2,_ = lane_lines[2]
        left_x2 = lx2
        right_x2 = rx2
        if x1 < x2:
            left_x2 = x2
        else:
            right_x2 = x2
    else:
        _,_,_,_ = lane_lines[0]
        _,_,left_1x2,_ = lane_lines[1]
        _,_,right_0x2,_ = lane_lines[2]
        _,_,_,_ = lane_lines[3]
        left_x2 = left_1x2
        right_x2 = right_0x2
    camera_mid_offset_percent = 0.00 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
    mid = int(width / 2 * (1 + camera_mid_offset_percent))
    x_offset = (left_x2 + right_x2) / 2 - mid
    y_offset = int(height / 2)*-1
    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel
    return steering_angle

def display_heading_line(frame, steering_angle, line_color=(126, 126, 126), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 + height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    return cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

def check_hwv(image):
    height = image.shape[0]
    width = image.shape[1]
    vertices = [
        (0,height),
        (0,height*2/3),
        (width*1/3,height*1/3),
        (width*2/3,height*1/3),
        (width,height*2/3),
        (width,height)
    ]
    return np.array([height,width,vertices])

def test_image(photo):
    image = cv2.cvtColor((photo),cv2.COLOR_BGR2RGB)
    canny_image = do_canny(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY))
    crop_img = region_interest(canny_image,np.array([vertices],np.int32))
    lines = cv2.HoughLinesP(crop_img,3,np.pi/180,100,np.array([]),minLineLength=70,maxLineGap=50)
    line_image = drawline(image,average_slope_intercept(photo,lines))
    angle = compute_steering_angle(average_slope_intercept(photo,lines))
    image_with_lines = cv2.addWeighted(photo,0.8,line_image,1,1)
    image_with_direction_line = cv2.addWeighted(image_with_lines,0.8,display_heading_line(image,angle),1,1)
    cv2.putText(image_with_direction_line, "steering angle: " + str(angle), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return cv2.imshow("combine",image_with_direction_line)

if __name__ == '__main__':
    photo = cv2.imread("images_3.png")
    height,width,vertices = check_hwv(photo)
    test_image(photo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()