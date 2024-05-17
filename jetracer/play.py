from __future__ import annotations
from pathlib import Path
from typing import Sequence

import argparse
import numpy as np
import os
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
import cv2, math, time, sys
import yaml

pygame.init()
pygame.joystick.init()

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

car = NvidiaRacecar()
joystick = pygame.joystick.Joystick(0)
joystick.init()

def settings():
    with open('config.yaml', encoding='UTF-8') as config:
        cfg = yaml.load(config, Loader=yaml.FullLoader)
    global WIDTH, HEIGHT, ROI_START_HEIGHT, ROI_HEIGHT, CANNY_LOW, CANNY_HIGH, HOUGH_ABS_SLOPE_RANGE, HOUGH_THRESHOLD, HOUGH_LENGTH, HOUGH_GAP, CANNY_LOW, CANNY_HIGH, DEBUG
    global before_left, before_right
    WIDTH= cfg['IMAGE']['WIDTH']
    HEIGHT = cfg['IMAGE']['HEIGHT']
    ROI_START_HEIGHT = cfg['IMAGE']['ROI_START_HEIGHT']
    ROI_HEIGHT = cfg['IMAGE']['ROI_HEIGHT']
    CANNY_LOW = cfg['CANNY']['LOW_THRESHOLD']
    CANNY_HIGH = cfg['CANNY']['HIGH_THRESHOLD']
    HOUGH_ABS_SLOPE_RANGE = cfg['HOUGH']['ABS_SLOPE_RANGE']
    HOUGH_THRESHOLD = cfg['HOUGH']['THRESHOLD']
    HOUGH_LENGTH = cfg['HOUGH']['MIN_LINE_LENGTH']
    HOUGH_GAP = cfg['HOUGH']['MIN_LINE_GAP']
    CANNY_LOW = cfg['CANNY']['LOW_THRESHOLD']
    CANNY_HIGH = cfg['CANNY']['HIGH_THRESHOLD']
    DEBUG = cfg['DEBUG']

    before_left = 0
    before_right = WIDTH
    
## perception
class Camera:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        width: int = 1920,
        height: int = 1080,
        _width: int = 960,
        _height: int = 540,
        frame_rate: int = 30,
        flip_method: int = 0,
        window_title: str = "Camera",
        save_path: str = "record",
        stream: bool = False,
        save: bool = False,
        log: bool = True,
        capture: bool = False,
        inference: bool = False
    ) -> None:
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self._width = _width
        self._height = _height
        self.frame_rate = frame_rate
        self.flip_method = flip_method
        self.window_title = window_title
        self.save_path = Path(save_path)
        self.stream = stream
        self.save = save
        self.log = log
        self.model = None
        self.capture = capture
        self.inference = inference

        # Check if OpenCV is built with GStreamer support
        # print(cv2.getBuildInformation())

        if isinstance(sensor_id, int):
            self.sensor_id = [sensor_id]
        elif isinstance(sensor_id, Sequence) and len(sensor_id) > 1:
            raise NotImplementedError("Multiple cameras are not supported yet")

        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        self.cap = [cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=id, flip_method=0), \
                    cv2.CAP_GSTREAMER) for id in self.sensor_id]


    def gstreamer_pipeline(self, sensor_id: int, flip_method: int) -> str:
        """
        Return a GStreamer pipeline for capturing from the CSI camera
        """
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                self.width,
                self.height,
                self.frame_rate,
                flip_method,
                self._width,
                self._height,
            )
        )
        
    def run(self, model_traffic=None, model_center=None) -> None:

        if self.cap[0].isOpened():
                   
            _, frame = self.cap[0].read()
            
            lpos, rpos = process_image(frame)
            mid = (lpos + rpos) / 2
            
        height, width, _ = frame.shape
        mid_point = int(mid)
        cv2.line(frame, (mid_point, 0), (mid_point, height), (0, 255, 0), 2)  # 녹색 선
        
        # 결과 이미지 표시
        cv2.imshow("Frame with Mid Line", frame)
        
             
        return mid


    @property
    def frame(self) -> np.ndarray:
        if self.cap[0].isOpened():
            return self.cap[0].read()[1]
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

def process_image(frame):
    global WIDTH, ROI_HEIGHT, ROI_START_HEIGHT, CANNY_LOW, CANNY_HIGH, DEBUG
    global before_left, before_right
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_yellow = (20, 100, 100)
    # upper_yellow = (30, 255, 255)
    # hsv_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # hsv_image = cv2.bitwise_and(frame, frame, mask=hsv_mask)
    # cv2.imshow("hsv", hsv_image)
    kernel_size = 7
    # blur_hsv = cv2.GaussianBlur(hsv_image,(kernel_size, kernel_size), 0)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    # edge_img = cv2.Canny(np.uint8(blur_hsv), CANNY_LOW, CANNY_HIGH)
    edge_gray = cv2.Canny(np.uint8(blur_gray), CANNY_LOW, CANNY_HIGH)

    roi = edge_gray[ROI_START_HEIGHT-ROI_HEIGHT:ROI_START_HEIGHT+ROI_HEIGHT, 0:WIDTH]
    #roi_src = frame[ROI_START_HEIGHT-ROI_HEIGHT:ROI_START_HEIGHT+ROI_HEIGHT, 0:WIDTH]
    all_lines = cv2.HoughLinesP(roi, 10, math.pi/180, HOUGH_THRESHOLD, HOUGH_LENGTH, HOUGH_GAP)
    #print(all_lines)
    if DEBUG:
        cv2.imshow("grey", edge_gray)
        # cv2.imshow("edge", edge_img)
        cv2.imshow("roi", roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
           
    if all_lines is None:
        return before_left, before_right

    # draw_lines(all_lines, roi_src)
    # elif len(all_lines) > 20 :
    #     return before_left, before_right
    #cv2.imshow("src", roi_src)
    left_lines, right_lines = divide_left_right(all_lines)
    #draw_lines(left_lines, roi_src)
    #cv2.imshow("left", roi_src)
    #print(left_lines, right_lines)

    lpos = getLinePos(left_lines, left=True)
    rpos = getLinePos(right_lines, right=True)
    return lpos, rpos

def getLinePos(lines, left=False, right=False):
    global WIDTH, HEIGHT
    global before_left, before_right
    global ROI_HEIGHT, ROI_START_HEIGHT

    m, b = get_line_params(lines)
    if (abs(m) <= sys.float_info.epsilon and abs(b) <= sys.float_info.epsilon):
        if left:
            return before_left
        elif right:
            return before_right
    y = ROI_HEIGHT

    return round((y-b) / m)

def get_line_params(lines):
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0
    size = len(lines)
    if size == 0:
        return 0, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_sum += x1 + x2
        y_sum += y1 + y2
        if x2 - x1 == 0:
            m_sum += 0
        else:
            m_sum += float(y2-y1)/float(x2-x1)

    x_avg = float(x_sum) / float(size * 2)
    y_avg = float(y_sum) / float(size * 2)

    m = m_sum / size
    b = y_avg - m * x_avg
    return m, b

def divide_left_right(lines):            
    global WIDTH
    global before_left, before_right
    slopes = []
    new_lines = []
    for line in lines:
        #print(line)
        x1, y1, x2, y2 = line[0]
        if x2-x1 == 0:
            slope = 0
        else:
            slope = float(y2-y1)/float(x2-x1)
       
        slopes.append(slope)
        new_lines.append(line[0])
    left_lines = []
    right_lines = []

    for i in range(len(slopes)):
        Line = new_lines[i]
        slope = slopes[i]

        x1, y1, x2, y2 = Line
        if (slope < 0 and abs(slope) > 0.5 and 0 < x1 < WIDTH * 0.75):
            left_lines.append([Line.tolist()])
            before_left = (x1 + x2) / 2
        elif (slope > 0 and abs(slope) > 0.5 and WIDTH * 0.25 < x1 < WIDTH):
            right_lines.append([Line.tolist()])
            before_right = (x1 + x2) / 2
    return left_lines, right_lines

def draw_lines(all_lines, frame):
    for line in all_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1),(x2,y2), color=(0,0,255), thickness=2)

if __name__ == '__main__':
    settings()
    args = argparse.ArgumentParser()
    args.add_argument('--sensor_id',
        type = int,
        default = 0,
        help = 'Camera ID')
    args.add_argument('--window_title',
        type = str,
        default = 'Camera',
        help = 'OpenCV window title')
    args.add_argument('--save_path',
        type = str,
        default = 'record',
        help = 'Image save path')
    args.add_argument('--save',
        action = 'store_true',
        help = 'Save frames to save_path')
    args.add_argument('--stream',
        action = 'store_true',
        help = 'Launch OpenCV window and show livestream')
    args.add_argument('--log',
        action = 'store_true',
        help = 'Print current FPS')
    args.add_argument('--capture',
        action = 'store_true')
    args.add_argument('--inference',
        action = 'store_true')
    args = args.parse_args()
    
cam = Camera(
    sensor_id = args.sensor_id,
    window_title = args.window_title,
    save_path = args.save_path,
    save = args.save,
    stream = args.stream,
    log = args.log,
    capture = args.capture,
    inference = args.inference)
        
## control
running = True

# PID control config
kp = 0.1
ki = 0.00001
kd = 0.0001

prev_error = 0.0
integral = 0.0

car.steering_offset = -0.07
steering_range = (-1.1, 1.1)
throttle_range = (0.17, 0.22)
drive = False
prev_cmd = 0.0

while running:
    pygame.event.pump()
    
    if drive == False:
        mode = input("Press Enter : ")
        drive = True
        print("Drive!")
        
    x_sen = cam.run()
    print (f"x_sen: {x_sen}")
    
    # PID control
    error = x_sen - 500
    integral += error
    derivative = error - prev_error
    
    steering_cmd = kp * error + ki * integral + kd * derivative

    car.steering = max(steering_range[0], min(steering_range[1], steering_cmd))
    print(steering_cmd)
    
    if drive:
        throttle = throttle_range[0] + (1 - abs(car.steering)/1.1) * (throttle_range[1] - throttle_range[0])
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        
    else:
        car.throttle = 0.0

    prev_cmd = car.throttle
    # print(car.throttle)

    if joystick.get_button(11): # start button
        running = False

    prev_error = error

