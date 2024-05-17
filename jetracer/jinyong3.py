from __future__ import annotations
from pathlib import Path
from typing import Sequence

import argparse
import cv2

import numpy as np
import os

import pygame
from jetracer.nvidia_racecar import NvidiaRacecar

import time

pygame.init()
pygame.joystick.init()

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

car = NvidiaRacecar()
joystick = pygame.joystick.Joystick(0)
joystick.init()

## perception

class Camera:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        width: int = 1920/4,
        height: int = 1080/4,
        _width: int = 960,
        _height: int = 540,
        frame_rate: int = 10,
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
        
        cls_dict = {0: 'bus', 1: 'crosswalk', 2: 'left', 3: 'right', 4: 'straight'}

        if self.cap[0].isOpened():
                   
            _, frame = self.cap[0].read()
            results = model_traffic.predict(frame)
            tmp_image_path = '/home/ircv3/HYU-2024-Embedded/jetracer/tmp/tmp_.jpg'
            results[0].save(tmp_image_path)
            cls = 'none'
            
            try:
                for i in range(len(results[0].__dict__['boxes'])):
                    cls = cls_dict[int(results[0].__dict__['boxes'].cls.item())]
                    print('clsclscls\t\t\t\t', cls)
            except:
                print(results[0].__dict__['boxes'].cls)
            
            image_ori = PIL.Image.open(tmp_image_path) # frame: ndarray
            width = image_ori.width
            height = image_ori.height

            with torch.no_grad():
                image = preprocess(image_ori)
                output = model_center(image).detach().cpu().numpy()
            x, y = output[0]

            x = (x / 2 + 0.5) * width
            y = (y / 2 + 0.5) * height
            
            return x, cls
    
        else:
            raise ValueError("cam is not opened")

    @property
    def frame(self) -> np.ndarray:
        """
        !!! Important: This method is not efficient for real-time rendering !!!

        [Example Usage]
        ...
        frame = cam.frame # Get the current frame from camera
        cv2.imshow('Camera', frame)
        ...

        """
        if self.cap[0].isOpened():
            return self.cap[0].read()[1]
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

if __name__ == '__main__':
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

from ultralytics import YOLO
model_traffic_path = '/home/ircv3/HYU-2024-Embedded/jetracer/model/yolov8n_traffic_sign_20240510_e32b16.pt'
model_traffic = YOLO(model_traffic_path) 

import torch
import torchvision
import PIL.Image
from cnn.center_dataset import TEST_TRANSFORMS

device = torch.device('cuda')
model_center = torchvision.models.alexnet(num_classes=2, dropout=0.0)
model_center.load_state_dict(torch.load('/home/ircv3/HYU-2024-Embedded/jetracer/model/road_following_model.pth'))
model_center = model_center.to(device)
    
def preprocess(image: PIL.Image):
    device = torch.device('cuda')    
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]
        
## control
running = True

prev_error = 0.0
integral = 0.0

car.steering_offset = -0.07
steering_range = (-1.1, 1.1)

drive = False



import sys
import termios
import tty


time.sleep(0.2)
car.throttle = 0.2
time.sleep(0.2)
car.throttle = 0
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

throttle_origin = 0.1
print("Press 'n' to increase throttle_origin by 0.01, 'm' to decrease throttle_origin by 0.01, and 'd' to stop.")

while True:
    key = get_key()
    if key == 'm':
        throttle_origin += 0.001
        car.throttle = throttle_origin
        print(f"throttle_origin increased to {throttle_origin:.3f}")
    elif key == 'n':
        throttle_origin -= 0.001
        car.throttle = throttle_origin
        print(f"throttle_origin decreased to {throttle_origin:.3f}")
    elif key == 'd':
        car.throttle = 0
        print(f"Exiting loop with throttle_origin = {throttle_origin:.3f}")
        break
    else:
        print("Invalid input, please enter 'n', 'm', or 'd'.")
    print(f"Current throttle_origin = {throttle_origin:.3f}")

# The next steps can be added here
print("Next steps go here.")
print(throttle_origin)




cross_timer=0
while running:
    pygame.event.pump()
    throttle_zero= throttle_origin

    
    x_sen, cls = cam.run(model_traffic, model_center)

    if cls == 'bus':
        time.sleep(0.1)
        car.throttle =0 
        time.sleep(0.1)
        
    if (cls == 'crosswalk')and(cross_timer == 0):
        time.sleep(2)
        car.throttle = 0
        time.sleep(2)
    
        cross_timer=1000
    if cross_timer>0:    
        cross_timer=cross_timer-1
    
    # PID control
    error = x_sen - 479
    throttle=throttle_zero
    if error > 150:
        car.steering = 1
        car.throttle=throttle

    elif error < -150:
        car.steering = -1
        car.throttle=throttle


    else:
        car.steering = error/150
        car.throttle=throttle


    prev_error = error

