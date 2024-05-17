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
                    
            result = model_traffic.predict(frame)
            tmp_image_path = '/home/ircv3/HYU-2024-Embedded/jetracer/tmp/tmp_.jpg'
            result[0].save(tmp_image_path)
            
            image_ori = PIL.Image.open(tmp_image_path) # frame: ndarray
            width = image_ori.width
            height = image_ori.height

            with torch.no_grad():
                image = preprocess(image_ori)
                output = model_center(image).detach().cpu().numpy()
            x, y = output[0]

            x = (x / 2 + 0.5) * width
            y = (y / 2 + 0.5) * height
            
        return x

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
throttle_range = (-0.4, 0.4)

normal = False
turbo = False
super_turbo = False
back_start = False

prev_cmd = 0.0

car.steering_offset = -0.07

print(car)

while running:
    pygame.event.pump()
    
    # Mode set
    if(joystick.get_button(7)):
        normal, turbo, super_turbo = True, False, False
    elif(joystick.get_button(9)):
        normal, turbo, super_turbo = False, True, False
    elif(joystick.get_button(6)):
        normal, turbo, super_turbo = False, False, True
    else:
        normal, turbo, super_turbo = False, False, False
            
    
    if(car.throttle < 0 and prev_cmd >= 0):
        back_start = True
        print('A')
            
    throttle = 0
        
    if(normal):
        car.throttle = 0.20
        time.sleep(0.2)
        car.throttle = 0
        time.sleep(0.2)

    elif(turbo):
        throttle = -joystick.get_axis(3)/4.
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        
    elif(super_turbo):
        throttle = -joystick.get_axis(3)
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
    
    else:
        car.throttle = 0.0

    prev_cmd = car.throttle
    
    x_sen = cam.run(model_traffic, model_center)
    ss = 479-x_sen
    po = 1.3
    p = 2
    if ss > 100:
        car.steering = -1
    elif ss < -100:
        car.sttering = 1
    else:
        car.steering = -ss/100
    
    if(car.throttle < 0 and prev_cmd >= 0):
        back_start = True
        print('A')
        
    # print(throttle)
    # print(car.throttle)
    if joystick.get_button(11): # start button
        running = False


