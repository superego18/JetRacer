from __future__ import annotations
from pathlib import Path
from typing import Sequence

import argparse
import cv2
import datetime
# import glob
import logging
import numpy as np
import os
import time

import pygame
from jetracer.nvidia_racecar import NvidiaRacecar

# for model inference
# # from ultralytics.utils.plotting import Annotator
# import supervision as sv
# import torchvision.transforms as transforms

# from jtop import jtop # Use this to monitor compute usage (for Jetson Nano)

logging.getLogger().setLevel(logging.INFO)

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

car = NvidiaRacecar()
joystick = pygame.joystick.Joystick(0)
joystick.init()
running = True

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

        # Make record directory
        if save:
            assert save_path is not None, "Please provide a save path"
            os.makedirs(self.save_path, exist_ok=True) # if path does not exist, create it
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)

            logging.info(f"Save directory: {self.save_path}")
        
        if capture:
            self.save_path = 'capture'
            os.makedirs(self.save_path, exist_ok=True) # if path does not exist, create it

            logging.info(f"Save directory: {self.save_path}")

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

    def run(self, model=None) -> None:
        """
        Streaming camera feed
        """
        if self.stream:
            print("Streaming is started now!")
            cv2.namedWindow(self.window_title)

        if self.cap[0].isOpened():
            try:
                while True:
                    pygame.event.pump()
                    t0 = time.time()
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    _, frame = self.cap[0].read()

                    if self.save:
                        cv2.imwrite(str(self.save_path / f"{timestamp}.jpg"), frame)

                    if self.log:
                        print(f"FPS: {1 / (time.time() - t0):.2f}")

                    if self.stream:
                        
                        if self.inference:
                            result = model.predict(frame)                 
                            cv2.imshow(self.window_title, result[0].plot())
                            
                        else:
                            cv2.imshow(self.window_title, frame)

                        if cv2.waitKey(1) == ord('q'):
                            break
                        elif joystick.get_button(1):
                            print("###############CAMERA OFF###############")
                            break
                        
            except Exception as e:
                print(e)
            finally:
                self.cap[0].release()
                cv2.destroyAllWindows()
                
    def capt(self) -> None:
        "Capture images for making custom dataset (chanju 240510)"
        
        if self.stream:
            cv2.namedWindow(self.window_title)
            
        if self.cap[0].isOpened():
            save_num = 0
            try:
                while True:
                    pygame.event.pump()
                    t0 = time.time()
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    _, frame = self.cap[0].read()
                    
                    if joystick.get_button(6):
                        self.save = True

                    if self.save:
                        cv2.imwrite(f"{self.save_path}/{timestamp}.jpg", frame)
                        
                        save_num += 1
                        print(f'Save num: {save_num}')
                        print(f'Save image: {self.save_path}/{timestamp}.jpg')
                        time.sleep(0.5)
                        self.save = False

                    if self.log:
                        print(f"FPS: {1 / (time.time() - t0):.2f}")

                    if self.stream:
                        cv2.imshow(self.window_title, frame)

                        if cv2.waitKey(1) == ord('q'):
                            break
                        elif joystick.get_button(1):
                            print("############### CAMERA OFF ###############")
                            break
                        
            except Exception as e:
                print(e)
            finally:
                self.cap[0].release()
                cv2.destroyAllWindows()
            
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
    
    if args.inference: 
        from ultralytics import YOLO
        model_path = '/home/ircv3/HYU-2024-Embedded/jetracer/model/yolov8n_traffic_sign_20240510_e32b16.pt'
        model = YOLO(model_path) 
    
    print('\n\nPLZ press button A to start cam running or capture task\n\n')
    while running:
        pygame.event.pump()
        if(joystick.get_button(0)):
            print("############### CAMERA ON ###############")
            if args.capture:
                cam.capt()
            else:
                cam.run(model)
