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
        width: int = 1920, # input # do not revise
        height: int = 1080, # input # do not revise
        _width: int = 960, # output
        _height: int = 540, # output
        frame_rate: int = 10,
        flip_method: int = 0, # do not flip (ex: 2 --> flip by vertex)
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

    def gstreamer_pipeline(self, sensor_id: int, flip_method: int) -> str: ## BGR
        """
        Return a GStreamer pipeline for capturing from the CSI camera
        
        파이프라인 구성 요소 설명 (by GPT)
            nvarguscamerasrc: NVIDIA의 Argus 카메라 소스 요소로, CSI 카메라에서 영상을 캡처합니다.
            video/x-raw(memory:NVMM): 비디오 캡처의 메모리 타입을 NVMM(NVIDIA Memory Management)으로 설정합니다.
            nvvidconv: NVIDIA 비디오 변환 요소로, 비디오를 변환하고 플립합니다.
            videoconvert: 비디오 형식을 변환합니다. 
            appsink: 변환된 비디오 스트림을 응용 프로그램으로 전달합니다.
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

    def run(self, model_traffic=None, model_center=None, model_center2=None) -> None:
        """
        Streaming camera feed
        """
        st = time.time()
        
        global cls_dict
        
        if self.stream:
            # print("Streaming is started now!")
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
                            
                            pil_image = Image.fromarray(frame)

                            with torch.no_grad():
                                image_ts = preprocess(pil_image)
                                output = model_center(image_ts).detach().cpu().numpy()
                                output2 = model_center2(image_ts).detach().cpu().numpy()
                                output3 = model_center3(image_ts).detach().cpu().numpy()
                            
                            x, y = output[0]

                            x = (x / 2 + 0.5) * self._width
                            y = (y / 2 + 0.5) * self._height
                            # print(f'Inferenced road center is ({x}, {y})')
                            
                            x2, y2 = output2[0]

                            x2 = (x2 / 2 + 0.5) * self._width
                            y2 = (y2 / 2 + 0.5) * self._height
                            
                            x3, y3 = output3[0]

                            x3 = (x3 / 2 + 0.5) * self._width
                            y3 = (y3 / 2 + 0.5) * self._height

                            results = model_traffic.predict(frame)
 
                            cls = 'none'
                            try:
                                for i in range(len(results[0].__dict__['boxes'])):
                                    cls = cls_dict[int(results[0].__dict__['boxes'].cls.item())]
                                    print('clsclscls\t\t\t\t', cls)
                            except:
                                print(results[0].__dict__['boxes'].cls)
                                
                            cv2.circle(frame, (int(x), int(y)), radius=5, color=(255, 0, 0))
                            cv2.circle(frame, (int(x2), int(y2)), radius=5, color=(0, 255, 0))
                            cv2.circle(frame, (int(x3), int(y3)), radius=5, color=(0, 0, 255))
                            
                            cv2.imshow(self.window_title, frame)
                        
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
                exit()
                
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
        
        print('\n\nPLZ wait a little second to load librarys and models\n\n')
        
        from ultralytics import YOLO
        model_traffic_path = '/home/ircv3/HYU-2024-Embedded/jetracer/model/yolov8n_traffic_sign_20240510_e32b16.pt'
        model_traffic = YOLO(model_traffic_path) 
        
        import torch
        import torchvision
        from PIL import Image
        import PIL.Image
        from cnn.center_dataset import TEST_TRANSFORMS
        
        device = torch.device('cuda')
        model_center = torchvision.models.alexnet(num_classes=2, dropout=0.0)
        model_center.load_state_dict(torch.load('/home/ircv3/HYU-2024-Embedded/jetracer/model/road_following_model.pth'))
        model_center = model_center.to(device)
        
        model_center2 = torchvision.models.alexnet(num_classes=2, dropout=0.0)
        model_center2.load_state_dict(torch.load('/home/ircv3/HYU-2024-Embedded/jetracer/model/road_following_model_new_e32.pth'))
        model_center2 = model_center2.to(device)
        
        model_center3 = torchvision.models.alexnet(num_classes=2, dropout=0.0)
        model_center3.load_state_dict(torch.load('/home/ircv3/HYU-2024-Embedded/jetracer/model/road_following_model_new2_e32.pth'))
        model_center3 = model_center3.to(device)
            
        def preprocess(image: PIL.Image):
        # def preprocess(image):
            device = torch.device('cuda')    
            image = TEST_TRANSFORMS(image).to(device)
            return image[None, ...]
        
        cls_dict = {0: 'bus', 1: 'crosswalk', 2: 'left', 3: 'right', 4: 'straight'}
            
    
    print('\n\nPLZ press button A to start cam running or capture task\n\n')
    while running:
        pygame.event.pump()
        #  if(joystick.get_button(0)):
        if True:
            # print("############### CAMERA ON ###############")
            if args.capture:
                cam.capt()
            else:
                if args.inference:
                    cam.run(model_traffic, model_center, model_center2)

                else:
                    cam.run()
