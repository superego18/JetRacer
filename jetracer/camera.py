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
        width: int = 1920, # input # do not revise  # [3280, 3280, 1920, 1640, 1280] #[3.4166, 3.4166, 2, 1.70833, 1.333] 
        height: int = 1080, # input # do not revise # [2464, 1848, 1080, 1232, 720] #[4.56.., 3.422, 2, 2.281.., 1.333]
        _width: int = 960, # output # value you need
        _height: int = 540, # output # value you need 
        frame_rate: int = 10, # value you need # max = [21, 28, ~30, ~30, ~60]
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
        '''
        VideoCapture 객체는 OpenCV에서 비디오 캡처를 위해 사용되는 클래스
        
        self.cap[0]가 가리키는 것은 VideoCapture 객체의 인스턴스
        
        - isOpened(): 비디오 캡처가 성공적으로 열렸는지 여부를 반환.
        - open(filename/device): 비디오 파일이나 카메라를 엽니다. 성공하면 True를 반환하고, 실패하면 False를 반환.
        - read(): 프레임을 캡처하고, 성공 여부와 프레임을 반환, (ret, frame = cap.read()).
        - release(): 비디오 캡처를 해제하고 모든 리소스를 반환.
        - grab(): 제 프레임 데이터를 반환하지는 않지만, 내부 버퍼에 다음 프레임을 가져옴.
        - retrieve(): 내부 버퍼에서 프레임을 가져옵니다. grab() 메서드가 호출된 후에 사용, (ret, frame = cap.retrieve()).
        '''

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
            "nvarguscamerasrc sensor-id=%d gainrange=\"10 10\" ispdigitalgainrange=\"5 5\" ! "
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

    def run(self, model_traffic=None, model_center=None, model_left=None, model_right=None, cls_dict=None) -> None:
        
        print("############### CAMERA ON ###############")
        
        if self.stream:
            cv2.namedWindow(self.window_title)

        if self.cap[0].isOpened():
            
            try:

                pygame.event.pump()
                
                t0 = time.time()
                
                _, frame = self.cap[0].read()

                if self.save:
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    cv2.imwrite(str(self.save_path / f"{timestamp}.jpg"), frame)

                if self.inference:
                    
                    # Determine the existence and class of traffic signs
                    results = model_traffic.predict(frame)

                    # Basic cls value
                    cls = 5
                    
                    # When there are multiple signs, choose the closest one
                    max_bbox_size = 0
                    for bbox in results[0].__dict__['boxes']:
                        
                        bbox_size = int(bbox.xywh[0][2]) * int(bbox.xywh[0][3]) 
                        print(int(bbox.xywh[0][2]), int(bbox.xywh[0][3]))
                        
                        if bbox_size > max_bbox_size:
                            max_bbox_size = bbox_size
                            cls = int(bbox.cls.item())                              
                            
                    # Choose the right model according to the traffic sign
                    pil_image = Image.fromarray(frame)
                    
                    with torch.no_grad():
                        image_ts = preprocess(pil_image)
                    
                        # No traffic sign | Crosswalk | Bus | Straight
                        if cls == 5 or cls == 0 or cls == 1 or cls == 4:
                            output = model_center(image_ts).detach().cpu().numpy()
                            color_ = (0, 255, 0)
                        # Left
                        elif cls == 2:
                            output = model_left(image_ts).detach().cpu().numpy()
                            color_ = (255, 0, 0)
                        # Right
                        else:
                            output = model_right(image_ts).detach().cpu().numpy()
                            color_ = (0, 0, 255)
                                            
                    x, y = output[0]
                    x = (x / 2 + 0.5) * self._width
                    y = (y / 2 + 0.5) * self._height

                if self.stream:
                    if self.inference:
                        annotated_frame = results[0].plot()
                        cv2.circle(annotated_frame, (int(x), int(y)), radius=5, color=color_)
                        cv2.imshow(self.window_title, annotated_frame)
                    else:
                        cv2.imshow(self.window_title, frame)

                #     if cv2.waitKey(1) == ord('q'):
                #         break
                # if joystick.get_button(1):
                #     print("############### CAMERA OFF ###############")
                #     break
                
                if self.log:
                    print(f'Determined class is {cls_dict[cls]}')
                    print(f'Inferenced road center is ({x:.1f}, {y:.1f})')
                    print(f"Real FPS: {1 / (time.time() - t0):.1f}")
                
                if self.inference:
                    return x, cls
                        
            except Exception as e:
                print(e)

            # finally:
            #     # TODO: Revise cam release
            #     self.cap[0].release()
            #     cv2.destroyAllWindows()
            #     exit()
                
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

                    # if self.save:
                    #     cv2.imwrite(f"{self.save_path}/{timestamp}.jpg", frame)
                        
                    #     save_num += 1
                    #     print(f'Save num: {save_num}')
                    #     print(f'Save image: {self.save_path}/{timestamp}.jpg')
                    #     time.sleep(0.5)
                    #     self.save = False
                    
                    if self.save:
                        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                        cv2.imwrite(f"{self.save_path}/{timestamp}.jpg", frame)

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
        
        print('\n### Please wait a little second to load librarys and models ###\n')
        
        from ultralytics import YOLO
        import torch
        import torchvision
        from PIL import Image
        from cnn.center_dataset import TEST_TRANSFORMS
        
        device = torch.device('cuda')
                
        model_traffic_path = '/home/ircv3/HYU-2024-Embedded/jetracer/model/yolov8n_traffic_sign_20240510_e32b16.pt'
        model_traffic = YOLO(model_traffic_path) 
        
        model_center = torchvision.models.alexnet(num_classes=2, dropout=0.0)
        model_center.load_state_dict(torch.load('/home/ircv3/HYU-2024-Embedded/jetracer/model/road_following_model_new2_e32.pth'))
        model_center = model_center.to(device)
        
        model_left = torchvision.models.alexnet(num_classes=2, dropout=0.0)
        model_left.load_state_dict(torch.load('/home/ircv3/HYU-2024-Embedded/jetracer/model/road_following_model_left_b8e64.pth'))
        model_left = model_left.to(device)
        
        model_right = torchvision.models.alexnet(num_classes=2, dropout=0.0)
        model_right.load_state_dict(torch.load('/home/ircv3/HYU-2024-Embedded/jetracer/model/road_following_model_right_b8e64.pth'))
        model_right = model_right.to(device)
            
        def preprocess(image: Image):
            device = torch.device('cuda')    
            image = TEST_TRANSFORMS(image).to(device)
            return image[None, ...]
        
        cls_dict = {0: 'bus', 1: 'crosswalk', 2: 'left', 3: 'right', 4: 'straight', 5: 'no_traffic_sign'}
            
    
    print('\n### Please press button A to start cam running or capture task ###\n')
    
    # TODO: Add self-driving button, cam open function
    while running:
        pygame.event.pump()
        #  if(joystick.get_button(0)):
        if True:
            # Choose cam's task by options. ['--capture', '--stream', '--log', '--inference']
            if args.capture:
                cam.capt()
            else:
                if args.inference:
                    cam.run(model_traffic, model_center, model_left, model_right, cls_dict)
                else:
                    cam.run()
