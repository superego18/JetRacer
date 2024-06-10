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
from collections import deque
import datetime

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

            # logging.info(f"Save directory: {self.save_path}")
        
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
            "nvarguscamerasrc sensor-id=%d gainrange=\"10 10\" ispdigitalgainrange=\"10 10\" ! "
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

    def run(self, model_center=None) -> None:
        
        print("############### CAMERA ON ###############")
        
        if self.stream:
            cv2.namedWindow(self.window_title)

        if self.cap[0].isOpened():
            
            try:
                pygame.event.pump()
                
                t0 = time.time()
                
                _, frame = self.cap[0].read()

                # if self.inference:
                    
                # Choose the right model according to the traffic sign
                pil_image = Image.fromarray(frame)
                
                with torch.no_grad():
                    image_ts = preprocess(pil_image)
                
                    output = model_center(image_ts).detach().cpu().numpy()
                    color_ = (0, 255, 0)
                                        
                x, y = output[0]
                x = (x / 2 + 0.5) * self._width
                y = (y / 2 + 0.5) * self._height

                if self.stream:
                    if self.inference:
                        cv2.circle(frame, (int(x), int(y)), radius=5, color=color_)
                    cv2.imshow(self.window_title, frame)

                #     if cv2.waitKey(1) == ord('q'):
                #         break
                # if joystick.get_button(1):
                #     print("############### CAMERA OFF ###############")
                #     break
                
                if self.save:
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    cv2.imwrite(str(self.save_path / f"ori_{timestamp}.jpg"), frame)
                
                if self.log:
                    print(f'Inferenced road center is ({x:.1f}, {y:.1f})')
                    print(f"Real FPS: {1 / (time.time() - t0):.1f}")
                    
                return x
                        
            except Exception as e:
                print(e)
                print('error is occured')

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
    args.add_argument('--add',
        type = int,
        default = 1)
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
    
    # if args.inference: 
        
    print('\n### Please wait a little second to load librarys and models ###\n')
    
    from ultralytics import YOLO
    import torch
    import torchvision
    from PIL import Image
    from cnn.center_dataset import TEST_TRANSFORMS
    
    device = torch.device('cuda')
            
    
    model_center = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    # model_center.load_state_dict(torch.load('/home/ircv3/HYU-2024-Embedded/jetracer/model/straight_best1.pth'))
    model_center.load_state_dict(torch.load('/home/ircv3/HYU-2024-Embedded/jetracer/model/straight_best2.pth'))
    model_center = model_center.to(device)
    
    def preprocess(image: Image):
        device = torch.device('cuda')    
        image = TEST_TRANSFORMS(image).to(device)
        return image[None, ...]
    
        
## control
running = True

prev_error = 0.0
integral = 0.0
integral_deque = deque(maxlen=30)

car.steering_offset = -0.07
steering_range = (-1.1, 1.1)

# drive = False

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

time.sleep(0.2)
car.throttle = 0.35
time.sleep(0.2)
car.throttle = 0

throttle_zero = 0.2

running = True
paused = False
crosswalk_counter = 500
bus_counter=500
bus=1

while True:
    pygame.event.pump()
    
    throttle = throttle_zero
    x_sen= cam.run(model_center)
    if x_sen < 0:
        x_sen = 0
    elif x_sen > 960:
        x_sen = 960

    if(joystick.get_button(7)):
        throttle_zero+=0.001
    if(joystick.get_button(6)):
        throttle_zero-=0.001
    while(joystick.get_button(0)):
        car.throttle = 0
        time.sleep(0.5)
        pygame.event.pump()

    if(joystick.get_button(1)):
            running = False
            car.throttle = 0
            time.sleep(0.5)

    if running==False:
        break

    # PID 제어
    error = x_sen - 495
    integral_deque.append(error)
    integral = sum(integral_deque)
    derivative = error - prev_error
    
    if integral > 8000:
        integral = 8000
    elif integral < -8000:
        integral = -8000
    
    if error > 500:
        car.steering = 1
        car.throttle = throttle
    elif error < -500:
        car.steering = -1
        car.throttle = throttle
    else:
        steering = error / 450 + integral / 40000 + derivative / 400
        # if steering > 0:
        #     car.steering = steering * 1.2
        # else:
        #     car.steering = steering
            
        car.steering = steering
        car.throttle = throttle
            
        print(f"error : {error/450}")
        print(f"integral : {integral/40000}")
        print(f"derivative : {derivative/500}")
        print(f"steering : {car.steering}")
        print(f"x_sen : {x_sen}")
        
    prev_error = error

