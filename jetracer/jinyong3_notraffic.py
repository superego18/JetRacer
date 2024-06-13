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

    def run(self) -> None:
        
        global device, model_cup, cnt1
        
        cnt1 +=1
        
        print("############### CAMERA ON ###############")
        
        if self.stream:
            cv2.namedWindow(self.window_title)

        if self.cap[0].isOpened():
            
            # try:
                
            #     while True:
            pygame.event.pump()
            
            t0 = time.time()
            
            _, frame = self.cap[0].read()
            
            # img = torch.tensor(frame).unsqueeze(3).to(device)
            
            # print(img.shape)

            results = model_cup.predict(frame)
            
            blue_bool, red_bool = False, False
            red_cups_3d, blue_cups_3d = [], []
            
            if results[0].__dict__['boxes'].shape[0] >0 :
                
                red_cups_pts = []
                blue_cups_pts = []
            
                for bbox in results[0].__dict__['boxes']:
                    if bbox.conf >= 0.3:
                        x = (float(bbox.xywhn[0][0])) * 960
                        y = (float(bbox.xywhn[0][1]) + float(bbox.xywhn[0][3])/2) * 540
                        if int(bbox.cls.item()) == 0:
                            blue_cups_pts.append([x,y])
                        else:
                            red_cups_pts.append([x,y])
                            
                if len(red_cups_pts) > 0:    
                    red_cups_3d = undis_homo(red_cups_pts)
                    red_bool = True
                if len(blue_cups_pts) > 0:
                    blue_cups_3d = undis_homo(blue_cups_pts)
                    blue_bool = True
                            
                    
                if self.stream:
                    frame_3d = np.ones((540, 800, 3), dtype=np.uint8) * 255 # 매 프레임 초기화 # 54cm * 80cm  # h,w
                    annotated_frame = results[0].plot()
                    # plt.figure()
                    if red_bool:
                        for pt in red_cups_pts:
                            cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), radius=5, color=(0, 0, 255), thickness=-1)
                        for pt in red_cups_3d:
                            x = 540 - (pt[0]+3.5)*10
                            y = 400 - pt[1]*10
                            x_txt = round(pt[0]+3.5,1)
                            y_txt = round(pt[1],1)
                            txt = f'x, y: ({x_txt}, {y_txt})cm'
                            cv2.circle(frame_3d, (int(y), int(x)), radius=35, color=(0,0,255))
                            cv2.putText(frame_3d, txt, (int(y), int(x)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0)) #  font, font_scale, color, thickness

                        # # 3d
                        # x_values = [pt[0] for pt in red_cups_3d]
                        # y_values = [pt[1]+3.5 for pt in red_cups_3d]
                        # plt.scatter(x_values, y_values, color='red')
                    if blue_bool:
                        for pt in blue_cups_pts:
                            cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), radius=5, color=(255, 0, 0), thickness=-1)
                        for pt in blue_cups_3d:
                            x = 540 - (pt[0]+3.5)*10
                            y = 400 - pt[1]*10
                            x_txt = round(pt[0]+3.5,1)
                            y_txt = round(pt[1],1)
                            txt = f'x, y: ({x_txt}, {y_txt})cm'
                            cv2.circle(frame_3d, (int(y), int(x)), radius=35, color=(255,0,0))
                            cv2.putText(frame_3d, txt, (int(y), int(x)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0)) #  font, font_scale, color, thickness

                        # # 3d
                        # x_values = [pt[0] for pt in blue_cups_3d]
                        # y_values = [pt[1]+3.5 for pt in blue_cups_3d]
                        # plt.scatter(x_values, y_values, color='red')
                        
                    # cv2.imshow(self.window_title, annotated_frame)
                    combined_image = cv2.hconcat([annotated_frame, frame_3d])
                    # cv2.imshow('Combined Image', combined_image)
                    
                    # plt.xlabel('X')
                    # plt.ylabel('Y')
                    # plt.title('Scatter Plot with Points')
                    # plt.grid(True)
                    # plt.show()
                    
                    
                    
                    return blue_cups_3d, red_cups_3d, combined_image
                         
                    # if cv2.waitKey(1) == ord('q'):
                    #     break

                        
                    # if joystick.get_button(1):
                    #     print("############### CAMERA OFF ###############")
                    #     break
                     
                    
                    if self.save:
                        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                        cv2.imwrite(str(self.save_path / f"ori_{timestamp}.jpg"), frame)
                        
                    
                    if self.log:
                        print(f'Inferenced road center is ({x:.1f}, {y:.1f})')
                        print(f"Real FPS: {1 / (time.time() - t0):.1f}")
                        
            # except Exception as e:
            #     print(e)
            #     print('error is occured')

            # finally:
            #     # TODO: Revise cam release
            #     self.cap[0].release()
            #     cv2.destroyAllWindows()
            #     exit()
                
            return blue_cups_3d, red_cups_3d, None
                
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
    
    cnt1, cnt2 = 0, 0
        
    print('\n### Please wait a little second to load librarys and models ###\n')
    
    from ultralytics import YOLO
    import torch
    import pickle
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    device = torch.device('cuda')
            
    model_cup_path = '/home/ircv3/HYU-2024-Embedded/jetracer/model/yolo_traffic_cup2.pt'
    model_cup = YOLO(model_cup_path)
    # model_cup = model_cup.to(device)
    
    with open('calibration_data.pkl', 'rb') as f:
        calib_data = pickle.load(f)

    mtx = np.array(calib_data['I'])
    dist = np.array(calib_data['dist'])

    with open('homography_data.pkl', 'rb') as f:
        homo_data = pickle.load(f)

    H = np.array(homo_data['H'])
    
    def undis_homo(pts):
        
        global mtx, dist, H
                
        dist_pts = np.array(pts, dtype=np.float32)
        undist_pts = cv2.undistortPoints(dist_pts, mtx, dist) # (n, 1, 2)

        x2 = (undist_pts[:, :, 0] + 1) / 2 * 960 # (2, 1)
        y2 = (undist_pts[:, :, 1] + 1) / 2 * 540

        undist_pts = np.hstack([x2, y2], dtype=np.float32) # (n, 2)

        pts_homo = np.concatenate([undist_pts, np.ones((undist_pts.shape[0], 1))], axis=1)
        pts_tran_homo = np.dot(H, pts_homo.T).T
        pts_tran = pts_tran_homo[:, :2] / pts_tran_homo[:, 2:] # (n, 2)

        return pts_tran
    
## control
running = True

prev_steering = 0
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
throttle = throttle_zero

k = 10 # TODO 튜닝해야함
vx = 150 # [cm/s]

psi_deque = deque(maxlen=1)
distance_error_deque = deque(maxlen=1)

def weighted_moving_average(values, alpha=0.5):
    weighted_sum = 0.0
    weight_sum = 0.0
    n = len(values)
    for i in range(n):
        weight = alpha ** (n - i)
        weighted_sum += weight * values[i]
        weight_sum += weight

    return weighted_sum / weight_sum


while True:
    try:
        pygame.event.pump()
        
        blue_pts, red_pts, _ = cam.run()
        
        ####################### 여기서부터 수정 ###################### 
        # y축 방향은 왼쪽, 파란컵은 오른쪽
        
        if len(blue_pts) == 0:
            # blue_pts = perv_blue_pts
            blue_pts = np.array([[0, -25], [-25, -25]])
            
        else:
            blue_pts = np.append(blue_pts, [[0, -25], [-25, -25]], axis=0)
            
        if len(red_pts) == 0:
            # red_pts = prev_red_pts
            red_pts = np.array([[0, 25], [-25, 25]])
            
        else:
            red_pts = np.append(red_pts, [[0, 25], [-25, 25]], axis=0)
            
        # x 값이 200 이상인 배열 제거
        max_x_value = 200
        
        blue_pts = blue_pts[blue_pts[:, 0] < max_x_value]
        red_pts = red_pts[red_pts[:, 0] < max_x_value]
            
        # x, y 좌표 분리
        blue_x = blue_pts[:, 0]
        blue_y = blue_pts[:, 1]
        red_x = red_pts[:, 0]
        red_y = red_pts[:, 1]

        # 다항식 차수 (점의 개수 - 1)
        degree_blue = len(blue_x) - 1
        degree_red = len(red_x) - 1
        max_degree = 2

        # 다항식 피팅
        if degree_blue == 1 and degree_red == 1:
            car.steering = prev_steering

        # elif(degree_blue == 1 and degree_red != 1): # 빨갱이
        #     red_poly = np.polyfit(red_x, red_y, max_degree)
        #     derivative_coeffs = np.polyder(red_poly)
        #     slope_at_zero = np.polyval(derivative_coeffs, 0)
        #     angle_in_degrees = np.degrees(np.arctan(slope_at_zero))
        #     car.steering = max(min(angle_in_degrees/50, 1.0), 0.2)
        #     psi_deque.append(angle_in_degrees)

        # elif(degree_blue != 1 and degree_red == 1): # 파랭이
        #     blue_poly = np.polyfit(blue_x, blue_y, max_degree)
        #     derivative_coeffs = np.polyder(blue_poly)
        #     slope_at_zero = np.polyval(derivative_coeffs, 0)
        #     angle_in_degrees = np.degrees(np.arctan(slope_at_zero))
        #     car.steering = max(min(angle_in_degrees/50, -0.2), -1.0)
        #     psi_deque.append(angle_in_degrees)


            
        else:
            # # 빨간색만 봤을 때
            if degree_blue == 1 and degree_red != 1:
                red_poly = np.polyfit(red_x, red_y, max_degree)
                blue_poly = red_poly.copy()
                blue_poly[-1] = min(blue_poly[-1] - 40, 40)
                # print("Blue polynomial:", blue_poly)
            
            # 파란색만 봤을 때
            elif degree_blue != 1 and degree_red == 1:
                blue_poly = np.polyfit(blue_x, blue_y, max_degree)
                red_poly = blue_poly.copy()
                red_poly[-1] = max(red_poly[-1] +40,40)
                # print("Red polynomial:", red_poly)

            else:
                blue_poly = np.polyfit(blue_x, blue_y, max_degree)
                red_poly = np.polyfit(red_x, red_y, max_degree)

            # 다항식 중앙값 -> heading error and distanse_error
            mid_poly = (blue_poly + red_poly) / 2
            derivative_coeffs = np.polyder(mid_poly) # 다항식의 도함수의 계수 계산
            slope_at_zero = np.polyval(derivative_coeffs, 0) # x = 0에서의 기울기 계산
            angle_in_degrees = np.degrees(np.arctan(slope_at_zero)) # 기울기를 라디안 각도로 변환
            # TODO 현재 따라가야할 경로의 y절편을 사용하는데, x축이 차량의 종방향이므로 그렇게 설정함.
            distance_error = mid_poly[-1] # [cm] # y 절편은 다항식의 상수 항
            

            psi_deque.append(angle_in_degrees)
            psi = weighted_moving_average(psi_deque)
            
            if distance_error*angle_in_degrees < 0 and abs(distance_error) < 20:
                distance_error = 0
            
            distance_error_deque.append(distance_error)
            ctr_term = np.degrees(np.arctan2(k*weighted_moving_average(distance_error_deque), vx))

            steering_angle_degrees = -psi - ctr_term # degrees 기준
            
            # TODO 22 수정
            if steering_angle_degrees > 50:
                car.steering = 0.5
                
            elif steering_angle_degrees < -50:
                car.steering = -0.5
            else:
            # TODO 44 수정    
                car.steering = steering_angle_degrees/100

            # 결과 출력
            # 중앙값 다항식 출력
            # print("Mid polynomial coefficients : ", mid_poly)
            # print(f"Angle error    : {angle_in_degrees:>10.2f} degrees")
            # print(f"Distance error : {distance_error:>10.2f} cm")
            # print(f"psi            : {psi:>10.2f} degrees")
            # print(f"ctr_term       : {ctr_term:>10.2f} degrees")
            print(f"k              : {k:>10.2f}")
            # print(f"steering angle : {steering_angle_degrees:>10.2f} degrees")
            # print(f"car.steering   : {car.steering:>10.2f}")
            
                # 다항식 그리기

            # x_new = np.linspace(0, 200, 100)              
            # blue_y_new = np.polyval(blue_poly, x_new)
            # red_y_new = np.polyval(red_poly, x_new)
            # mid_y_new = np.polyval(mid_poly, x_new)
            # plt.clf()
            # plt.plot(blue_x, blue_y, 'o', label='Blue points', color='blue')
            # plt.plot(x_new, blue_y_new, label='Blue polynomial', color='blue')
            # plt.plot(red_x, red_y, 'o', label='Red points', color='red')
            # plt.plot(x_new, red_y_new, label='Red polynomial', color='red')
            # plt.plot(x_new, mid_y_new, '--', label='Mid polynomial', color='black')
            # plt.xlim(0, 200)
            # plt.ylim(-100, 100)
            # plt.title(f'e_psi: {angle_in_degrees:>10.2f}, e_d: {distance_error:>10.2f}, psi: {psi:>10.2f}, c: {ctr_term:>10.2f}, k: {k:>10.2f},\nd: {steering_angle_degrees:>10.2f}, s: {car.steering:>10.2f}')
            # plt.legend()
            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            # plt.savefig(f'/home/ircv3/HYU-2024-Embedded/jetracer/capture/ss/{timestamp}.png')
            
            # plt.draw()
            # plt.pause(0.01)
            
            
            
            if _:
                cv2.imwirte(f'/home/ircv3/HYU-2024-Embedded/jetracer/capture/ss/{timestamp}__.png', _)
            
        if(joystick.get_button(7)):
            throttle+=0.001
        if(joystick.get_button(6)):
            throttle-=0.001
        if(joystick.get_button(9)):
            k=k*1.01
        if(joystick.get_button(8)):
            k=k*0.99
        while(joystick.get_button(0)):
            car.throttle = 0
            time.sleep(0.5)
            pygame.event.pump()

        if(joystick.get_button(1)):
                running = False
                car.throttle = 0
                time.sleep(0.5)

        if running==False:
            print('cntcntcntcnt', cnt1, cnt2)
            break
        cnt2 += 1
        print("-------------------------------------------------")
        car.throttle = throttle
        prev_steering = car.steering
        perv_blue_pts = blue_pts
        prev_red_pts = red_pts
    
    except:
        print('error')