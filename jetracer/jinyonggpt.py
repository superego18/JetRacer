from jetracer.nvidia_racecar import NvidiaRacecar
import pygame
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import argparse

# For headless mode
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize JetRacer and joystick
car = NvidiaRacecar()
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

class Camera:
    def __init__(self, sensor_id=0, width=1920, height=1080, frame_rate=30):
        self.cap = cv2.VideoCapture(sensor_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame_rate = frame_rate
        self.executor = ThreadPoolExecutor(max_workers=2)

    def read_frame(self):
        _, frame = self.cap.read()
        return frame

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        frame = transform(frame)
        return frame

    def capture_and_preprocess(self):
        frame = self.read_frame()
        return self.preprocess(frame)

    def capture_and_preprocess_async(self):
        future = self.executor.submit(self.capture_and_preprocess)
        return future

def model_inference(image_tensor):
    # Load your model here
    model = None  # Load your model
    output = model(image_tensor)
    return output

def control_car(output):
    # Process output and control car steering
    # Example: Calculate steering angle based on output
    steering_angle = 0.0  # Calculate steering angle
    car.steering = steering_angle

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_id', type=int, default=0, help='Camera ID')
    args = parser.parse_args()

    # Initialize camera
    cam = Camera(sensor_id=args.sensor_id)

    # Main loop
    running = True
    while running:
        pygame.event.pump()

        # Capture and preprocess frame asynchronously
        future = cam.capture_and_preprocess_async()
        image_tensor = future.result()

        # Perform model inference
        with torch.no_grad():
            output = model_inference(image_tensor)

        # Control the car based on model output
        control_car(output)

        # Check if the start button is pressed to exit
        if joystick.get_button(11):  # start button
            running = False

# Release resources
cv2.destroyAllWindows()
car.free()
