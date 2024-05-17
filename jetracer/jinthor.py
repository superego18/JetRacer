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

running = True
throttle_range = (-0.4, 0.4)

normal = False
turbo = False
super_turbo = False
back_start = False

prev_cmd = 0.0

car.steering_offset = -0.07
car.throttle =0
prev_cmd = car.throttle



while running:
    pygame.event.pump()
    car.throttle = 0
    time.sleep(1)
    car.throttle = 0.10
    time.sleep(1)
    car.throttle = 0.15
    time.sleep(1)
    car.throttle = 0.20
    time.sleep(1)
    car.throttle = 0.25

    car.throttle =0.3

    
