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
            
    car.steering = joystick.get_axis(0)
    # print(car.steering)
    
    if(car.throttle < 0 and prev_cmd >= 0):
        back_start = True
        print('A')
            
    throttle = 0
    # if(joystick.get_button(1)):
        
    #     if(back_start == True):
    #         for _ in range(100):
    #             car.throttle = -0.25
    #         time.sleep(0.1)
    #         for _ in range(100):
    #             car.throttle = 0.01
    #         back_start = False
            
    #     throttle = -joystick.get_axis(3)
    #     car.throttle = -0.25
        
    #     continue
        
        
    if(normal):
        throttle = -joystick.get_axis(3)/4.
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        
    elif(turbo):
        throttle = -joystick.get_axis(3)/3.
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        
    elif(super_turbo):
        throttle = -joystick.get_axis(3)
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
    
    else:
        car.throttle = 0.0

    prev_cmd = car.throttle
    
    if(car.throttle < 0 and prev_cmd >= 0):
        back_start = True
        print('A')
        
    # print(throttle)
    # print(car.throttle)
    if joystick.get_button(11): # start button
        running = False
