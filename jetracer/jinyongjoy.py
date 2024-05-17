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
            

    if(car.throttle < 0 and prev_cmd >= 0):
        back_start = True
        print('A')
            
    throttle = 0
     
    if(normal):

        car.throttle = 0.18
    elif(turbo):
        throttle = -joystick.get_axis(3)/3.
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        
    elif(super_turbo):
        throttle = -joystick.get_axis(3)
        car.throttle = max(throttle_range[0], min(throttle_range[1], throttle))
    
    else:
        car.throttle = 0.0

    prev_cmd = car.throttle

    # print(throttle)
    # print(car.throttle)
    if joystick.get_button(11): # start button
        running = False
