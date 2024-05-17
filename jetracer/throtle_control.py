import os
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar



import time
import sys
import termios
import tty



car = NvidiaRacecar()





time.sleep(0.2)
car.throttle = 0.20
time.sleep(0.2)
car.throttle = 0
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

throttle_origin = 0.0
print("Press 'n' to increase throttle_origin by 0.01, 'm' to decrease throttle_origin by 0.01, and 'd' to stop.")

while True:
    key = get_key()
    if key == 'm':
        throttle_origin += 0.001
        car.throttle = throttle_origin
        print(f"throttle_origin increased to {throttle_origin:.3f}")
    elif key == 'n':
        throttle_origin -= 0.001
        car.throttle = throttle_origin
        print(f"throttle_origin decreased to {throttle_origin:.3f}")
    elif key == 'd':
        car.throttle = 0
        print(f"Exiting loop with throttle_origin = {throttle_origin:.3f}")
        break
    else:
        print("Invalid input, please enter 'n', 'm', or 'd'.")
    print(f"Current throttle_origin = {throttle_origin:.3f}")

# The next steps can be added here
print("Next steps go here.")
print(throttle_origin)
