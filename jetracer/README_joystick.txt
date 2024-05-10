shell > jstest /dev/input/js0

'''
Driver version is 2.1.0.
Joystick (SHANWAN Android Gamepad) has 8 axes (X, Y, Z, Rz, Gas, Brake, Hat0X, Hat0Y)
and 15 buttons (BtnA, BtnB, BtnC, BtnX, BtnY, BtnZ, BtnTL, BtnTR, BtnTL2, BtnTR2, BtnSelect, BtnStart, BtnMode, BtnThumbL, BtnThumbR).
'''

AXES (int: [-32767, 32767])
0: left stick - lateral direction (left(-)~right(+))
1: left stick - longitudinal direction (up(-)~down(+))
2: right stick - lateral direction
3: right stick - longitudinal direction
4: default = -32767 ---> push TR2 = +32767
5: default = -32767 ---> push TL2 = +32767
6: move key - left(-32767), right(+32767)
7: move key - up(-32767), right(+32767)

Buttons (bool: [default=off, push=on])
0: A
1: B
2:
3: X
4: Y
5:
6: TL
7: TR
8: TL2
9: TR2
10: Select
11: Start
12: Home
13: ThumbL
14: ThumbR