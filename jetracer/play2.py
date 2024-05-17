import numpy as np
import cv2, random, math, time, sys
import yaml

def settings():
    with open('config.yaml', encoding='UTF-8') as config:
        cfg = yaml.load(config, Loader=yaml.FullLoader)
    global WIDTH, HEIGHT, ROI_START_HEIGHT, ROI_HEIGHT, CANNY_LOW, CANNY_HIGH, HOUGH_ABS_SLOPE_RANGE, HOUGH_THRESHOLD, HOUGH_LENGTH, HOUGH_GAP, CANNY_LOW, CANNY_HIGH, DEBUG
    global before_left, before_right
    global START_SPEED, MIN_SPEED, MAX_SPEED, STOP
    global ACCELERATION_STEP, DECELERATION_STEP
    global P_GAIN, I_GAIN, D_GAIN
    global pError, iError, dError
    pError = 0
    iError = 0
    dError = 0
    WIDTH= cfg['IMAGE']['WIDTH']
    HEIGHT = cfg['IMAGE']['HEIGHT']
    ROI_START_HEIGHT = cfg['IMAGE']['ROI_START_HEIGHT']
    ROI_HEIGHT = cfg['IMAGE']['ROI_HEIGHT']
    CANNY_LOW = cfg['CANNY']['LOW_THRESHOLD']
    CANNY_HIGH = cfg['CANNY']['HIGH_THRESHOLD']
    HOUGH_ABS_SLOPE_RANGE = cfg['HOUGH']['ABS_SLOPE_RANGE']
    HOUGH_THRESHOLD = cfg['HOUGH']['THRESHOLD']
    HOUGH_LENGTH = cfg['HOUGH']['MIN_LINE_LENGTH']
    HOUGH_GAP = cfg['HOUGH']['MIN_LINE_GAP']
    CANNY_LOW = cfg['CANNY']['LOW_THRESHOLD']
    CANNY_HIGH = cfg['CANNY']['HIGH_THRESHOLD']
    START_SPEED = cfg['CAR']['START_SPEED']
    MIN_SPEED = cfg['CAR']['MIN_SPEED']
    MAX_SPEED = cfg['CAR']['MAX_SPEED']
    STOP = cfg['CAR']['STOP']
    ACCELERATION_STEP = cfg['CAR']['ACCELERATION_STEP']
    DECELERATION_STEP = cfg['CAR']['DECELERATION_STEP']
    P_GAIN = cfg['PID']['P_GAIN']
    I_GAIN = cfg['PID']['I_GAIN']
    D_GAIN = cfg['PID']['D_GAIN']
    DEBUG = cfg['DEBUG']

    before_left = 0
    before_right = WIDTH

def us_settings():
    with open('config.yaml', encoding='UTF-8') as config:
        cfg = yaml.load(config, Loader=yaml.FullLoader)
    global WIDTH, HEIGHT, ROI_START_HEIGHT, ROI_HEIGHT, CANNY_LOW, CANNY_HIGH, HOUGH_ABS_SLOPE_RANGE, HOUGH_THRESHOLD, HOUGH_LENGTH, HOUGH_GAP, CANNY_LOW, CANNY_HIGH, DEBUG
    global before_left, before_right
    global START_SPEED, MIN_SPEED, MAX_SPEED, STOP
    global ACCELERATION_STEP, DECELERATION_STEP
    global P_GAIN, I_GAIN, D_GAIN
    global pError, iError, dError
    pError = 0
    iError = 0
    dError = 0
    WIDTH= cfg['IMAGE']['WIDTH']
    HEIGHT = cfg['IMAGE']['HEIGHT']
    ROI_START_HEIGHT = cfg['IMAGE']['ROI_START_HEIGHT']
    ROI_HEIGHT = cfg['IMAGE']['ROI_HEIGHT']
    CANNY_LOW = cfg['CANNY']['LOW_THRESHOLD']
    CANNY_HIGH = cfg['CANNY']['HIGH_THRESHOLD']
    HOUGH_ABS_SLOPE_RANGE = cfg['HOUGH']['ABS_SLOPE_RANGE']
    HOUGH_THRESHOLD = cfg['HOUGH']['THRESHOLD']
    HOUGH_LENGTH = cfg['HOUGH']['MIN_LINE_LENGTH']
    HOUGH_GAP = cfg['HOUGH']['MIN_LINE_GAP']
    CANNY_LOW = cfg['CANNY']['LOW_THRESHOLD']
    CANNY_HIGH = cfg['CANNY']['HIGH_THRESHOLD']
    START_SPEED = cfg['CAR']['START_SPEED']
    MIN_SPEED = cfg['CAR']['US_MIN_SPEED']
    MAX_SPEED = cfg['CAR']['US_MAX_SPEED']
    STOP = cfg['CAR']['STOP']
    ACCELERATION_STEP = cfg['CAR']['ACCELERATION_STEP']
    DECELERATION_STEP = cfg['CAR']['DECELERATION_STEP']
    P_GAIN = cfg['PID']['P_GAIN']
    I_GAIN = cfg['PID']['I_GAIN']
    D_GAIN = cfg['PID']['D_GAIN']
    DEBUG = cfg['DEBUG']
    before_left = 0
    before_right = WIDTH

def traffic_recognition(frame):
   
    # roi = frame[0:240, 80:560]      # 관심 영역 설정
    roi1 = frame[200:400, 0:180]
    # roi2 = frame[0:200, 150:640]      # 관심 영역 설정
    # # 조정 필요
    # cv2.imshow('roi', roi)

    # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # hsv 변환
    hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
    # hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)  # hsv 변환
    # cv2.imshow('hsv', hsv)

    # 빨간색 색상 검출 영역 설정
    lower_red = np.array([155, 0, 230])
    upper_red = np.array([179, 20, 255])

    # 초록색 색상 검출 영역 설정
    # lower_green = np.array([125, 0, 245])
    # upper_green = np.array([160, 15, 255])
   
    # 초록색 영역 기준으로 색상 검출
    # mask_green = cv2.inRange(hsv2, lower_green, upper_green)
    # mask_green = cv2.inRange(hsv2, lower_green, upper_green)


    # 빨간색 영역 기준으로 색상 검출
    mask_red = cv2.inRange(hsv1, lower_red, upper_red)
    # mask_red = cv2.inRange(hsv1, lower_red, upper_red)


    # 검출되는 각 색상을 ROI 영역에서 확인
    # green = cv2.bitwise_and(roi2, roi2, mask=mask_green)
    red = cv2.bitwise_and(roi1, roi1, mask=mask_red)
    # green = cv2.bitwise_and(roi2, roi2, mask=mask_green)
    # red = cv2.bitwise_and(roi1, roi1, mask=mask_red)

    if DEBUG:
        # cv2.imshow('GREEN', green)
        cv2.imshow('RED', red)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    # blur_green = cv2.medianBlur(green, 5)
    blur_red = cv2.medianBlur(red, 5)
    # cv2.imshow('blur_green', blur_green)
    # cv2.imshow('blur_red', blur_red)

    # bgr_green = cv2.cvtColor(blur_green, cv2.COLOR_HSV2BGR)
    bgr_red = cv2.cvtColor(blur_red, cv2.COLOR_HSV2BGR)
    # cv2.imshow('bgr_green', bgr_green)
    # cv2.imshow('bgr_red', bgr_red)

#    gray_green = cv2.cvtColor(bgr_green, cv2.COLOR_BGR2GRAY)
    gray_red = cv2.cvtColor(bgr_red, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_green', gray_green)
    # cv2.imshow('gray_red', gray_red)

    # 파라미터 값 설정 필요
    # circle_green = cv2.HoughCircles(gray_green, cv2.HOUGH_GRADIENT, 1, 20, param1= 35, param2= 10, minRadius= 21, maxRadius= 40)
    circle_red = cv2.HoughCircles(gray_red, cv2.HOUGH_GRADIENT, 1, 20, param1= 30, param2= 10, minRadius= 21, maxRadius= 40)

    if circle_red is not None:
        for i in range(circle_red.shape[1]):
            x, y, r = circle_red[0][i]
            print(r)
        return 'STOP'
   
    # elif circle_green is not None:
    #     for i in range(circle_green.shape[1]):
    #         x, y, r = circle_green[0][i]
    #         print(r)
    #     return 'GREEN'
   
    else:
        return 'GO'

def draw_lines(all_lines, frame):
    for line in all_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1),(x2,y2), color=(0,0,255), thickness=2)

def divide_left_right(lines):            # traffic_message = make_message(0,0)
            # comm.write(traffic_message.encode())
    global WIDTH
    global before_left, before_right
    slopes = []
    new_lines = []
    for line in lines:
        #print(line)
        x1, y1, x2, y2 = line[0]
        if x2-x1 == 0:
            slope = 0
        else:
            slope = float(y2-y1)/float(x2-x1)
       
        slopes.append(slope)
        new_lines.append(line[0])
    left_lines = []
    right_lines = []

    for i in range(len(slopes)):
        Line = new_lines[i]
        slope = slopes[i]

        x1, y1, x2, y2 = Line
        if (slope < 0 and abs(slope) > 0.5 and 0 < x1 < WIDTH * 0.75):
            left_lines.append([Line.tolist()])
            before_left = (x1 + x2) / 2
        elif (slope > 0 and abs(slope) > 0.5 and WIDTH * 0.25 < x1 < WIDTH):
            right_lines.append([Line.tolist()])
            before_right = (x1 + x2) / 2
    return left_lines, right_lines

def get_line_params(lines):
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0
    size = len(lines)
    if size == 0:
        return 0, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_sum += x1 + x2
        y_sum += y1 + y2
        if x2 - x1 == 0:
            m_sum += 0
        else:
            m_sum += float(y2-y1)/float(x2-x1)

    x_avg = float(x_sum) / float(size * 2)
    y_avg = float(y_sum) / float(size * 2)

    m = m_sum / size
    b = y_avg - m * x_avg
    return m, b

def getLinePos(lines, left=False, right=False):
    global WIDTH, HEIGHT
    global before_left, before_right
    global ROI_HEIGHT, ROI_START_HEIGHT

    m, b = get_line_params(lines)
    if (abs(m) <= sys.float_info.epsilon and abs(b) <= sys.float_info.epsilon):
        if left:
            return before_left
        elif right:
            return before_right
    y = ROI_HEIGHT
    #print("left: {} right: {} x: ".format(left, right, (y-b)/m))
    return round((y-b) / m)

def process_image(frame):
    global WIDTH, ROI_HEIGHT, ROI_START_HEIGHT, send2arduino, CANNY_LOW, CANNY_HIGH, DEBUG
    global before_left, before_right
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = (0, 0, 185)
    upper_white = (180, 111, 255)
    hsv_mask = cv2.inRange(hsv, lower_white, upper_white)
    hsv_image = cv2.bitwise_and(frame, frame, mask=hsv_mask)
    #cv2.imshow("hsv", hsv_image)
    kernel_size = 7
    blur_hsv = cv2.GaussianBlur(hsv_image,(kernel_size, kernel_size), 0)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    edge_img = cv2.Canny(np.uint8(blur_hsv), CANNY_LOW, CANNY_HIGH)
    edge_gray = cv2.Canny(np.uint8(blur_gray), CANNY_LOW, CANNY_HIGH)
    #cv2.imshow("edge", edge_img)

    roi = edge_gray[ROI_START_HEIGHT-ROI_HEIGHT:ROI_START_HEIGHT+ROI_HEIGHT, 0:WIDTH]
    #roi_src = frame[ROI_START_HEIGHT-ROI_HEIGHT:ROI_START_HEIGHT+ROI_HEIGHT, 0:WIDTH]
    #cv2.imshow('ROI', roi)
    all_lines = cv2.HoughLinesP(roi, 10, math.pi/180, HOUGH_THRESHOLD, HOUGH_LENGTH, HOUGH_GAP)
    #print(all_lines)
    if DEBUG:
        # cv2.imshow("grey", edge_gray)
        # cv2.imshow("ROI", roi)
        cv2.imshow("edge", edge_img)
        cv2.imshow("roi", roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
           
    if all_lines is None:
        return before_left, before_right
    elif len(all_lines) >= 5:
     #   return before_left, before_right
         return before_left, before_right + 30 # 우측으로 추가적인 조절 필요
   
    #draw_lines(all_lines, roi_src)
    # elif len(all_lines) > 20 :
    #     return before_left, before_right
    #cv2.imshow("src", roi_src)
    left_lines, right_lines = divide_left_right(all_lines)
    #draw_lines(left_lines, roi_src)
    #cv2.imshow("left", roi_src)
    #print(left_lines, right_lines)

    lpos = getLinePos(left_lines, left=True)
    rpos = getLinePos(right_lines, right=True)
    return lpos, rpos

def drive():
    settings()
    speed = START_SPEED
    cap = cv2.VideoCapture(2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 가로
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 세로

    print('Start')
    if not cap.isOpened():
        print('could not open webcam')
        exit()
   
    # lpf = LowPassFilter(cutoff_freq = 3, ts = 0.01)

    while cap.isOpened():
        ret, image = cap.read()
        lpos, rpos = process_image(image)
        mid = (lpos + rpos) / 2
        errorFromMid = int(round(mid - (WIDTH / 2))) + 8
        errorFromMid = round(pidControl(errorFromMid))
        if abs(errorFromMid) > 5:
            speed -= DECELERATION_STEP
            speed = max(MIN_SPEED, speed)
        else:
            speed += ACCELERATION_STEP
            speed = min(speed, MAX_SPEED)
        if errorFromMid < 0 :
            errorFromMid = max(-40, errorFromMid)
        else:
            errorFromMid = min(40, errorFromMid)
        send2arduino = make_message(errorFromMid, speed)
        comm.write(send2arduino.encode())

def drive_us():
    us_settings()
    speed = 100
    env = fl.libCAMERA()
    ch0, ch1 = env.initial_setting(capnum=2)
    ch0.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 가로
    ch0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 세로
    ch1.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 가로
    ch1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 세로
    while True:
        _, frame0, _, frame1 = env.camera_read(ch0, ch1)
        # cv2.imshow("1", frame0)
        # cv2.imshow("2", frame1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        traffic_result = traffic_recognition(frame1)
        if traffic_result == "GO":
            lpos, rpos = process_image(frame0)
            mid = (lpos + rpos) / 2
            errorFromMid = int(round(mid - (WIDTH / 2)) + 12)
            errorFromMid = round(pidControl(errorFromMid))
            if abs(errorFromMid) > 5:
                speed -= DECELERATION_STEP
                speed = max(MIN_SPEED, speed)
            else:
                speed += ACCELERATION_STEP
                speed = min(speed, MAX_SPEED)
            if errorFromMid < 0 :
                errorFromMid = max(-40, errorFromMid)
            else:
                errorFromMid = min(40, errorFromMid)
            send2arduino = make_message(errorFromMid, speed)
            time.sleep(0.1)
            print(send2arduino)
            comm.write(send2arduino.encode())
        elif traffic_result == "STOP":
            print(traffic_result)
            traffic_message = make_message(0,0)
            time.sleep(0.2)
            comm.write(traffic_message.encode())