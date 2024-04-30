# 라즈베리파이에서 실행

import cv2
import imutils
import numpy as np
from gaze_tracking import GazeTracking
from imutils.video import VideoStream
import time
import f_detector
from picamera.array import PiRGBArray
from picamera import PiCamera

# GazeTracking 초기화
gaze = GazeTracking()
detector = f_detector.eye_blink_detector()
COUNTER = 0
TOTAL = 0
start_look_time = None
look_direction = None

prev_left_pupil = None
prev_right_pupil = None
center_line_position = None
center_line_position_r = None

# 카메라 초기화
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))


# 카메라 웜업 시간을 줍니다.
time.sleep(0.1)

# 비디오 스트림 반복
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    start_time = time.time()
    # 프레임 가져오기
    frame = frame.array

    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=720)

    # 시선 추적
    gaze.refresh(frame)

    # 눈동자 십자표시
    frame = gaze.annotated_frame()

    # 눈 깜빡임 감지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectangles = detector.detector_faces(gray, 0)
    boxes_face = f_detector.convert_rectangles2array(rectangles, frame)
    
    if len(boxes_face) != 0:
        # 선택된 얼굴의 좌표 가져오기
        areas = f_detector.get_areas(boxes_face)
        index = np.argmax(areas)
        rectangles = rectangles[index]
        boxes_face = np.expand_dims(boxes_face[index], axis=0)
        # 눈 깜빡임 감지
        frame = f_detector.bounding_box(frame, boxes_face)
        COUNTER, TOTAL = detector.eye_blink(gray, rectangles, COUNTER, TOTAL)
        
        # 눈 깜빡임 횟수 표시
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
        
        # 시선 추적 결과 표시
        # 좌우 반전되어 있으므로 반대로
        if gaze.is_right():
            text = "Looking left"
            look_direction = "left"
            start_look_time = time.time()
        elif gaze.is_left():
            text = "Looking right"
            look_direction = "right"
            start_look_time = time.time()
        elif gaze.is_center():
            text = "Looking center"
            look_direction = None
            start_look_time = None
            ##변경
            # 눈이 센터에 위치할 때 현재 눈동자의 좌표를 저장하여 파란색 선의 위치를 유지
            if prev_left_pupil is not None:
                center_line_position = prev_left_pupil[0]
            if prev_right_pupil is not None:
                center_line_position_r = prev_right_pupil[0]
            ##

        else:
            text = "Eyes closed"
            look_direction = None
            start_look_time = None
        
        cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
        left_pupil = gaze.pupil_left_coords() #왼쪽 동공 좌표 반환
        right_pupil = gaze.pupil_right_coords() #오른쪽 동공 좌표 반환
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) #텍스트 추가
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) #텍스트 추가


        # 눈동자 기준으로 파란색 세로 선(center일 때 위치 유지) 그리기
        if center_line_position is not None:
            cv2.line(frame, (int(center_line_position), 0), (int(center_line_position), frame.shape[0]), (255, 0, 0), 2)
        if center_line_position_r is not None:
            cv2.line(frame, (int(center_line_position_r), 0), (int(center_line_position_r), frame.shape[0]), (255, 0, 0), 2)

        # 현재 눈동자의 좌표를 저장
        if left_pupil is not None:
            prev_left_pupil = left_pupil
        if right_pupil is not None:
            prev_right_pupil = right_pupil



    # fps 계산
    end_time = time.time() - start_time
    fps = 1 / end_time
    cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)

    # 영상 출력
    cv2.imshow("Combined Result", frame)

    # 종료 키 입력시 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 다음 프레임을 위해 버퍼 비우기
    rawCapture.truncate(0)

# 카메라 스트림 정리 및 창 닫기
cv2.destroyAllWindows()
