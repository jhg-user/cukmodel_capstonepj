# """
# Demonstration of the GazeTracking library.
# Check the README.md for complete documentation.
# """

# #model 경로 : cukmodel\gaze_tracking\trained_models\shape_predictor_68_face_landmarks.dat

# import cv2
# from gaze_tracking import GazeTracking

# gaze = GazeTracking()
# webcam = cv2.VideoCapture(0)

# while True:
#     # We get a new frame from the webcam
#     _, frame = webcam.read()

#     # We send this frame to GazeTracking to analyze it
#     gaze.refresh(frame)

#     frame = gaze.annotated_frame()
#     text = ""

#     if gaze.is_blinking():
#         text = "Blinking"
#     elif gaze.is_right():
#         text = "Looking right"
#     elif gaze.is_left():
#         text = "Looking left"
#     elif gaze.is_center():
#         text = "Looking center"

#     cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

#     left_pupil = gaze.pupil_left_coords()
#     right_pupil = gaze.pupil_right_coords()
#     cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#     cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

#     cv2.imshow("Demo", frame)

#     if cv2.waitKey(1) == 27:
#         break
   
# webcam.release()
# cv2.destroyAllWindows()


# # eye_blink_detection.py
# # 웹캠에서 눈 깜빡임 횟수 측정 가능
# # 눈 깜빡임 횟수에 따른 졸음 인식 되도록 코드 수정해야함
# # 눈 깜빡임 횟수(이 코드)랑 눈 감은 시간(sleepymodel.py) 합쳐서 졸음 인식 판단하면 좋을듯함
# """
# 파이썬 버전 3.7, 3.8이 안정적 -> dlib 사용하려면 64비트 필요
# 파이썬 설치 후
# python -m pip install --upgrade pip
# pip install numpy
# python -m pip install opencv-python
# pip install dlib
# pip install scipy
# pip install imutils
# """
# # 코드 분석

# from imutils.video import VideoStream
# import cv2
# import time
# import f_detector
# import imutils
# import numpy as np

# # instancio detector
# detector = f_detector.eye_blink_detector()
# # iniciar variables para el detector de parapadeo
# COUNTER = 0
# TOTAL = 0

# # ----------------------------- video -----------------------------
# #ingestar data
# vs = VideoStream(src=0).start()
# while True:
#     star_time = time.time()
#     im = vs.read()
#     im = cv2.flip(im, 1)
#     im = imutils.resize(im, width=720)
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     # detectar_rostro    
#     rectangles = detector.detector_faces(gray, 0)
#     boxes_face = f_detector.convert_rectangles2array(rectangles,im)
#     if len(boxes_face)!=0:
#         # seleccionar el rostro con mas area
#         areas = f_detector.get_areas(boxes_face)
#         index = np.argmax(areas)
#         rectangles = rectangles[index]
#         boxes_face = np.expand_dims(boxes_face[index],axis=0)
#         # blinks_detector
#         COUNTER,TOTAL = detector.eye_blink(gray,rectangles,COUNTER,TOTAL)
#         # agregar bounding box
#         img_post = f_detector.bounding_box(im,boxes_face,['blinks: {}'.format(TOTAL)])
#     else:
#         img_post = im 
#     # visualizacion 
#     end_time = time.time() - star_time    
#     FPS = 1/end_time
#     cv2.putText(img_post,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
#     cv2.imshow('blink_detection',img_post)
#     if cv2.waitKey(1) &0xFF == ord('q'):
#         break


import cv2
import imutils
import numpy as np
from gaze_tracking import GazeTracking
from imutils.video import VideoStream
import time
import f_detector

# GazeTracking 초기화
gaze = GazeTracking()
detector = f_detector.eye_blink_detector()
COUNTER = 0
TOTAL = 0
# 비디오 스트림 시작
vs = VideoStream(src=0).start()

# 결과 영상 출력을 위한 창 생성
cv2.namedWindow("Combined Result")

# 무한 루프
while True:
    start_time = time.time()
    
    # 프레임 읽기
    frame = vs.read()
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
        elif gaze.is_left():
            text = "Looking right"
        elif gaze.is_center():
            text = "Looking center"
        else:
            text = "Blink"
        
        cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
        left_pupil = gaze.pupil_left_coords() #왼쪽 동공 좌표 반환
        right_pupil = gaze.pupil_right_coords() #오른쪽 동공 좌표 반환
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) #텍스트 추가
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1) #텍스트 추가

    # fps 계산
    end_time = time.time() - start_time
    fps = 1 / end_time
    cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)

    # 영상 출력
    cv2.imshow("Combined Result", frame)

    # 종료 키 입력시 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 스트림 정리 및 창 닫기
vs.stop()
cv2.destroyAllWindows()


