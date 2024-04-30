# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
# 눈 깜빡임을 감지하기 위한 임계값과 연속 프레임 수를 설정(민감하게 감지하거나 더 엄격하게 감지
# EYE_AR_THRESH: 눈의 종횡비(threshold)
# EYE_AR_CONSEC_FRAMES: 눈이 임계값 아래로 유지되어야 하는 연속 프레임 수

# 1. EYE_AR_THRESH (눈의 종횡비 임계값):
#    - 눈의 종횡비(threshold)는 일반적으로 눈이 완전히 닫힌 상태일 때 작아집니다. 
# 따라서 `EYE_AR_THRESH` 값을 낮추면 더 쉽게 눈 깜빡임을 감지할 수 있습니다.
# 하지만 이 값이 너무 낮으면 눈이 살짝만 감겨도 깜빡임으로 감지될 수 있습니다.
# 그러므로 이 값을 조정할 때는 주변 환경과 사용자의 특성을 고려해야 합니다.
# 예를 들어, 사용자의 눈 크기나 조명 상태에 따라서 다를 수 있습니다.

# 2. EYE_AR_CONSEC_FRAMES (연속 프레임 수):
#    - `EYE_AR_CONSEC_FRAMES`는 눈이 임계값 아래로 유지되어야 하는 연속 프레임 수를 나타냅니다.
# 이 값이 높을수록 눈 깜빡임을 더 엄격하게 감지할 수 있습니다.
# 그러나 이 값이 너무 높으면 눈이 깜빡여도 감지되지 않을 수 있습니다.
# 따라서 이 값을 조정할 때는 눈 깜빡임의 민감도와 사용자의 특성을 고려해야 합니다.

# EYE_AR_THRESH = 0.23 #baseline
EYE_AR_THRESH = 0.20 #이렇게 해도 잘 안됨... 그래도 이정도 나쁘지 않음
EYE_AR_CONSEC_FRAMES = 1

# eye landmarks
eye_landmarks = "gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat"
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
