import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('Videos/7.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)

while True:
    success, img = cap.read()
    img = cv.resize(img, (680, 680), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)  # resizing original video

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results= faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box #bounding box from the class "mediapipe.python.solution_base.SolutionOutputs"
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                    int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(img, bbox, (255,0,255), 2)
            cv.putText(img, f'FPS: {int(detection.score[0]*100)}%',
                       (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv.imshow('Image', img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()