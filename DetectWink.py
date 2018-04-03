import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

SCALE = 1.12
MIN_NEIGHBORS = 4
FLAG = 0 | cv2.CASCADE_SCALE_IMAGE
MIN_SIZE_FACE = (20, 20)
MIN_SIZE_WINK = (18, 12)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('ExternalClassifiers/haarcascade_frontalface_alt_tree.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier('ExternalClassifiers/parojosG.xml')
left_cascade = cv2.CascadeClassifier('ExternalClassifiers/ojoI.xml')
right_cascade = cv2.CascadeClassifier('ExternalClassifiers/ojoD.xml')

def detectWink(frame, location, ROI, cascade):
    eyes = cascade.detectMultiScale(ROI, SCALE, MIN_NEIGHBORS, FLAG, MIN_SIZE_WINK)
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return len(eyes) == 1  # number of eyes is one

def detectWinkL(frame, location, ROI):
    eyes = left_cascade.detectMultiScale(ROI, SCALE, MIN_NEIGHBORS, FLAG, MIN_SIZE_WINK)
    # if not eyes:
    #     return False
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return True

def detectWinkR(frame, location, ROI):
    eyes = right_cascade.detectMultiScale(ROI, SCALE, MIN_NEIGHBORS, FLAG, MIN_SIZE_WINK)
    # if not eyes:
    #     return False
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return True

def detect(frame, faceCascade, eyesCascade):
    faces = faceCascade.detectMultiScale(frame, SCALE, MIN_NEIGHBORS, FLAG, MIN_SIZE_FACE)
    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = frame[y:y + h, x:x + w]
        if detectWinkL(frame, (x, y), faceROI) ^ detectWinkR(frame, (x, y), faceROI):
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected

# def detect(frame, faceCascade, eyesCascade):
#     faces = faceCascade.detectMultiScale(frame, SCALE, MIN_NEIGHBORS, FLAG, MIN_SIZE_FACE)
#     detected = 0
#     for f in faces:
#         x, y, w, h = f[0], f[1], f[2], f[3]
#         faceROI = frame[y:y + h, x:x + w]
#         if detectWink(frame, (x, y), faceROI, eyesCascade):
#             detected += 1
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         else:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     return detected

def run_on_folder(cascade1, cascade2, folder):
    if (folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f, 1)
        if type(img) is np.ndarray:
            img = cv2.resize(img, (450, 400), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lCnt = detect(img, cascade1, cascade2)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount


def runonWebCam(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while (showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False

    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()

def runonVideo(face_cascade, eyes_cascade, fileLocation):
    videocapture = cv2.VideoCapture(fileLocation)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while (showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False

    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    # if len(sys.argv) != 1 and len(sys.argv) != 2:
    #     print(sys.argv[0] + ": got " + len(sys.argv) - 1
    #           + "arguments. Expecting 0 or 1:[image-folder]")
    #     exit()

    # load pretrained cascades

    # folderName = sys.argv[1]
    detections = run_on_folder(face_cascade, eye_cascade, "Pictures/people+wink")
    print("Total of ", detections, "detections")

    # if (len(sys.argv) == 2):  # one argument
    #     folderName = sys.argv[1]
    #     detections = run_on_folder(face_cascade, eye_cascade, folderName)
    #     print("Total of ", detections, "detections")
    # else:  # no arguments
    #     runonVideo(face_cascade, eye_cascade, "InputFiles/BTS-Wink-Compilation.mp4")
