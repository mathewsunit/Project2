import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

FACE_CASCADE_FAST = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
MOUTH_CASCADE = cv2.CascadeClassifier('ExternalClassifiers/Mouth.xml')
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')

SCALE_FACTOR_FACES_FAST = 1.02
MIN_NEIGHBOURS_FACES_FAST = 3
FLAG_FACES_FAST = 0|cv2.CASCADE_SCALE_IMAGE
MIN_SIZE_FACES_FAST = (40, 40)

SCALE_FACTOR_FACES = 1.02
MIN_NEIGHBOURS_FACES = 12
FLAG_FACES = cv2.CASCADE_SCALE_IMAGE
MIN_SIZE_FACES = (45, 45)

SCALE_FACTOR_MOUTH = 1.15
MIN_NEIGHBOURS_MOUTH = 3
FLAG_MOUTH = 0
MIN_SIZE_MOUTH = (20, 20)

def detectShush(frame, location, ROI):
    mouths = MOUTH_CASCADE.detectMultiScale(ROI, SCALE_FACTOR_MOUTH, MIN_NEIGHBOURS_MOUTH, FLAG_MOUTH, MIN_SIZE_MOUTH)
    for (mx, my, mw, mh) in mouths:
        mx += location[0]
        my += location[1]
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
    return len(mouths) == 0

def pre_process_face(frame):
    hist, bins = np.histogram(frame.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    gray_frame = cdf[frame]
    return gray_frame

def pre_process(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.bilateralFilter(frame,9,100,100)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame = clahe.apply(frame)
    return frame

def detect(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE_FAST.detectMultiScale(
        gray_frame, SCALE_FACTOR_FACES, MIN_NEIGHBOURS_FACES, FLAG_FACES, MIN_SIZE_FACES)
    detected = 0
    for (x, y, w, h) in faces:
        x1 = x
        h2 = int(h / 2)
        y1 = y + h2
        mouthROI = gray_frame[y1:y1 + h2, x1:x1 + w]
        if detectShush(frame, (x1, y1), mouthROI):
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected

def detect_fast(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = pre_process(gray_frame)
    faces = FACE_CASCADE_FAST.detectMultiScale(gray_frame,SCALE_FACTOR_FACES_FAST,MIN_NEIGHBOURS_FACES_FAST,FLAG_FACES_FAST,MIN_SIZE_FACES_FAST)
    detected = 0
    for (x, y, w, h) in faces:
        # ROI for mouth
        x1 = x
        h2 = int(h / 2)
        y1 = y + h2
        mouthROI = gray_frame[y1:y1 + h2, x1:x1 + w]
        if detectShush(frame, (x1, y1), mouthROI):
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected

def resize(frame):
    try:
        r = 320.0 / frame.shape[1]
        dim = (320, int(frame.shape[0] * r))
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return resized
    except AttributeError:
        print("Unable to read")


def run_on_folder(cascade1, cascade2, folder):
    if (folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    windowName = None
    totalCnt = 0
    for f in files:
        img = cv2.imread(f)
        if (type(img) == None): continue
        img  = resize(img)
        if type(img) is np.ndarray:
            lCnt = detect(img)
            totalCnt += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCnt


def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()
    windowName = "Live Video"
    showframe = True
    while (showframe):
        ret, frame = videocapture.read()
        if not ret:
            print("Can't capture frame")
            break
        detect(frame)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showframe = False

    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1 +
              "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('ExternalClassifiers/Mouth.xml')

    if (len(sys.argv) == 2):  # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, mouth_cascade, folderName)
        print("Total of ", detections, "detections")
    else:  # no arguments
        runonVideo(face_cascade, mouth_cascade)