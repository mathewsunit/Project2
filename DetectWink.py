import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import sys

SCALE_FACTOR_FACES_FAST = 1.02
MIN_NEIGHBOURS_FACES_FAST = 3
FLAG_FACES_FAST = 0|cv2.CASCADE_SCALE_IMAGE
MIN_SIZE_FACES_FAST = (22, 22)

SCALE_FACTOR_WINK_FAST = 1.12
MIN_NEIGHBOURS_WINK_FAST = 4
FLAG_WINK_FAST = 0|cv2.CASCADE_SCALE_IMAGE
MIN_SIZE_WINK_FAST = (20, 20)

SCALE_FACTOR_FACES = 1.02
MIN_NEIGHBOURS_FACES = 12
FLAG_FACES = cv2.CASCADE_SCALE_IMAGE
MIN_SIZE_FACES = (45, 45)

SCALE_FACTOR_WINK = 1.12
MIN_NEIGHBOURS_WINK = 3
FLAG_WINK = cv2.CASCADE_SCALE_IMAGE
MIN_SIZE_WINK = (12, 12)

# FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
EYE_CASCADE_FAST = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')
FACE_CASCADE_FAST = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
FACE_CASCADE = cv2.CascadeClassifier('ExternalClassifiers/haarcascade_frontalface_alt2.xml')
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')
# EYE_CASCADE = cv2.CascadeClassifier('ExternalClassifiers/haarcascade eye.xml')

def detect_wink(frame, location, ROI):
    eyes = EYE_CASCADE.detectMultiScale(ROI, SCALE_FACTOR_WINK, MIN_NEIGHBOURS_WINK, FLAG_WINK, MIN_SIZE_WINK)
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return len(eyes) == 1

def detect_wink_fast(frame, location, ROI):
    eyes = EYE_CASCADE_FAST.detectMultiScale(ROI, SCALE_FACTOR_WINK_FAST, MIN_NEIGHBOURS_WINK_FAST, FLAG_WINK_FAST, MIN_SIZE_WINK_FAST)
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return len(eyes) == 1

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
    # gray_frame = pre_process(gray_frame)
    faces = FACE_CASCADE.detectMultiScale(gray_frame,SCALE_FACTOR_FACES,MIN_NEIGHBOURS_FACES,FLAG_FACES,MIN_SIZE_WINK)
    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y + h, x:x + w]
        if detect_wink(frame, (x, y), faceROI):
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected

def detect_fast(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_frame = pre_process(gray_frame)
    faces = FACE_CASCADE_FAST.detectMultiScale(gray_frame,SCALE_FACTOR_FACES_FAST,MIN_NEIGHBOURS_FACES_FAST,FLAG_FACES_FAST,MIN_SIZE_FACES_FAST)
    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y + h, x:x + w]
        if detect_wink_fast(frame, (x, y), faceROI):
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected

def run_on_folder(folder):
    if (folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    windowName = None
    totalCount = 0
    for f in files:
        if(f.endswith(".gif")): continue
        img = cv2.imread(f, 1)
        if (type(img) == None): continue
        img  = resize(img)
        if type(img) is np.ndarray:
            lCnt = detect(img)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount

def resize(frame):
    r = 320.0 / frame.shape[1]
    dim = (320, int(frame.shape[0] * r))
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized

def run_on_webcam():
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
        detect_fast(frame)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
    videocapture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()
    if (len(sys.argv) == 2):  # one argument
        input = sys.argv[1]
        detections = run_on_folder(input)
        print("Total of ", detections, "detections")
    else:
        run_on_webcam()