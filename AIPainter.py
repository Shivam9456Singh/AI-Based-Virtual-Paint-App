import numpy as np
import os
import HandMotionModule as htm
import mediapipe as mp
from flask import *
import time
import cv2

app = Flask(__name__)



def Hand_Track():
    class handDetector():
        def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
            self.mode = mode
            self.maxHands = maxHands
            self.complexity = complexity
            self.detectionCon = detectionCon
            self.trackCon = trackCon

            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                            self.detectionCon, self.trackCon)
            self.mpDraw = mp.solutions.drawing_utils

            self.tipIds = [4, 8, 12, 16, 20]

        def findHands(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            # print(results.multi_hand_landmarks)

            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            return img

        def findPosition(self, img, handNo=0, draw=True):
            self.lmList = []
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (250, 128, 0), cv2.FILLED)

            return self.lmList

        def fingersUp(self):
            # first creating the list named fingers
            # and then we are checking if the tip of
            # our thumb is on right or in the left
            fingers = []

            # thumb finger
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # four fingers
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            return fingers


    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print((lmList[4]))
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1080, 800))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 127), 3)

        # cv2.imshow("Image", img)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cv2.waitKey(1)

def Draw():
    ###########
    brushThickness = 15
    eraserThickness = 100
    #######

    folderPath = "Header"
    myList = os.listdir(folderPath)
    #For Printing overlaylist
    #print(myList)
    overlayList = []

    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)

    #for printing length of overlaylist
    #print(len(overlayList))
    header = overlayList[0]
    selectColor = (220,248,255)

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    detector = htm.handDetector(detectionCon=0.85)
    xp, yp =0,0
    imgCanvas = np.zeros((720 ,1280, 3),np.uint8)

    while True:
        #1. Import img
        success, img = cap.read()
        img = cv2.flip(img,1)

        #2. find Hand marks
        img = detector.findHands(img)
        lmList = detector.findPosition(img,draw=False)


        if len(lmList)!=0:
            #print(lmList)

            #index and middle finger tip = x1,y1
            x1,y1 = lmList[8][1:]
            x2,y2 = lmList[12][1:]

        #3. check which finger are up

            fingers = detector.fingersUp()
            #print(fingers)
            #4. if selection mode - two fingers are up
            if fingers[1] and fingers[2]:
                # whenever we are having selection or drawing again it won't give line
                # will not connect to the previous coordinates .
                xp, yp = 0, 0

                #print("Selection Mode")
                #changing and selecting the brush colors
                if y1 < 125:
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        selectColor = (57,79,205)

                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        selectColor = (102,205,0)

                    elif 800 < x1 < 950:
                        header = overlayList[2]
                        selectColor = (205,197,0)

                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        selectColor = (0,0,0)

                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), selectColor, cv2.FILLED)


            #5. draw mode - index finger is up
            if fingers[1] and fingers[2]==False:
                cv2.circle(img,(x1,y1),15,selectColor,cv2.FILLED)
                #print("Drawing Mode")

                if xp==0 and yp==0:
                    xp,yp = x1,y1

                if selectColor==(0,0,0):
                    cv2.line(img, (xp, yp), (x1, y1), selectColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), selectColor, eraserThickness)

                else:
                    cv2.line(img, (xp,yp), (x1,y1),selectColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), selectColor, brushThickness)

                xp ,yp = x1,y1

                # Here if we will consider previous x and y then from staring it will create
                # a line upto the first point which will look bad
                # therefore to avoid this if we are having previous x and y as (0,0) then
                # we will assign the value (x1,y1) =(xp,yp) as our first point and the starting point
                # after reaching to first point we have to keep on updating xp and yp

    # considering the image to be created on canvas i have taken the two output screen
        # using Canvas we are detecting our motion of fingers and hand tracking is done through webcam .

      # The reason for doing binary of a image is that we want to merge two images (canvas,webcam)
        # into a single one so we are inversing the color in canvas image
        imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)

        # Here we are merging gray canvas image to the cv2 image
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,imgInv)
        img = cv2.bitwise_or(img,imgCanvas)


        img[0:125,0:1280] = header
        img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
        #cv2.imshow("Image",img)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        #this is the demostration of canvas image and web image at same time
        #cv2.imshow("Canvas",imgCanvas)
        #cv2.imshow("Inv",imgInv)
        cv2.waitKey(1)

def Finger_Count():
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    folderPath = "FingerImages"
    myList = os.listdir(folderPath)
    # print(myList)
    overlayList = []

    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        # print(f'{folderPath}/{imPath}')
        overlayList.append(image)

    # print(len(overlayList))
    pTime = 0

    detector = htm.handDetector(detectionCon=0.75)

    # mediapipe trackpoints of fingers
    tipIds = [4, 8, 12, 16, 20]

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1080, 800))
        # print(lmList)

        if len(lmList) != 0:
            fingers = []
            # for thumb checking we have taken a single track point below which thumb is closed
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # print(fingers)
            totalFingers = fingers.count(1)
            # print(totalFingers)

            h, w, c = overlayList[0].shape
            # we have taken the last element of the finger list as 6 so -1 is giving the last finger
            img[0:h, 0:w] = overlayList[totalFingers - 1]

            cv2.rectangle(img, (20, 225), (178, 425), (201, 230, 252), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (113, 198, 113), 20)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'fps:{int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 0, 255), 3)

        #cv2.imshow("Image",img)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cv2.waitKey(1)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/finger_count')
def finger_count():
    return Response(Finger_Count(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/draw')
def draw():
    return Response(Draw(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hand_track')
def hand_track():
    return Response(Hand_Track(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
