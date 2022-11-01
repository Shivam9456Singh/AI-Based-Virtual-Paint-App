import cv2
import time
import os
import HandMotionModule as htm
from flask import *

app = Flask(__name__)

def Finger_Count():
        wCam , hCam = 640, 480
        cap = cv2.VideoCapture(0)
        cap.set(3,wCam)
        cap.set(4,hCam)

        folderPath = "FingerImages"
        myList = os.listdir(folderPath)
        #print(myList)
        overlayList = []

        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            #print(f'{folderPath}/{imPath}')
            overlayList.append(image)

        #print(len(overlayList))
        pTime = 0

        detector = htm.handDetector(detectionCon=0.75)

        #mediapipe trackpoints of fingers
        tipIds = [4,8,12,16,20]


        while True:
            success, img = cap.read()
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)
            img = cv2.flip(img, 1)
            #print(lmList)


            if len(lmList)!=0:
                fingers = []
                # for thumb checking we have taken a single track point below which thumb is closed
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                for id in range(1,5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                #print(fingers)
                totalFingers = fingers.count(1)
                #print(totalFingers)





                h, w, c = overlayList[0].shape
                # we have taken the last element of the finger list as 6 so -1 is giving the last finger
                img[0:h,0:w] =  overlayList[totalFingers-1]

                cv2.rectangle(img,(20,225),(178,425),(201,230,252),cv2.FILLED)
                cv2.putText(img, str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,
                            10,(113,198,113),20)

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime

            cv2.putText(img,f'fps:{int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,
                        3,(0,0,255),3)

            #cv2.imshow("Image",img)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            cv2.waitKey(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(Finger_Count(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
