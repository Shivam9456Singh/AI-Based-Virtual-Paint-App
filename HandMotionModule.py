import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands = 2,complexity=1, detectionCon =0.5,trackCon=0.5):
      self.mode = mode
      self.maxHands = maxHands
      self.complexity = complexity
      self.detectionCon  = detectionCon
      self.trackCon = trackCon

      self.mpHands = mp.solutions.hands
      self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.complexity,
                                      self.detectionCon,self.trackCon)
      self.mpDraw = mp.solutions.drawing_utils

      self.tipIds = [4,8,12,16,20]

    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    def findPosition(self,img,handNo=0,draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
              #print(id,lm)
              h, w, c = img.shape
              cx, cy = int(lm.x*w), int(lm.y*h)
              #print(id, cx, cy)
              self.lmList.append([id,cx,cy])
              if draw:
                cv2.circle(img,(cx,cy),5,(250, 128, 0),cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        # first creating the list named fingers
        # and then we are checking if the tip of
        # our thumb is on right or in the left
        fingers = []

        #thumb finger
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # four fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers




def main():
    pTime = 0
    cTime = 0
    # try:
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("rtsp://192.168.1.2:5000/out.h264")
    # cap = cv2.VideoCapture(
    #     "udpsrc port=5000 ! application/x-rtp,media=video,encoding-name=H264 ! queue ! rtpjitterbuffer latency=500 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink drop=1",
    #     cv2.CAP_GSTREAMER)

    cap = None

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print((lmList[4]))


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 127), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)






if __name__ =="__main__":
    main()
