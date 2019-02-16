import cv2
import numpy as np
import pyautogui


class PointTracker:

    def __init__(self):
        pyautogui.PAUSE = 0
        self.mouseMode = True
        self.scrollMode = False

        self.isHistCreated = False
        self.traversePoints = []

        screenSize = pyautogui.size()
        self.screenSizeX = screenSize[0]
        self.screenSizeY = screenSize[1]

    def createHistogram(self, frame):
        rows, cols, _ = frame.shape
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros([20, 20, 3], dtype=hsvFrame.dtype)

        y0, x0 = int(0.5*rows), int(0.2*cols)
        roi = hsvFrame[y0:y0 + 20, x0:x0 + 20, :]

        hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    def drawLocker(self, frame):
        rows, cols, _ = frame.shape

        y0, x0 = int(0.5*rows), int(0.2*cols)
        cv2.rectangle(frame, (x0, y0), (x0 + 20, y0 + 20), (0, 255, 0), 1)

    def detect(self, frame, hist):
        histMask = self.histMasking(frame, hist)
        cv2.imshow("histMask", histMask)
        contours = self.getContours(histMask)
        maxContour = self.getMaxContours(contours)

        centroid = self.getCentroid(maxContour)
        cv2.circle(frame, centroid, 5, [255, 0, 0], -1)

        if maxContour is not None:
            convexHull = cv2.convexHull(maxContour, returnPoints=False)
            defects = cv2.convexityDefects(maxContour, convexHull)
            farthestPoint = maxContour[maxContour[:,:,1].argmin()][0]
            print("Centroid: {}, Farthest point: {}".format(centroid, farthestPoint))
            if farthestPoint is not None:
                # Reduce noise in farthestPoint
                if len(self.traversePoints) > 0:
                    if abs(farthestPoint[0] - self.traversePoints[-1][0]) < 10:
                        farthestPoint[0] = self.traversePoints[-1][0]
                    if abs(farthestPoint[1] - self.traversePoints[-1][1]) < 10:
                        farthestPoint[1] = self.traversePoints[-1][1]
                farthestPoint = tuple(farthestPoint)
                print(farthestPoint)

                cv2.circle(frame, farthestPoint, 5, [0, 0, 255], -1)

                if len(self.traversePoints) < 10:
                    self.traversePoints.append(farthestPoint)
                else:
                    self.traversePoints.pop(0)
                    self.traversePoints.append(farthestPoint)

            self.drawPath(frame, self.traversePoints)
            self.execute(farthestPoint, frame)

    def execute(self, farthestPoint, frame):
        if self.mouseMode:
            targetX = farthestPoint[0]
            targetY = farthestPoint[1]
            pyautogui.moveTo(targetX*self.screenSizeX/frame.shape[1], targetY*self.screenSizeY/frame.shape[0])
        elif self.scrollMode:
            if len(self.traversePoints) >= 2:
                movedDistance = self.traversePoints[-1][1] - self.traversePoints[-2][1]
                pyautogui.scroll(-movedDistance/2)

    def drawPath(self, frame, traversePoints):
        for i in range(1, len(self.traversePoints)):
            thickness = int((i + 1)/2)
            cv2.line(frame, traversePoints[i-1], traversePoints[i], [0, 0, 255], thickness)

    def histMasking(self, frame, hist):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        # thresh = cv2.dilate(thresh, kernel, iterations=5)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

        thresh = cv2.merge((thresh, thresh, thresh))
        return cv2.bitwise_and(frame, thresh)

    def getCentroid(self, contour):
        moment = cv2.moments(contour)
        if moment['m00'] != 0:
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            return cx, cy
        else:
            return None

    def getMaxContours(self, contours):
        if len(contours) > 0:
            maxIndex = 0
            maxArea = 0

            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)

                if area > maxArea:
                    maxArea = area
                    maxIndex = i
            return contours[maxIndex]

    def getContours(self, histMask):
        grayHistMask = cv2.cvtColor(histMask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grayHistMask, 0, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def startDetecting(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            k = cv2.waitKey(1) & 0xFF

            if k == ord("z"):
                self.isHistCreated = True
                hist = self.createHistogram(frame)
            elif k == ord("m"):
                self.mouseMode = True
                self.scrollMode = False
            elif k == ord("n"):
                self.scrollMode = True
                self.mouseMode = False
            elif k == ord("c"):
                pyautogui.click()

            if self.isHistCreated:
                self.detect(frame, hist)
            else:
                self.drawLocker(frame)

            cv2.imshow("Output", frame)
            if k == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PointTracker()
    detector.startDetecting()
