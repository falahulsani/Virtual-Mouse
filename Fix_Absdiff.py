import cv2
import numpy as np
import imutils
import math
import mouse
import wx


def printThreshold(thr):
    print("! Changed Threshold to " + str(thr))

#-------------------------------------------------------------------------------
# Function - Mencari rata - rata pada jalannya background
#-------------------------------------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # inisialisasi background
    if bg is None:
        bg = image.copy().astype("float")
        return
    # menhitung rata - rata tertimbang, diakumulasi dan memperbarui background
    cv2.accumulateWeighted(image, bg, accumWeight)

#-------------------------------------------------------------------------------
# Function - Untuk mencari daerah tangan pada gambar
#-------------------------------------------------------------------------------
def segment(image):
    global bg
    # Mencari absolute difference (perbedaan mutlak) antara background dan current frame (background saat ini)
    diff = cv2.absdiff(bg.astype("uint8"), image)

    return diff

def mark_hand_center(res):
    max_d=0
    pt=(0,0)
    x,y,w,h = cv2.boundingRect(res)
    for ind_y in range(int(y+0.3*h),int(y+0.8*h)): #around 0.25 to 0.6 region of height (Faster calculation with ok results)
        for ind_x in range(int(x+0.3*w),int(x+0.6*w)): #around 0.3 to 0.6 region of width (Faster calculation with ok results)
            dist= cv2.pointPolygonTest(res,(ind_x,ind_y),True)
            if(dist>max_d):
                max_d=dist
                pt=(ind_x,ind_y)
    if(max_d>radius_thresh*drawing.shape[1]):
        thresh_score=True
        cv2.circle(drawing,pt,int(max_d),(255,0,0),2)
        cv2.circle(drawing,pt,3,(255,0,0),-1)
    else:
        thresh_score=False
    return pt,max_d,thresh_score

def mark_fingers(hull,pt,radius):
    global first_iteration
    global finger_ct_history
    finger=[(hull[0][0][0],hull[0][0][1])]
    j=0

    cx = pt[0]
    cy = pt[1]

    for i in range(len(hull)):
        dist = np.sqrt((hull[-i][0][0] - hull[-i+1][0][0])**2 + (hull[-i][0][1] - hull[-i+1][0][1])**2)
        if (dist>18):
            if(j==0):
                finger=[(hull[-i][0][0],hull[-i][0][1])]
            else:
                finger.append((hull[-i][0][0],hull[-i][0][1]))
            j=j+1

    temp_len=len(finger)
    i=0
    while(i<temp_len):
        dist = np.sqrt( (finger[i][0]- cx)**2 + (finger[i][1] - cy)**2)
        if(dist<finger_thresh_l*radius or dist>finger_thresh_u*radius or finger[i][1]>cy+radius):
            finger.remove((finger[i][0],finger[i][1]))
            temp_len=temp_len-1
        else:
            i=i+1

    temp_len=len(finger)
    if(temp_len>5):
        for i in range(1,temp_len+1-5):
            finger.remove((finger[temp_len-i][0],finger[temp_len-i][1]))

    palm=[(cx,cy),radius]

    if(first_iteration):
        finger_ct_history[0]=finger_ct_history[1]=len(finger)
        first_iteration=False
    else:
        finger_ct_history[0]=0.34*(finger_ct_history[0]+finger_ct_history[1]+len(finger))

    if((finger_ct_history[0]-int(finger_ct_history[0]))>0.8):
        finger_count=int(finger_ct_history[0])+1
    else:
        finger_count=int(finger_ct_history[0])

    finger_ct_history[1]=len(finger)

    count_text="FINGERS:"+str(finger_count)
    cv2.putText(drawing,count_text,(int(0.05*drawing.shape[1]),int(0.10*drawing.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)

    for k in range(len(finger)):
        angle=0
        if k+1 < len(finger):
            lineGeom1=[(0,0),(0,0)]
            lineGeom2=[(0,0),(0,0)]
            try:
                lineGeom1=[(finger[k][0],finger[k][1]), (cx,cy)]
                lineGeom2=[(finger[k+1][0],finger[k+1][1]), (cx,cy)]
            except:
                pass
            print('lineGeom1', lineGeom1)
            print('lineGeom2', lineGeom2)
            angle=AngleFromLines([lineGeom1, lineGeom2])
            cv2.putText(drawing, str(angle), (finger[k][0], finger[k][1]), cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,255),1,8)
            print('angle', angle)

        cv2.circle(drawing,finger[k],10,255,2)
        cv2.line(drawing,finger[k],(cx,cy),255,2)
    return finger,palm

def GetAngle (p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dX = x2 - x1
    dY = y2 - y1
    rads = math.atan2 (-dY, dX) #wrong for finding angle/declination?
    return math.degrees (rads)

def LineToXYs (line): #return first and last coordinates
    firstX, firstY = (line[0][0], line[0][1])
    lastX, lastY = (line[1][0], line[1][1])
    return [(firstX, firstY), (lastX, lastY)]

def AngleFromLines (lines):
    global angle
    #print('lines', lines)
    #lines is a python list of line geometries that share a vertex
    for line1 in lines:
        for line2 in lines:
            if line1 == line2:
                continue
            line1StPnt, line1EndPnt = LineToXYs (line1) #get start and end xys for first line
            line2StPnt, line2EndPnt  = LineToXYs (line2) #get start and end xys for second line
            angle1 = GetAngle (line1StPnt, line1EndPnt) #calc angle - Doesn't work
            angle2 = GetAngle (line2StPnt, line2EndPnt) #calc angle - Doesn't work
            print ("first line start and end coordinates:", line1StPnt, line1EndPnt)
            print ("second line start and end coordinates:", line2StPnt, line2EndPnt)
            print ("angle 1:", angle1)
            print ("angle 2:", angle2)
            angle = abs (angle1 - angle2)
            print ("angle between lines:", angle)

    return angle

isBgCaptured = 0
bg = None
contourAxisangle = 0
angle = 0
app = wx.App(False)
(sx,sy)= wx.GetDisplaySize()
#(camx,camy)=(310,200)
mouseOn = 0
state = False

if __name__ == "__main__":
    accumWeight = 0.5

    camera = cv2.VideoCapture(0)
    camera.set(10, 200)

    # Trackbar Untuk Mengatur Threshold
    cv2.namedWindow("Settings")
    cv2.resizeWindow("Settings", 350, 50)
    cv2.createTrackbar('Threshold', 'Settings', 30, 255, printThreshold)

    # Mengatur ROI
    top, right, bottom, left = 5, 250, 330, 630

    first_iteration=True
    finger_ct_history=[0,0]
    finger_thresh_l=2.0
    finger_thresh_u=3.8
    radius_thresh=0.04 # factor of width of full frame
    while camera.isOpened():
        ret, frame = camera.read()

        threshold = cv2.getTrackbarPos('Threshold', 'Settings')

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]
        camx = width - 360
        camy = height - 280
        roi = frame[top:bottom, right:left]
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        drawing = np.zeros(roi.shape, np.uint8)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 50, 100)
        gray = cv2.blur(gray.copy(), (5,5))

        cv2.imshow('original', clone)

        if isBgCaptured == 1:
            img = segment(gray)

            thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]

            thresh = cv2.dilate(thresh.copy(),cv2.getStructuringElement(cv2.MORPH_DILATE,(2,2)),iterations=2)

            # open = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
            # close = cv2.morphologyEx(open,cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))

            #cv2.imshow('Threshold',thresh)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # contours = sorted(contours, key=lambda x:cv2.contourArea(x))
            length = len(contours)
            minArea = 500
            if length > 0:
                for i in range(length):
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > minArea:
                        minArea = area
                        ci = i
                        res = contours[ci]
                        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)

                        extTop = tuple(res[res[:, :, 1].argmin()][0])

                        pt,max_d,thresh_score = mark_hand_center(res)
                        chull = cv2.convexHull(res)
                        finger,palm=mark_fingers(chull,pt,max_d)
                        x = extTop[0]*sx/camx
                        y = extTop[1]*sy/camy

                        state=''
                        if len(finger) == 1:
                            if state==True:
                                state=False
                                mouse.release(button='left')
                                cv2.putText(drawing,'DROP',(int(0.05*drawing.shape[1]),int(0.2*drawing.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
                            else:
                                mouse.move(x,y,absolute=True, duration=.1)
                                cv2.putText(drawing,'MOVE',(int(0.05*drawing.shape[1]),int(0.2*drawing.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
                        if len(finger) == 2:
                            if angle > 20 and angle < 40:
                                if state==False:
                                    state=True
                                    mouse.press(button='left')
                                    cv2.putText(drawing,'Left Click',(int(0.05*drawing.shape[1]),int(0.2*drawing.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
                                    cv2.waitKey(50)
                            elif angle > 100 and angle < 130:
                                mouse.click(button='right')
                                cv2.putText(drawing,'Right Click',(int(0.05*drawing.shape[1]),int(0.2*drawing.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
                                cv2.waitKey(50)
                        elif len(finger) == 3:
                            cv2.putText(drawing, 'Scroll Up',(int(0.05 * drawing.shape[1]), int(0.2 * drawing.shape[0])),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)
                        elif len(finger) == 4:
                            cv2.putText(drawing, 'Scroll Down',(int(0.05 * drawing.shape[1]), int(0.2 * drawing.shape[0])),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)
                        else:
                            pass
            cv2.imshow('Output', drawing)

        k = cv2.waitKey(10)
        if k == ord('q'):
            break
        elif k == ord('b'):
            run_avg(gray, accumWeight)
            isBgCaptured = 1
            print('!!!Background Captured!!!')

    cv2.destroyAllWindows()
    camera.release()
