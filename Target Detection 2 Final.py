import imutils
import cv2
from djitellopy import Tello
import math
import numpy as np
import time
import csv
from matplotlib import pyplot as plt
from threading import Thread

def KeySwitch():
    key = cv2.waitKey(1) & 0xFF
    if key == ord("w"):
        jay.land()               #############
        
        
def Corner_Point(frame):
    font = cv2.FONT_HERSHEY_COMPLEX
    #img1 = cv2.imread('/home/jayakant/Pictures/Screenshots/Redbox_Contours_thickness-6.png',1)
    #img1 = cv2.resize(img1,(960,720))
    #img1 = cv2.resize(img1,(500,450))
    img1 = frame
    hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv ', hsv) 
    #lower red
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    #upper red
    lower_red2 = np.array([170,50,50])
    upper_red2 = np.array([180,255,255]) 
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_add = cv2.add(mask,mask2)
    kernel = np.ones((3, 3), np.uint8)    # kernal 3 is best
    opening = cv2.morphologyEx(mask_add, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('opening ', opening ) 
    # Converting image to a binary image
    # ( black and white only image).
    #_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    _, threshold = cv2.threshold(opening, 110, 255, cv2.THRESH_BINARY)
    #cv2.imshow('threshold ', threshold )   
    # Detecting contours in image.
    #contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]  # find contours in the image, keeping only the one largest ones , then sort them from left-to-right
    # Going through every contours found in the image.
    Coordinate_arr = []
    for cnt in cnts :
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        # draws boundary of contours.
        #cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5) 
        cv2.drawContours(opening, [approx], 0, (0, 0, 255), 5)       
        # Used to flatted the array containing
        # the co-ordinates of the vertices.
        n = approx.ravel() 
        i = 0
        for j in n :
            if(i % 2 == 0):
                x = n[i]
                y = n[i + 1]      
                # String containing the co-ordinates.
                string = str(x) + "," + str(y)       
                if(i == 0):
                    # text on topmost co-ordinate.
                    #cv2.putText(img2, string, (x, y),font, 0.5, (255, 0, 0))
                    cv2.putText(img1, string, (x, y),font, 0.5, (255, 0, 0))
                    
                    Coordinate_arr.append(string)                                     
                else:
                    # text on remaining co-ordinates.
                    #cv2.putText(img2, string, (x, y), font, 0.5, (0, 255, 0)) 
                    cv2.putText(img1, string, (x, y), font, 0.5, (0, 255, 0))                
                    Coordinate_arr.append(string)
                              
            i = i + 1         
        #print('Coordinate_arr = ',Coordinate_arr)  
        coords = [list(map(int,i.split(','))) for i in Coordinate_arr]
        # print(type(coords))
        # print('coords == ',coords[2])
        # print('coords === ',coords[3][1])
    return frame , coords  


     
def DetectTarget(frame , status):
    X_e_m_arr = []
    Y_e_m_arr = []
    D_arr = []
    yaw_e = 0    
    # convert the frame to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 150)
    #cv2.imshow("edged", edged)
    #find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)        
        # ensure that the approximated contour is "roughly" rectangular
        if len(approx) >= 4 and len(approx) <= 6:
            #compute the bounding box of the approximated contour and use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            #result = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
            #print('w = ',w,'h = ',h)            
            aspectRatio = w / float(h)
            #print('aspectRatio = ',aspectRatio)
            #cv2.imshow("Frame_r", result)
            # compute the solidity of the original contour
            area = cv2.contourArea(c)
            hullArea = cv2.contourArea(cv2.convexHull(c))
            solidity = area / float(hullArea)
            #print('solidity = ',solidity)
            # compute whether or not the width and height, solidity, and aspect ratio of the contour falls within appropriate bounds
            keepDims = w > 25 and h > 25
            keepSolidity = solidity > 0.9
            keepAspectRatio = aspectRatio >= 0.70 and aspectRatio <= 1.8            
            if keepDims and keepSolidity and keepAspectRatio:
                # draw an outline around the target and update the status text
                cv2.drawContours(frame, [approx], -1, (0, 0, 255), 5)#cv.drawContours (image, contours, contourIdx, color, thickness = 1, 
                                                                     #lineType = cv.LINE_8, hierarchy = new cv.Mat(), maxLevel = INT_MAX, offset = new cv.Point(0, 0))
                frame , coords = Corner_Point(frame) 
                                                                  
                cv2.drawContours(gray, [approx], -1, (0, 0, 255), 5)   # thickness = 5 best
                #cv2.imshow("gray", gray)
                #cv2.imshow("frame ===", frame)
                status = "Target(s) Acquired"
                # compute the center of the contour region and draw the crosshairs
                M = cv2.moments(approx)
                (cX, cY) = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"]))
                (startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
                (startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
                cv2.line(frame, (startX, cY), (endX, cY), (255, 0, 0), 3)
                cv2.line(frame, (cX, startY), (cX, endY), (255, 0, 0), 3)
                #le = len(coords)
                m = []
                n = []
                # print('coords = ',coords)
                # print('cx = ',cX)
                for list in coords:   
                    #print('list = ', list)
                    if list[0] < cX:
                        #print('list = ', list[0])
                        m.append(list[1])
                        if 1 < len(m) < 3:
                            diff_l = abs(m[0] - m[1])
                            # print('diff_l = ', diff_l)
                            # print() 
                    if list[0] > cX:  
                        n.append(list[1])
                        if 1 < len(n) < 3:
                            diff_r = abs(n[0] - n[1])
                            # print('diff_r = ', diff_r)                                    
                try:
                    yaw_e = diff_r - diff_l 
                except:
                    yaw_e = 0                    
                # print()                          
                # X_e_p = cX - 480 # Error = Desired - Current , Error of x-direction
                # Y_e_p = cY - 360 # Error = Desired - Current , Error of y-direction
                X_e_p = -(480 - cX)   # Error = Desired - Current , Error of x-direction
                Y_e_p = (165 - cY)
                #Y_e_p = 360 - cY   # Error = Desired - Current , Error of x-direction        (960,720)
                #cv2.circle(frame , (480,360) , 5 , (0,255,0) , -1)
                cv2.circle(frame , (480,165) , 5 , (0,0,255) , -1)
                
                #cv2.circle(frame , (cX,cY) , 5 , (255,0,0) , -1)  # Centre of target
                D = (0.55 * 956)/w # Depth of object --- D = WF/P   W = marker width, F = focal length, P = apparent width in pixels  , Marker width & Height ( W  = 0.52m , H = 0.38m)
                #D2 = (0.6 * 958.2)/h
                X_e_m = (X_e_p / 956) * D
                Y_e_m = (Y_e_p / 958.2) * D
                X_e_m_arr.append(X_e_m)
                Y_e_m_arr.append(Y_e_m)
                D_arr.append(D)
                #print('c_X = ',cX,'c_Y = ',cY)
                #print('X_e_p = ',X_e_p,'Y_e_p = ',Y_e_p)
                #print('X_e_m = ',X_e_m,'Y_e_m = ',Y_e_m)
                #print(' D = ', D)
                # print('w_r = ',w,'h_r = ',h)
    # draw the status text on the frame
    cv2.putText(frame, status, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    return frame , X_e_m_arr , Y_e_m_arr , D_arr , yaw_e



def FollowTarget(jay , X_e_m , Y_e_m , D , yaw_e):
    attitude = jay.query_attitude()
    pi_de = 0
    ro_de = 0
    ya_d = 0
    try:
        #pi_degree =  jay.get_pitch() 
        pi_de = attitude['pitch']
        ro_de = attitude['roll']
        ya_d = attitude['yaw']
        pi = (3.1416*pi_de)/180
        ro = (3.1416*ro_de)/180
        #ya = (3.1416*0.001)/180
        
        #print('pi_de ==',pi_de , 'pi_degree = ',pi_degree)
    except:
        pi = 0
        ro = 0
        print()
        print('KeyError: pitch #######################################')
        print()
        #jay.land()                                      ###############################
    M1 = np.array([[math.cos(pi), 0, math.sin(pi)],
                    [0, 1, 0],
                    [-math.sin(pi), 0, math.cos(pi)]])
    M2 = np.array([[1, 0, 0],
                    [0, math.cos(ro), -math.sin(ro)],
                    [0, math.sin(ro), math.cos(ro)]])
    M3 = np.array([[0 , 0 , 1],
                   [1 , 0 , 0],
                   [0 , 1 , 0]])
    # X_e_m = np.array(X_e_m)
    # Y_e_m = np.array(Y_e_m)
    M4 = np.array([X_e_m,
                   Y_e_m,
                   [1]])
    #print("M4 = ",M4)
    [Xe , Ye, Ze] = np.array(M1.dot(M2).dot(M3).dot(M4))      # coordinate (Xe,Ye,Ze) Error  in drone camera frame
    
    # Calculate the integral error
    global integral_error_x , integral_error_y , integral_error_z , integral_error_yaw
    #integral_error_x += Xe
    integral_error_y += Ye
    integral_error_z += Ze
    integral_error_yaw += yaw_e
    #print('integral_error_y = ',integral_error_y,'integral_error_z = ',integral_error_z)
    # Calculate the derivative error
    global derivative_error_x , derivative_error_y , derivative_error_z , derivative_error_yaw
    #derivative_error_x = Xe - derivative_error_x
    derivative_error_y = Ye - derivative_error_y
    derivative_error_z = Ze - derivative_error_z
    derivative_error_yaw = yaw_e - derivative_error_yaw
    # Calculate the control signal
    #control_x = kp * Xe + ki * integral_error_x + kd * derivative_error_x
    control_y = kp_y * Ye + ki_y * integral_error_y + kd_y * derivative_error_y
    control_z = kp_z * Ze + ki_z * integral_error_z + kd_z * derivative_error_z
    if -4 < yaw_e < 4:
        control_yaw = 0
    else:
        control_yaw = kp_ya * yaw_e + ki_ya * integral_error_yaw + kd_ya * derivative_error_yaw
    
    # print()
    # print()
    # print('Y_e == ',Ye)
    # print('Z_e == ',Ze)
    # #print('D == ',D)
    print('yaw_e = ',yaw_e,'control_yaw = ',control_yaw)
    print('derivative_error_yaw  = ',derivative_error_yaw )
    # #print('derivative_error_yaw  = ',derivative_error_yaw )
    # print('pro_err = ', kp_ya * yaw_e , 'deri_error = ', kd_ya * derivative_error_yaw,'integral_error_yaw = ', ki_ya *integral_error_yaw)
    # print()
    # print()
    # print('D = ',D)
    #D = float(D[0])
    #print('D = ',D)
    if D > 1.6 and D < 1.7:
        fb = 0
    elif D < 1.60:
        fb = -10
    elif D > 1.7:
        fb = 10

    # if x == 0:
    #     up_down = 0
    #     error_x = 0    
    # print('Xe = ',Xe,'Ye = ',Ye,'Ze = ',Ze)
    # print('control_y = ',control_y,'control_z = ',control_z)
    control_y = int(control_y)
    control_z = int(control_z)
    control_yaw = int(control_yaw)
    #print('control_y int = ',control_y,'control_z = ',control_z)
    #fb = control_x
    left_right = control_y
    up_down = control_z
    yaw = control_yaw
    #left_right  = int(np.clip(left_right ,-20,20))
    # up_down   = int(np.clip(up_down ,-8,8))
    #yaw = int(np.clip(yaw ,-10,10))
    
    jay.send_rc_control(left_right , fb, up_down, yaw) ######################     go_xyz_speed(self, x, y, z, speed) 
    
    Ye = float(Ye[0])
    Ze = float(Ze[0])
    return control_y , control_z , control_yaw , pi_de , ro_de , ya_d , Ye , Ze


def Mission():
    print()
    print('go forward===================')
    #time.sleep(0.5)
    #jay.go_xyz_speed(210, 0, 0, 80) 
    jay.move_forward(215)    ######################
    #jay.move_forward(90)
    #time.sleep(0.001)
    jay.rotate_clockwise(180) 
    #time.sleep(0.001)
    #jay.go_xyz_speed(180, 0, 0, 80) 
    jay.move_forward(180)    #####################
    #jay.move_forward(70) 
    #time.sleep(0.01)
    global count_1
    count_1 += 1
    print('count_1 = ',count_1)
    

# cap = cv2.VideoCapture('/home/jayakant/Pictures/ref9.mp4')
# cap = cv2.VideoCapture(0)
jay = Tello()
jay.connect()
print("Battery Percent",jay.get_battery())
jay.streamon()
jay.takeoff()                         ############
##jay.send_up(35)              
##jay.send_rc_control(0,0,20,0) 
#jay.move_up(20)         
time.sleep(1)                         ############

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("Tello.avi", fourcc, 20.0, (960 , 720))
# Set the target position
target_x = 0  
target_y = 0 
target_z = 0
# Set the PID gains
kp_y = 58##20    ###33
ki_y = 0.33#0.81##0.04############0.042
kd_y = 4.2##5.5

kp_z = 106#40
ki_z = 0.31#0.37###0.04
kd_z = 5.1#6.6

kp_ya = 0.45#########30.5############0.45
ki_ya = 0.0001
kd_ya = 0.007
# Initialize the integral and derivative errors
integral_error_x = 0
integral_error_y = 0
integral_error_z = 0
integral_error_yaw = 0
derivative_error_x = 0
derivative_error_y = 0
derivative_error_z = 0
derivative_error_yaw = 0

tic = time.time()
data = []

# Killswitch = Thread(target = KeySwitch)
# Killswitch.start()
count = 0
count_1 = 0
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        jay.land()               #############
        break 
    # Killswitch = Thread(target = KeySwitch)
    # Killswitch.start()
    #(grabbed, frame) = cap.read()
    frame = jay.get_frame_read().frame 
    frame = cv2.resize(frame,(960,720))
    status = "No Targets"
    # check to see if we have reached the end of the video
    #if not grabbed:
        #break
    # Get the current position of the drone
    #position = jay.get_position()
    frame , X_e_m , Y_e_m , D , yaw_e = DetectTarget(frame , status)
    #D = float(D[0])
    # print('X_e_m =',X_e_m)
    # print(type(X_e_m))
    
    if bool(X_e_m):        
        D = float(D[0])
        if count == 0:            
            control_y , control_z , control_yaw , pi_de , ro_de , ya_d ,Ye , Ze = FollowTarget(jay , X_e_m , Y_e_m , D , yaw_e)
            # print('Ye = ',Ye)
            # print('Ze = ',Ze)
            #print('D = ',D)
            print()
            data.append([round(time.time()-tic,3), X_e_m[0], Y_e_m[0], control_y, control_z , control_yaw , pi_de , ro_de , ya_d , D , Ye , Ze , yaw_e])
            ##if Ye >= -0.4 and Ye <= 0.4 and Ze >= -0.4 and Ze <= 0.4 and D < 3.5:  
            if -0.02 < Ye < 0.02 and  -0.02 < Ze < 0.02 and 1.6 < D < 1.75 and -6 < yaw_e < 6 and count == 0:
            #if -0.015 < Ye < 0.015 and  -0.015 < Ze < 0.015 and 1.6 < D < 1.75 and -8 < yaw_e < 8 and count == 0:
                print('Ye = ',Ye)
                print('Ze = ',Ze)
                print('D = ',D)
                count += 1
                #time.sleep(0.1)
                print('count = ',count)
                #print('go forward===================')
                t1 = Thread(target = Mission)
                t1.start()
    
                # #time.sleep(0.5)
                # #jay.go_xyz_speed(200, 0, 0, 40) 
                # jay.move_forward(210)    ######################
                # time.sleep(0.2)
                # jay.rotate_clockwise(180) 
                # time.sleep(0.2)
                # #jay.go_xyz_speed(180, 0, 0, 40) 
                # jay.move_forward(180)    #####################
                # time.sleep(0.2)
                # print('Mission complete===================')
                # jay.land()               #####################
                # break    
    elif count_1 == 1:
        print()
        print('Mission complete===================')
        print()        
        jay.streamoff()
        jay.land()               #####################          
        break
    
    else:
        pass
    
    cv2.imshow("Frame", frame)
    out.write(frame)
#cap.release()
cv2.destroyAllWindows()  
     
logIndex = time.time()
with open('/home/jayakant/Dji/Tello-'+str(logIndex)+'.csv', 'w', newline='') as csvfile:
    fieldnames = ['time', 'X_e_m', 'Y_e_m', 'control_y', 'control_z' , 'control_yaw' , 'pi_de' , 'ro_de' ,'ya_d' , 'D' , 'Ye' , 'Ze' , 'yaw_e']
    # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer = csv.writer(csvfile)

    # write the header
    writer.writerow(fieldnames)

    # write multiple rows
    writer.writerows(data)
data = np.array(data)
time_arr = data[:, 0]
X_e_m_arr = data[:, 1]
Y_e_m_arr = data[:, 2]
yaw_e = data[:, 12]
D = data[:, 9]

plt.subplot(2, 2, 1)
plt.plot(time_arr, X_e_m_arr)
#ref_x = np.ones(shape = errorX_arr.shape) * (w_s/2)
plt.subplot(2, 2, 2)
plt.plot(time_arr, Y_e_m_arr)

plt.subplot(2, 2, 3)
plt.plot(time_arr, yaw_e)
#plt.plot(time_arr, ref_y)
plt.subplot(2, 2, 4)
plt.plot(time_arr, D)
plt.show()                         	
cv2.destroyAllWindows()
