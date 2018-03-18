import cv2
import imutils
from sklearn.metrics import pairwise
import numpy as np
from subprocess import check_output,Popen,STDOUT,PIPE
import math

background=None

#TO find running aveages
def running_avg(image,alpha):
	global background
	if background is None:
		background = image.copy().astype("float")
		return
	cv2.accumulateWeighted(image,background,alpha)

def segment(image,threshold=25):
	global background
	diff = cv2.absdiff(background.astype("uint8"),image)

	thresholded = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)[1]
	#print(thresholded)

	(_,contours,_) = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	if(len(contours)) == 0:
		return
	else:
		segmented=max(contours,key=cv2.contourArea)  #get max contour area
		return (thresholded,segmented)

def count(thresholded,segmented):

	convex_hull = cv2.convexHull(segmented)
	extreme_top = tuple(convex_hull[convex_hull[:,:,1].argmin()][0])
	extreme_bottom = tuple(convex_hull[convex_hull[:,:,1].argmax()][0])
	extreme_left = tuple(convex_hull[convex_hull[:,:,0].argmin()][0])
	extreme_right = tuple(convex_hull[convex_hull[:,:,0].argmax()][0])

	centre_x=(extreme_left[0]+extreme_right[0])/2
	centre_y=(extreme_top[1]+extreme_bottom[1])/2

	distance = pairwise.euclidean_distances([(centre_x,centre_y)],Y=[extreme_left,extreme_right,extreme_top,extreme_bottom])[0]
	max_distance=distance[distance.argmax()]

	radius=int(0.8*max_distance)

	circumference=(2*np.pi*radius)

	circular_roi = np.zeros(thresholded.shape[:2],dtype="uint8")
	cv2.circle(circular_roi,(int(centre_x),int(centre_y)),radius,255,1)
	circular_roi=cv2.bitwise_and(thresholded,thresholded,mask=circular_roi)

	(_,contours,_) = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	count=0

	for c in contours:
		(x,y,w,h) = cv2.boundingRect(c)
		if((centre_y+(centre_y*0.25)) > (y+h)) and ((circumference*0.25)>c.shape[0]):	#if point is palm and radius is less than 25% of circumference
			count+=1

	return count

if __name__=='__main__':
	alpha=0.5
	camera=cv2.VideoCapture(0)
	num_frames=0

	top, right, bottom, left = 10, 350, 225, 590

	calibrated=False

	while(True):
		(grabbed,frame) = camera.read()
		frame=imutils.resize(frame,width=700)
		frame=cv2.flip(frame,1) #flip so that image is not mirror view
		clone=frame.copy()
		#print(frame)
		(height,width) = frame.shape[:2]
		roi = frame[top:bottom, right:left]

		gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
		gray=cv2.GaussianBlur(gray,(7,7),0)


		if num_frames<30:
			running_avg(gray,alpha)
		else:
			hand=segment(gray)
			if hand is not None:
				(thresholded,segmented) = hand

				cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
				fingers=count(thresholded,segmented)
				#fingers-=1
				#print(type(fingers))

				cv2.putText(clone,str(fingers),(70,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
				if fingers == 1:
					cv2.putText(frame,"Unmute", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
					handle=Popen('nircmd.exe mutesysvolume 1',shell=True,stdout=PIPE,stderr=STDOUT,stdin=PIPE)
					print("hey1")
				elif fingers == 2:
					cv2.putText(frame, "Increase Volume", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
					handle=Popen('nircmd.exe changesysvolume 2000',shell=True,stdout=PIPE,stderr=STDOUT,stdin=PIPE)
					print("hey2")
				elif fingers == 3:
					cv2.putText(frame,"Max Volumn", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
					handle=Popen('nircmd.exe setsysvolume 65535',shell=True,stdout=PIPE,stderr=STDOUT,stdin=PIPE)
					print("hey3")
				elif fingers == 4:
					cv2.putText(frame,"Decrease Volume", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
					handle=Popen('nircmd.exe changesysvolume -2000',shell=True,stdout=PIPE,stderr=STDOUT,stdin=PIPE)
					print("hey4")
				elif fingers == 5:
					handle=Popen('nircmd.exe mutesysvolume 1',shell=True,stdout=PIPE,stderr=STDOUT,stdin=PIPE)
					print("hey5")
					cv2.putText(frame,"Volume 0%", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
				#print(segmented)
				cv2.imshow("Thresholded",thresholded)

		cv2.rectangle(clone,(left,top),(right,bottom),(0,255,0),2)
		

		num_frames+=1
		cv2.imshow("Video",clone)

		k=cv2.waitKey(30) & 0xff
		if k == 27:
			break

	camera.release()
	cv2.destroyAllWindows()