#!/usr/bin/env python3
# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import os
import numpy as np
from datetime import datetime
import sys

RATIO = None
IMG = None
BLUR = None
METODO = None
TRSHVAL = None
PERI = None
refPt = []
cropping = False

def init(image, ratio, blur):
	global imageName
	global RATIO
	global IMG
	global BLUR
	global METODO
	METODO = 2
	RATIO = ratio
	IMG = image
	BLUR = blur
	# initialize the list of reference points and boolean indicating
	# whether cropping is being performed or not
	refPt = []
	cropping = False

def shapeDetect(contour, periparam):
	per = periparam/1000
	perimeter = per*cv2.arcLength(contour,True)
	polygon = cv2.approxPolyDP(contour,perimeter,True)
	return polygon

def perimeterCallback(x):
	nothing(cv2.getTrackbarPos('threshold','Image Controllers'))

def nothing(x):
	# print(x)
	global TRSHVAL
	TRSHVAL = x
	PERI = cv2.getTrackbarPos('perimeter','Image Controllers')

	imageTemp = IMG.copy()
	ths = []
	ths_names = []

	thresh1 = cv2.threshold(BLUR,x,255,cv2.THRESH_BINARY)[1]
	# showImage("normal_binary",thresh1)
	thresh2 = cv2.threshold(BLUR,x,255,cv2.THRESH_BINARY_INV)[1]
	# showImage("normal_binary_inv",thresh2)
	thresh3 = cv2.threshold(BLUR,x,255,cv2.THRESH_TRUNC)[1]
	# showImage("normal_trunc",thresh3)
	thresh4 = cv2.threshold(BLUR,x,255,cv2.THRESH_TOZERO)[1]
	# showImage("normal_tozero",thresh4)
	thresh5 = cv2.threshold(BLUR,x,255,cv2.THRESH_TOZERO_INV)[1]
	# showImage("normal_tozero_inv",thresh5)
	thresh6 = cv2.threshold(BLUR,x,255,cv2.THRESH_BINARY)[1]
	# showImage("binary_otsu",thresh6)
	
	ths = [thresh1,thresh2,thresh3,thresh4,thresh5,thresh6]
	ths_names = ['BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV','OTZU']

	if x%2==1:
		thresh_mean = cv2.adaptiveThreshold(BLUR,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,x,2)
		# showImage("th_mean", thresh_mean)
		thresh_gauss = cv2.adaptiveThreshold(BLUR,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,x,2)
		# showImage("th_gauss", thresh_gauss)
		ths.append(thresh_mean)
		ths.append(thresh_gauss)
		ths_names.append("MEAN")
		ths_names.append("GAUSS")

	for i in range(len(ths_names)):	
		# find contours in the thresholded image and initialize the shape detector	
		cnts = cv2.findContours(ths[i].copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		# loop over the contours
		for c in cnts:

			c = c.astype("float")
			c *= RATIO
			c = c.astype("int")
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			ret = cv2.matchShapes(box,c,METODO,0.0)
			# ============= shape detect
			polygon = shapeDetect(c, PERI);
			# ============= end shape detect
			
			# ============= is a square?
			if len(polygon)==4:
				imageTemp = blurSquares(imageTemp, c)				
			# drawContornos(imageTemp,c)	

		showImage(ths_names[i]+str(METODO),imageTemp)

def showImage(name, img):
	cv2.imshow(name, img)

def saveImage(directory, name, img):
	if directory[:-1] != '/':
		directory = directory + "/"
	if not os.path.exists("./BM"+str(METODO)+"_"+directory):
		os.makedirs("./BM"+str(METODO)+"_"+directory)
	cv2.imwrite("./BM"+str(METODO)+"_"+directory+imageName+"_"+str(datetime.now().strftime("%Y%m%d-%H_%M"))+"_"+name+'.jpeg',img)

def blurRoi(img, x1, y1, x2, y2, k):
	# roi in the image First [y:y+dy, x:x+dx]
	# y increase
	yy1 = y1
	yy2 = y2
	# x increase
	xx1 = x1
	xx2 = x2
	# select Region of Interest
	roiF = img[yy1:yy2,xx1:xx2]
	# blur
	roiF_blurred = cv2.GaussianBlur(roiF, (k, k), 0)
	img[yy1:yy2,xx1:xx2] = roiF_blurred
	return img

# c is a contour
# image is the image
def blurSquares(image, c):
	# compute the center of the contour, then detect the name of the
	# shape using only the contour using original contour c
	M = cv2.moments(c)
	# print(M)
	# ========== ROI
	x,y,w,h = cv2.boundingRect(c)
	# ========== end ROI
	# Check whether is a rectangle or not
	if w/h>3:
		cX = 0
		cY = 0
		if M["m00"] != float(0):
			cX = int((M["m10"] / M["m00"]) * RATIO)
			cY = int((M["m01"] / M["m00"]) * RATIO)
		
		if M["m00"]*RATIO >=100 and M["m00"]*RATIO <= 2000:
		
			image = blurRoi(image, x, y, x+w, y+h, 31)
			# DRAWING
			cv2.rectangle(image, (x,y), (x+w,y+h), (253,240,47), 2)
		# print("Polygon: {} Sides: {}".format(polygon, len(polygon)))
		# DRAWING
		# cv2.putText(image, "S: {:f}".format(ret), (cX-50, cY), cv2.FONT_HERSHEY_SIMPLEX, 
		# 	0.5, (204,0,204), 1)
		# TEXT DRAWING
		# cv2.putText(image, "M: {:.3f}".format( M["m00"]*RATIO), (cX-50, cY+15), cv2.FONT_HERSHEY_SIMPLEX, 
		# 	0.5, (204,0,204), 1)
		# cv2.putText(image, "dx/dy: {}".format( w/h ), (cX-50, cY+30), cv2.FONT_HERSHEY_SIMPLEX, 
		# 	0.5, (204,0,204), 1)
		# cv2.putText(image, "Sh: {}".format( len(polygon) ), (cX-50, cY+30), cv2.FONT_HERSHEY_SIMPLEX, 
		# 	0.5, (204,0,204), 1)
	return image

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that click_and_croppping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

def drawContornos(image, c):
	cv2.drawContours(image, [c], -1, (0, 204, 0), 1)
	# cv2.drawContours(image, [polygon], -1, (204, 0, 0), 2)

def processImage(directory, imgSrc):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	image = cv2.imread(imgSrc)
	resized = imutils.resize(image, width=300)
	ratio = image.shape[0] / float(resized.shape[0])	

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (1, 1), 0)
	
	# ==== TEMP ==== to modify the image
	cv2.namedWindow('Image Controllers', cv2.WINDOW_NORMAL)
	# trackbar on the image
	cv2.createTrackbar('threshold','Image Controllers',52,255,nothing)
	cv2.createTrackbar('perimeter','Image Controllers',10,100,perimeterCallback)

	init(image, ratio, blurred)

	while (1):

		k = cv2.waitKey(0) & 0xff
		print(imgSrc)
		print(k)
		if k == 27:
			# Do nothing
			break
		if k == 13:
			# Save the current state of the image
			print("enter")
			break
	# imageTemp = image.copy()
	# thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,treshVal,2)
	# showImage("th", thresh)
	# print(str(treshVal))
	# # find contours in the thresholded image and initialize the shape detector	
	# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# # loop over the contours
	# for c in cnts:

	# 	c = c.astype("float")
	# 	c *= ratio
	# 	c = c.astype("int")
	# 	rect = cv2.minAreaRect(c)
	# 	box = cv2.boxPoints(rect)
	# 	box = np.int0(box)
	# 	ret = cv2.matchShapes(box,c,METODO,0.0)
	# 	# ============= shape detect
	# 	polygon = shapeDetect(c);
	# 	# ============= end shape detect
	# 	# is a square?
	# 	# if len(polygon)==4:
	# 	# 	blurSquares(imageTemp, ratio, c)				
	# 	drawContornos(imageTemp,c)	

	# showImage("-Final-"+str(METODO),imageTemp)

	

	# treshVal = cv2.getTrackbarPos('threshold','Image Controllers')
	# saveImage(directory+"-Final-"+str(METODO),image)

def main():
	dirFol = ''
	imageName = ''
	BASE = os.path.dirname(os.path.abspath(__file__))

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--folder", required=False,
		help="path to the input image")
	args = vars(ap.parse_args())

	if args['folder']:
		dirFol = args['folder']
		for file in os.listdir(dirFol):
		    filename = os.fsdecode(file)
		    if filename.endswith(".jpeg") or filename.endswith(".jpg"): 
			    imageName = filename.split('.')[0]
			    try:
			    	print("[INFO] processing image {}".format(imageName))
			    	processImage(dirFol, os.path.join(dirFol, filename))
			    # we are trying to control-c out of the script, so break from the
				# loop (you still need to press a key for the active window to
				# trigger this)
			    except KeyboardInterrupt:
			    	print("[INFO] manually leaving script")
			    	break
				# an unknown error has occurred for this particular image
			    except:
			    	print("[INFO] skipping image...")
	else:
		sys.exit("No arguments provided, either --image or --folder")


if __name__ == '__main__':
	main()

# show the output image
# saveImage(directory,"Final",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ========= Blur alternatives
# blurred = cv2.medianBlur(gray, 5)
# ========= Tresh alternatives
# thresh = cv2.threshold(blurred, 135, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
# ========= boundaries alternatives??? 
# hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
# boundaries = [([254,252,27],[255,255,250]),([160,51,160],[252,182,252]),([175,151,61],[254,252,27])]

# for (lower,upper) in boundaries:
# 	lower = np.array(lower, dtype = "uint8")
# 	upper = np.array(upper, dtype = "uint8")

# 	mask = cv2.inRange(image, lower, upper)
# 	output = cv2.bitwise_and(image, image, mask = mask)

# 	cv2.imshow("images", np.hstack([image, output]))
# 	cv2.waitKey(0)

##
