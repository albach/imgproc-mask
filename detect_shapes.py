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

imgCounterSave = 0
imgCounterShow = 0
metodo = 2
dirFol = ''
global imageName
imageName = ''
BASE = os.path.dirname(os.path.abspath(__file__))

def showImage(name, img):
	global imgCounterShow
	imgCounterShow += 1
	cv2.imshow(str(imgCounterShow)+" "+name, img)

def saveImage(directory, name, img):
	if directory[:-1] != '/':
		directory = directory + "/"
	global imgCounterSave
	imgCounterSave += 1
	if not os.path.exists("../BM"+str(metodo)+"_"+directory):
		os.makedirs("../BM"+str(metodo)+"_"+directory)
	cv2.imwrite("../BM"+str(metodo)+"_"+directory+imageName+"_"+str(datetime.now().strftime("%Y%m%d-%H_%M"))+"_"+name+'.jpeg',img)

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

def processImage(directory, imgSrc):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	global imgCounterSave
	imgCounterSave = 0
	image = cv2.imread(imgSrc)
	# print("Image object: {}".format(image))
	resized = imutils.resize(image, width=300)
	ratio = image.shape[0] / float(resized.shape[0])	

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (1, 1), 0)
	thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,43,2)
	# saveImage(directory,"thresh",thresh)
	inverted = cv2.bitwise_not(thresh);

	# Miguel: find contours in the thresholded image and initialize the shape detector	
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	sd = ShapeDetector()
	i = 0
	# loop over the contours
	for c in cnts:
		c2 = c
		c2 = c2.astype("float")
		c2 *= ratio
		c2 = c2.astype("int")
		rect = cv2.minAreaRect(c2)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		ret = cv2.matchShapes(box,c2,metodo,0.0)
		# ============= shape detect
		perimeter = 0.07*cv2.arcLength(c2,True)
		polygon = cv2.approxPolyDP(c2,perimeter,True)
		# ============= end shape detect
		if len(polygon)==4:
			# compute the center of the contour, then detect the name of the
			# shape using only the contour using original contour c
			M = cv2.moments(c)
			# print(M)
			cX = 0
			cY = 0
			if M["m00"] != float(0):
				cX = int((M["m10"] / M["m00"]) * ratio)
				cY = int((M["m01"] / M["m00"]) * ratio)
			if M["m00"]*ratio >=1000 and M["m00"]*ratio <= 2000:
				# ========== ROI
				x,y,w,h = cv2.boundingRect(c2)
				# ========== end ROI
				image = blurRoi(image, x,y,x+w,y+h,31)
				# DRAWING
				cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
				# print("Polygon: {} Sides: {}".format(polygon, len(polygon)))
				# DRAWING
				cv2.putText(image, "S: {:f}".format(ret), (cX-50, cY), cv2.FONT_HERSHEY_SIMPLEX, 
					0.5, (204,0,204), 1)
				cv2.putText(image, "M: {:.3f}".format( M["m00"]*ratio), (cX-50, cY+15), cv2.FONT_HERSHEY_SIMPLEX, 
					0.5, (204,0,204), 1)
				# cv2.putText(image, "Sh: {}".format( len(polygon) ), (cX-50, cY+30), cv2.FONT_HERSHEY_SIMPLEX, 
				# 	0.5, (204,0,204), 1)
		# else:
			# It is not a square image between the given dimensions
			# cv2.drawContours(image, [c2], -1, (0, 204, 0), 2)
			# cv2.drawContours(image, [polygon], -1, (204, 0, 0), 2)	
	saveImage(directory,"-Final-"+str(metodo),image)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to the input image")
ap.add_argument("-f", "--folder", required=False,
	help="path to the input image")
args = vars(ap.parse_args())

if args['image']:
	dirFol = args['image'].split('.')[0] + datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
	# print(type(args["image"]))
	processImage(dirFol, args['image'])
elif args['folder']:
	dirFol = args['folder']

	for file in os.listdir(dirFol):
	    filename = os.fsdecode(file)
	    if filename.endswith(".jpeg") or filename.endswith(".jpg"): 
		    imgCounterShow = 0
		    imageName = filename.split('.')[0]
		    processImage(dirFol, os.path.join(dirFol, filename))
	    else:
	        continue
else:
	sys.exit("No arguments provided, either --image or --folder")

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
