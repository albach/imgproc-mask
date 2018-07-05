#!/usr/bin/env python3
# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from blur.blurr import BlurWatermark
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
	global TRSHVAL, PERI
	TRSHVAL = x
	PERI = cv2.getTrackbarPos('perimeter','Image Controllers')

	showImage("imigy",prepareImage())

def prepareImage():

	global TRSHVAL, PERI

	imageTemp = IMG.copy()

	thresh5 = cv2.threshold(BLUR,TRSHVAL,255,cv2.THRESH_TOZERO_INV)[1]

	# find contours in the thresholded image and initialize the shape detector	
	cnts = cv2.findContours(thresh5.copy(), cv2.RETR_EXTERNAL,
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

	return imageTemp

def showImage(name, img):
	cv2.imshow(name, img)

def saveImage(directory, name, img):
	if directory[:-1] != '/':
		directory = directory + "/"
	if not os.path.exists("./MODS/"+"_"+directory):
		os.makedirs("./MODS/"+"_"+directory)
	cv2.imwrite("./MODS/"+"_"+directory+name+"-M"+'.jpeg',img)

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
	global finalSave
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
			if not finalSave:
				cv2.rectangle(image, (x,y), (x+w,y+h), (253,240,47), 2)
	
	return image

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, IMG
 
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
		# clone = IMG.copy()
		# clone = IMG.copy()
		# draw a rectangle around the region of interest
		# cv2.rectangle(IMG, refPt[0], refPt[1], (0, 255, 0), 2)
		y = refPt[0][1]
		x = refPt[0][0]
		h = refPt[1][1]
		w = refPt[1][0]
		blurArea = blurRoi(IMG, x, y, w, h, 31)
		cv2.imshow("imigy", blurArea)

def drawContornos(image, c):
	cv2.drawContours(image, [c], -1, (0, 204, 0), 1)
	# cv2.drawContours(image, [polygon], -1, (204, 0, 0), 2)

def processImage(directory, imgSrc):
	global imageName,finalSave;
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	image = cv2.imread(imgSrc)
	clone = image.copy()
	resized = imutils.resize(image, width=300)
	ratio = image.shape[0] / float(resized.shape[0])	

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (1, 1), 0)
	
	init(image, ratio, blurred)
	
	# ==== TEMP ==== to modify the image
	cv2.namedWindow('Image Controllers', cv2.WINDOW_NORMAL)
	# trackbar on the image
	cv2.createTrackbar('threshold','Image Controllers',52,255,nothing)
	cv2.createTrackbar('perimeter','Image Controllers',10,100,perimeterCallback)

	cv2.namedWindow("imigy")
	cv2.setMouseCallback("imigy", click_and_crop)
	 

	while True:

		showImage("imigy",image)
		k = cv2.waitKey(0) & 0xff

		if k == 27:
			# Save and next image
			saveImage(directory,imageName,image)
			break
		elif k == ord("r"):
			image = clone.copy()
			init(image, ratio, blurred)
		elif k == 13:
			# Save the current state of the image
			print("enter")
			if len(refPt) == 2:
				print("manual roi based...")
				y = refPt[0][1]
				x = refPt[0][0]
				h = refPt[1][1]
				w = refPt[1][0]
				blurArea = blurRoi(image, x, y, w, h, 31)
				saveImage(directory,imageName,blurArea)
			else:
				print("no manual roi, controlled based...")
				finalSave = True
				finalImage = prepareImage()
				saveImage(directory,imageName,finalImage)
				finalSave = False
			break

	# close all open windows
	cv2.destroyAllWindows()

def main():
	dirFol = ''
	global imageName
	global finalSave
	imageName = ''
	finalSave = False
	BASE = os.path.dirname(os.path.abspath(__file__))

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--folder", required=False,
		help="path to the input image")
	# args = vars(ap.parse_args())

	# ============================== 
	# ============ TODO ============ 
	# ============================== 
	# ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prop", required=True,
		help="path to the properties file containing the area points")
	ap.add_argument("-k", "--kgauss", required=True,
		help="parameter for the GaussianBlur minimum 1")
	# # ap.add_argument("-i", "--image", required=False,
	# # 	help="path to the input image")
	# # ap.add_argument("-f", "--folder", required=False,
	# # 	help="path to the input image")

	args = vars(ap.parse_args())

	# TODO: This is too complicated
	if args['folder']:
		dirFol = args['folder']
		for file in os.listdir(dirFol):
		    filename = os.fsdecode(file)
		    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith("png"): 
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
			    except Exception as e:
			    	print(e)
			    	print("[INFO] skipping image...")

		bm = BlurWatermark(dirFol,args['prop'],args['kgauss'])
		print("K gauus: {}".format(bm.k))
		print("K gauus: {}".format(bm.folder))
		print("K gauus: {}".format(bm.prop))
		bm.startBlur()

	else:
		sys.exit("No arguments provided, either --folder, --prop or --kgauss ")


if __name__ == '__main__':
	main()
