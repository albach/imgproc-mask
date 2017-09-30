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
global directory
directory = ''
global imageName
imageName = ''
BASE = os.path.dirname(os.path.abspath(__file__))

def showImage(name, img):
	global imgCounterShow
	imgCounterShow += 1
	cv2.imshow(str(imgCounterShow)+" "+name, img)

def saveImage(directory, name, img):
	# showImage(name,img)
	global imgCounterSave
	imgCounterSave += 1
	# print("../"+directory+"-"+imageName+"_"+str(datetime.now().strftime("%Y-%m-%d")))
	if not os.path.exists("../"+directory+"-"+imageName+"_"+str(datetime.now().strftime("%Y-%m-%d"))):
		os.makedirs("../"+directory+"-"+imageName+"_"+str(datetime.now().strftime("%Y-%m-%d")))
	cv2.imwrite("../"+directory+"-"+imageName+"_"+str(datetime.now().strftime("%Y-%m-%d"))+"/"+str(imgCounterSave)+" "+name+'.jpeg',img)

def processImage(directory, imgSrc):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	# print("Folder: {} img: {}".format(directory, imgSrc))
	image = cv2.imread(imgSrc)
	# print("Image object: {}".format(image))
	# showImage("test", image)
	resized = imutils.resize(image, width=300)
	saveImage(directory,"resized",resized)
	ratio = image.shape[0] / float(resized.shape[0])

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	saveImage(directory,"gray",gray)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	saveImage(directory,"blurred",blurred)
	# thresh = cv2.threshold(blurred, 135, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)[1]
	# thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	saveImage(directory,"thresh",thresh)
	inverted = cv2.bitwise_not(thresh);
	saveImage(directory,"inverted",inverted)
	# find contours in the thresholded image and initialize the
	# shape detector 
	saveImage(directory,"original",image)

	# Miguel: Initially the thresh
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	sd = ShapeDetector()
	i = 0
	# loop over the contours
	for c in cnts:
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		if i:
			c2 = c
			c2 = c2.astype("float")
			c2 *= ratio
			c2 = c2.astype("int")
			rect = cv2.minAreaRect(c2)
			# area = cv2.contourArea(c)
			box = cv2.boxPoints(rect)
			box = np.int0(box)

			ret = cv2.matchShapes(box,c2,1,0.0)
			print("Match Shape value: {} ".format(ret))

			M = cv2.moments(c)
			# print(M)
			cX = 0
			cY = 0
			if M["m00"] != float(0):
				cX = int((M["m10"] / M["m00"]) * ratio)
				cY = int((M["m01"] / M["m00"]) * ratio)
			
			shape = sd.detect(c)
			
			# c = c.astype("float")
			# c *= ratio
			# c = c.astype("int")

			rect = cv2.minAreaRect(c)
			# area = cv2.contourArea(c)
			# areaMoments = M["m00"]
			# box = cv2.boxPoints(rect)
			# box = np.int0(box)
			# print("area {} - areaMoment {}".format(area, areaMoments))
			# if shape:
			
			# if ret<= 0.18 and M["m00"] >=400 and M["m00"] <= 1000 :
			if ret<= 0.25 :
			# if M:
				# peri = cv2.arcLength(c, True)
				print("HOla/?")
				# mask defaulting to black for 3-channel and transparent for 4-channel
				# (of course replace corners with yours)
				maskIni = np.zeros(image.shape, dtype=np.uint8)
				mask = maskIni
				# saveImage(directory,"maskInitialize"+str(i),maskIni)

				# roi_corners = np.array([[(10,10), (200,200), (10,200)]], dtype=np.int32)
				roi_corners = np.array([[tuple(tup) for el in [y * ratio for y in c] for tup in el]], dtype=np.int32)
				# fill the ROI so it doesn't get wiped out when the mask is applied
				# channel_count2 = image.shape[1]  # i.e. 3 or 4 depending on your image
				# ignore_mask_color = (255,)*channel_count3
				# cv2.fillPoly(mask, roi_corners, ignore_mask_color)
				# showImage("maskAgain"+str(i), mask)

				white = (255, 255, 255)
				cv2.fillPoly(mask, roi_corners, white)
				saveImage(directory,"mask"+str(i),mask)

				# apply the mask
				masked_image_bit = cv2.bitwise_and(image, maskIni)
				saveImage(directory,"mask.bitwise.and.image"+str(i),masked_image_bit)
				
				# masked_image_bit_not = cv2.bitwise_not(masked_image_bit)
				# saveImage(directory,"mask.bitwise.not.image"+str(i),masked_image_bit_not)

				# blurred_mask_image_bit_not = cv2.GaussianBlur(masked_image_bit_not,(23, 23), 10)
				# saveImage(directory,"blurred_maskimgbit_not"+str(i),blurred_mask_image_bit_not)

				blurred_mask_image_bit = cv2.GaussianBlur(masked_image_bit,(23, 23), 10)
				# saveImage(directory,"blurred_maskimgbit"+str(i),blurred_mask_image_bit)
				
				blurMask_add_img = cv2.add(image,blurred_mask_image_bit)
				saveImage(directory,"blurmask.add.image"+str(i),blurMask_add_img)

				# blurMaskNot_add_img = cv2.add(image,blurred_mask_image_bit_not)
				# saveImage(directory,"blurmaskNot.add.image"+str(i),blurMaskNot_add_img)

				mask_add_img = cv2.add(image, maskIni)
				saveImage(directory,"mask.add.image"+str(i),mask_add_img)

				# invBlurMask = cv2.bitwise_not(blurred_mask_image_bit)
				# saveImage(directory,"invertedBlurredMask"+str(i),invBlurMask)

				#add blurred mask with content to original image with white space
				# maskImg_add_blurImag = cv2.add(mask_add_img, invBlurMask)
				# saveImage(directory,"img.mask.add.blurImg"+str(), maskImg_add_blurImag)

				# invBlurMask = cv2.bitwise_not(blurred_mask_image_bit)
				# saveImage(directory,"invertedBlurredMask_bit_not"+str(i),invBlurMask)

				# invBlurMask_not = cv2.bitwise_not(blurred_mask_image_bit_not)
				# saveImage(directory,"invertedBlurredNotMask_bit_not"+str(i),invBlurMask_not)
				
				# masked_img_xor = cv2.bitwise_xor(mask_add_img, blurred_mask_image_bit)
				# saveImage(directory,"mask.xor.image"+str(i),masked_img_xor)

				image = mask_add_img

				# masked_not_xor_img = cv2.bitwise_or(mask_add_img, invBlurMask_not)
				# saveImage(directory,"maskNot.xor.image"+str(i),masked_not_xor_img)

				# img1_bg = cv2.bitwise_and(roi_corners,roi_corners,masked_image)
				# showImage("bitwise", img1_bg)
				# result = blurred_mask / image
				# saveImage(directory,"resultMinus"+str(i),result)

				# save the result
				# saveImage(directory,"masked"+str(i),masked_image)
				# (x, y, w, h) = cv2.boundingRect(cv2.approxPolyDP(c, 0.04 * peri, True))
				# print("and this?: ", list(zip(*c)))
				# print('whats this?', [tuple(tup) for el in c for tup in el])
				# print('shape: ',tuple(c))
				# multiply the contour (x, y)-coordinates by the resize ratio,
				# then draw the contours and the name of the shape on the image			

				cv2.drawContours(image, [box],0,(0,0,255),2)
				# cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
				cv2.putText(image, "S: "+str(ret), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 
					0.5, (0, 0, 0), 2)
		i+=1

	saveImage(directory,"Final",image)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to the input image")
ap.add_argument("-f", "--folder", required=False,
	help="path to the input image")
args = vars(ap.parse_args())

if args['image']:
	directory = args['image'].split('.')[0] + datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
	# print(type(args["image"]))
	processImage(directory, args['image'])
elif args['folder']:
	directory = args['folder']

	for file in os.listdir(directory):
	    filename = os.fsdecode(file)
	    if filename.endswith(".jpeg") or filename.endswith(".jpg"): 
	        # print(os.path.join(directory, filename))
	        # print(filename)
	        # print(type(filename))
	        imageName = filename.split('.')[0]
	        processImage(directory, os.path.join(directory, filename))
	    else:
	        continue
else:
	sys.exit("No arguments provided, either --image or --folder")

# show the output image
# saveImage(directory,"Final",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()