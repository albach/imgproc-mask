# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import os
import numpy as np

def saveImage(directory, name, img):
	if not os.path.exists(directory):
		os.makedirs(directory)
	cv2.imwrite("./"+directory+"/"+name+'.jpeg',img)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
directory = args['image'].split('.')[0]

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
cv2.imshow("Image resized", resized)
saveImage(directory,"resized",resized)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image gray", gray)
saveImage(directory,"gray",gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Image blurred", blurred)
saveImage(directory,"blurred",blurred)
# thresh = cv2.threshold(blurred, 135, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Image thresh", thresh)
saveImage(directory,"thresh",thresh)
inverted = cv2.bitwise_not(thresh);
cv2.imshow("Image inve", inverted)
saveImage(directory,"inverted",inverted)
# find contours in the thresholded image and initialize the
# shape detector 
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
	if i==23:
		M = cv2.moments(c)
		# print(M)
		cX = 0
		cY = 0
		if M["m00"] != float(0) and M["m00"] != float(0):
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)
		shape = sd.detect(c)
		peri = cv2.arcLength(c, True)

		# mask defaulting to black for 3-channel and transparent for 4-channel
		# (of course replace corners with yours)
		mask = np.zeros(image.shape, dtype=np.uint8)
		cv2.imshow("mask"+str(i), mask)
		cv2.imshow("image"+str(i), image)

		# roi_corners = np.array([[(10,10), (200,200), (10,200)]], dtype=np.int32)
		roi_corners = np.array([[tuple(tup) for el in [y * ratio for y in c] for tup in el]], dtype=np.int32)
		# fill the ROI so it doesn't get wiped out when the mask is applied
		# channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
		# ignore_mask_color = (255,)*channel_count
		white = (255, 255, 255)
		cv2.fillPoly(mask, roi_corners, white)
		cv2.imshow("maskAgain"+str(i), mask)

		# apply the mask
		masked_image_bit = cv2.bitwise_and(image, mask)
		cv2.imshow("mask.bitand.image"+str(i), masked_image_bit)
		saveImage(directory,"mask.bitand.image"+str(i),masked_image_bit)
		blurred_mask_image_bit = cv2.GaussianBlur(masked_image_bit,(23, 23), 10)
		cv2.imshow("blurred_maskimgbit"+str(i), blurred_mask_image_bit)
		saveImage(directory,"blurred_maskimgbit"+str(i),blurred_mask_image_bit)
		masked_image = cv2.add(image, mask)
		cv2.imshow("mask.add.image"+str(i), masked_image)
		saveImage(directory,"mask.add.image"+str(i),masked_image)
		cv2.imshow("inverted?", cv2.bitwise_not(blurred_mask_image_bit))
		masked_img_xor = cv2.bitwise_xor(masked_image, blurred_mask_image_bit)
		cv2.imshow("mask.xor.image"+str(i),masked_img_xor)
		saveImage(directory,"mask.xor.image"+str(i),masked_img_xor)

		# img1_bg = cv2.bitwise_and(roi_corners,roi_corners,masked_image)
		# cv2.imshow("bitwise", img1_bg)
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
		c = c.astype("float")
		c *= ratio
		c = c.astype("int")
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (255, 255, 255), 2)
	i+=1


# show the output image
cv2.imshow("Image", image)
saveImage(directory,"final",image)
cv2.waitKey(0)
cv2.destroyAllWindows()