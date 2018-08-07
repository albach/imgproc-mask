'''
 * Python script to Gaussian blur a section of an image.
 *
 * usage: python blurr.py -p <property filename> -k <kernel-size> 31
'''
import cv2, sys, os, argparse

class BlurWatermark():
	"""docstring for BlurWatermark"""
	def __init__(self, imagesFolder, propertyFile, kgauss):
		# super(BlurWatermark, self).__init__()
		self.prop = propertyFile
		self.folderName = imagesFolder
		self.folder = "./MODS/"+"_"+imagesFolder
		self.k = int(kgauss)

	# def showImage(self, name, img):
	# 	cv2.imshow(name, img)

	def saveImage(self, name, img):
		if not os.path.exists(self.folder+"/"+self.folderName):
			os.makedirs(self.folder+"/"+self.folderName)
		cv2.imwrite(self.folder+"/"+self.folderName+"/"+name+'.jpeg',img)

	def blurRoi(self, img, x1, y1, x2, y2, k):
		# roi in the image First [y:y+dy, x:x+dx]
		# y increase
		yy1 = y1-5
		yy2 = y2+5
		# x increase
		xx1 = x1-5
		xx2 = x2+5
		# select Region of Interest
		roiF = img[yy1:yy2,xx1:xx2]
		# blur
		roiF_blurred = cv2.GaussianBlur(roiF, (self.k, self.k), 0)
		img[yy1:yy2,xx1:xx2] = roiF_blurred
		return img

	def startBlur(self):
		imgCounter = 0
		points = []

		f = open(self.prop, "r")
		for line in f:
			points.append(line.strip("\n\r").split("\t"))
		f.close()

		print(points)
		# print(k)
		
		# Loop over the files
		for file in os.listdir(self.folder):
		    filename = os.fsdecode(file)
		    # Only files with .jpeg or jpg extension
		    if filename.endswith(".jpeg") or filename.endswith(".jpg"): 
			    # Only name, no extension
			    imageName = filename.split('.')[0]
			    # Image read
			    img = cv2.imread(os.path.join(self.folder, filename))    
			    # temporal Image
			    tempImg = img
			    # Loop over the different regions to be blurred
			    for point in points:
			    	x1 = int(point[0].split(",")[0])
			    	y1 = int(point[0].split(",")[1])
			    	x2 = int(point[1].split(",")[0])
			    	y2 = int(point[1].split(",")[1])
			    	# modify image
			    	tempImg = self.blurRoi(tempImg, x1, y1, x2, y2, self.k)	    	
			    img = tempImg
			    self.saveImage(str(imgCounter)+"."+imageName+"-M", img)
			    print("Finished!")
			    imgCounter = imgCounter + 1
		    else:
		        continue

		return "Finished"