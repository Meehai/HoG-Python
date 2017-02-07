from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt

def computeCenteredXGradient(grayscaleImage):
	height = grayscaleImage.shape[0]
	width = grayscaleImage.shape[1]
	gradientXImage = np.zeros((height, width), dtype=int)
	centeredKernel = np.array([-1, 0, 1])
	for i in range(height):
		firstArray = np.array([grayscaleImage[i][0], grayscaleImage[i][0], grayscaleImage[i][1]])
		lastArray = np.array([grayscaleImage[i][width - 2], grayscaleImage[i][width - 1], grayscaleImage[i][width - 1]])
		gradientXImage[i][0] = np.dot(firstArray, centeredKernel)
		for j in range(1, width - 1):
			gradientXImage[i][j] = np.dot(grayscaleImage[i][j - 1: j + 2], centeredKernel)
		gradientXImage[i][width - 1] = np.dot(lastArray, centeredKernel)
	return gradientXImage

def computeCenteredYGradient(grayscaleImage):
	grayscaleImage = np.transpose(grayscaleImage)
	height = grayscaleImage.shape[0]
	width = grayscaleImage.shape[1]
	gradientYImage = np.zeros((height, width), dtype=int)
	centeredKernel = np.array([-1, 0, 1])
	for i in range(height):
		firstArray = np.array([grayscaleImage[i][0], grayscaleImage[i][0], grayscaleImage[i][1]])
		lastArray = np.array([grayscaleImage[i][width - 2], grayscaleImage[i][width - 1], grayscaleImage[i][width - 1]])
		gradientYImage[i][0] = np.dot(firstArray, centeredKernel)
		for j in range(1, width - 1):
			gradientYImage[i][j] = np.dot(grayscaleImage[i][j - 1: j + 2], centeredKernel)
		gradientYImage[i][width - 1] = np.dot(lastArray, centeredKernel)
	gradientYImage = np.transpose(gradientYImage)
	return gradientYImage

def computeCenteredGradient(gradientXImage, gradientYImage):
	height = gradientXImage.shape[0]
	width = gradientXImage.shape[1]
	gradientImage = [[(0, 0) for i in range(width)] for j in range(height)]
	for i in range(height):
		for j in range(width):
			orientation = np.arctan2(gradientYImage[i][j], gradientXImage[i][j]) * 180 / np.pi;
			if orientation < 0:
				orientation = 360 + orientation
			magnitude = np.sqrt(gradientXImage[i][j] * gradientXImage[i][j] + gradientYImage[i][j] * gradientYImage[i][j])
			gradientImage[i][j] = (orientation, magnitude)
	return gradientImage

def main():
	# citire poza, convert la grayscale
	im = Image.open(sys.argv[1])
	im = im.convert('L')
	image = np.array(im, dtype=float)
	
	height = image.shape[0]
	width = image.shape[1]
	gradientXImage = computeCenteredXGradient(image)
	gradientYImage = computeCenteredYGradient(image)
	gradientImage = computeCenteredGradient(gradientXImage, gradientYImage)
	magnitude = [[gradientImage[i][j][1] for j in range(width)] for i in range(height)]

	plt.imshow(magnitude)
	plt.show(block=True)

if __name__ == "__main__":
	main()
