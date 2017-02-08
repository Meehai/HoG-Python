import os
from Mihlib import *
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn import svm

DATASET_PATH="INRIAPerson/test_64x128_H96/pos"
CELL_SIZE=(8, 8) # 8x8 shape
CELLS_PER_BLOCK=(2, 2) # 2x2 cells in a block
NUM_BINS=18 # 18 bins over 360 degrees
POS = 1
NEG = 0

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

def readDataset(path):
	paths = []
	result = []
	for root, subdirs, files in os.walk(path):
		for file in files:
			paths.append(path+"/"+file)
		break

	for image_path in paths:
		(image, mode) = readImage(image_path)
		result.append(image)
	return result

def plotDataset(images):
	# for image in images:
	plot_image(images[0])
	plt.show(block=True)

# Given an image and a cell_size, return the indexes of top-left and bottom-right in the image 
def getCellIndexes(image, cell_size):
	indexes=[]
	for i in range(0, image.shape[I], cell_size[I]):
		bottom_right_i = min(i + cell_size[I], image.shape[I])
		for j in range(0, image.shape[J], cell_size[J]):
			top_left = (i, j)
			bottom_right_j = min(j + cell_size[J], image.shape[J])
			bottom_right = (bottom_right_i, bottom_right_j)
			indexes.append((top_left, bottom_right))
	return indexes

# Orientation binning. Given an image, indexes of a cell and a matrix of pairs of (orientation, magnitude), compute
# the histogram of the cell, weighted by the magnitude of each pixel, stored as an array length num_bins. 
def createOrientationBinning(image, num_bins, cell_indexes, gradient_values):
	bins = np.zeros(num_bins)
	top_left = cell_indexes[0]
	bottom_right = cell_indexes[1]
	for i in range(top_left[I], bottom_right[I]):
		for j in range(top_left[J], bottom_right[J]):
			(angle, magnitude) = gradient_values[i][j]
			bin_index = int(angle / (360 / num_bins))
			bins[bin_index] += magnitude
	return bins

def train(dataset):
	image = grayScaleImage(dataset[0])

	height = image.shape[0]
	width = image.shape[1]
	gradientXImage = computeCenteredXGradient(image)
	gradientYImage = computeCenteredYGradient(image)
	gradientImage = computeCenteredGradient(gradientXImage, gradientYImage)

	cell_indexes = getCellIndexes(image, CELL_SIZE)
	print("Number of cells of test image:", len(cell_indexes))

	#print(getCellIndexes(test_image, CELL_SIZE))
	print(createOrientationBinning(image, NUM_BINS, cell_indexes[0], gradientImage))

def trainSVM(hogMap, C):
	negativeExamples = hogMap[NEG]
	positiveExamples = hogMap[POS]
	trainingExamples = np.concatenate((negativeExamples, positiveExamples))
	classExamples = np.concatenate((np.zeros(len(negativeExamples)), np.ones(len(positiveExamples)))) 
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(trainingExamples, classExamples)

def main():
	images = readDataset(DATASET_PATH)
	print("Read dataset of ", len(images), "images")
	#plotDataset(images)
	train(images)
	hogMap = {0: [1,2,3], 1: [4,5,6,7]}
	C = 0.01
	trainSVM(hogMap)

if __name__ == "__main__":
	main()
