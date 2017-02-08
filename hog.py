import os
from Mihlib import *
from PIL import Image
import numpy as np
from numpy import linalg as LA
import sys
import matplotlib.pyplot as plt

DATASET_PATH="INRIAPerson/train_64x128_H96/pos"
CELL_SIZE=(8, 8) # 8x8 shape
CELLS_PER_BLOCK=(2, 2) # 2x2 cells in a block
NUM_BINS=18 # 18 bins over 360 degrees
DETECTION_WINDOW=(128,64)

def computeCenteredXGradient(grayscaleImage):
	height = grayscaleImage.shape[0]
	width = grayscaleImage.shape[1]
	gradientXImage = np.zeros(DETECTION_WINDOW, dtype=int)
	centeredKernel = np.array([-1, 0, 1])
	start_height = int((height - DETECTION_WINDOW[I]) / 2)
	end_height = start_height + DETECTION_WINDOW[I] + 1
	for i in range( start_height, start_height + DETECTION_WINDOW[I]):
		start_width = int((width - DETECTION_WINDOW[J]) / 2)
		for j in range(start_width, start_width + DETECTION_WINDOW[J]):
			gradientXImage[i-start_height][j-start_width] = np.dot(grayscaleImage[i][j - 1: j + 2], centeredKernel)
	return gradientXImage

def computeCenteredYGradient(grayscaleImage):
	grayscaleImage = np.transpose(grayscaleImage)
	height = grayscaleImage.shape[0]
	width = grayscaleImage.shape[1]
	gradientYImage = np.zeros((DETECTION_WINDOW[J], DETECTION_WINDOW[I]), dtype=int)
	centeredKernel = np.array([-1, 0, 1])
	start_height = int((height - DETECTION_WINDOW[J]) / 2)
	for i in range( start_height, start_height + DETECTION_WINDOW[J]):
		start_width = int((width - DETECTION_WINDOW[I]) / 2)
		for j in range(start_width, start_width + DETECTION_WINDOW[I]):
			gradientYImage[i-start_height][j-start_width] = np.dot(grayscaleImage[i][j - 1: j + 2], centeredKernel)
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
	return np.array(gradientImage)

def readDataset(path):
	paths = []
	result = []
	for root, subdirs, files in os.walk(path):
		for file in files:
			paths.append(path+"/"+file)
		break

	for image_path in paths:
		image = Image.open(image_path).convert("L")
		image = np.array(image)
		result.append(image)
	return (paths, result)

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
def createOrientationBinning(num_bins, cell_indexes, gradient_values):
	bins = np.zeros(num_bins)
	top_left = cell_indexes[0]
	bottom_right = cell_indexes[1]
	for i in range(top_left[I], bottom_right[I]):
		for j in range(top_left[J], bottom_right[J]):
			(angle, magnitude) = gradient_values[i][j]
			bin_index = int(angle / (360 / num_bins))
			bins[bin_index] += magnitude
	return bins

# Output: a matrix of each Histogram cell of shape num_cells_i, num_cells_j
def getOrientationBinMatrix(gradientImage):
	cell_indexes = getCellIndexes(gradientImage, CELL_SIZE)
	#print("Number of cells of test image:", len(cell_indexes))

	num_cells_i = int(gradientImage.shape[I] / CELL_SIZE[I])
	num_cells_j = int(gradientImage.shape[J] / CELL_SIZE[J])
	cell_matrix = [[0 for i in range(num_cells_j)] for j in range(num_cells_i)]
	cell_index = 0
	for i in range(num_cells_i):
		for j in range(num_cells_j):
			cell_matrix[i][j] = createOrientationBinning(NUM_BINS, cell_indexes[cell_index], gradientImage)
			cell_index += 1
	return cell_matrix

def getHog(cells_matrix):
	num_cells_i = len(cells_matrix)
	num_cells_j = len(cells_matrix[0])
	num_blocks_i = num_cells_i - CELLS_PER_BLOCK[I] + 1
	num_blocks_j = num_cells_j - CELLS_PER_BLOCK[J] + 1

	hog = np.zeros((num_blocks_i * num_blocks_j * CELLS_PER_BLOCK[I] * CELLS_PER_BLOCK[J] * NUM_BINS))
	hog_index = 0
	for j in range(num_blocks_j):
		for i in range(num_blocks_i):
			cell_top_left = (i, j)
			cell_bottom_right = (i + CELLS_PER_BLOCK[I], j + CELLS_PER_BLOCK[J])
			block = np.zeros((CELLS_PER_BLOCK[I] * CELLS_PER_BLOCK[J] * NUM_BINS))
			cell_index = 0
			for cell_j in range(cell_top_left[J], cell_bottom_right[J]):
				for cell_i in range(cell_top_left[I], cell_bottom_right[I]):
					block[cell_index:cell_index+NUM_BINS] = cells_matrix[cell_i][cell_j]
					cell_index += NUM_BINS
			norm = LA.norm(block)
			if norm != 0:
				block /= LA.norm(block)
			hog[hog_index:hog_index + len(block)] = block
			hog_index += len(block)
	return hog

def train_image(image):
	height = image.shape[0]
	width = image.shape[1]
	gradientXImage = computeCenteredXGradient(image)
	gradientYImage = computeCenteredYGradient(image)
	gradientImage = computeCenteredGradient(gradientXImage, gradientYImage)

	cells_matrix = getOrientationBinMatrix(gradientImage)
	hog = getHog(cells_matrix)
	print(len(hog))

def train(paths, dataset):
	i=0
	for i in range(len(dataset)):
		image = dataset[i]
		path = paths[i]
		#print("Image:", path)
		train_image(image)
		break

def main():
	(paths, images) = readDataset(DATASET_PATH)
	print("Read dataset of", len(images), "images")
	#plotDataset(images)
	train(paths, images)

if __name__ == "__main__":
	main()
