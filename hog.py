import os
from Mihlib import *

DATASET_PATH="INRIAPerson/test_64x128_H96/pos"
CELL_SIZE=(8, 8) # 8x8 shape
CELLS_PER_BLOCK=(2, 2) # 2x2 cells in a block
NUM_BINS=18 # 18 bins over 360 degrees

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

def main():
	images = readDataset(DATASET_PATH)
	print("Read dataset of ", len(images), "images")
	#plotDataset(images)

	image = images[0]
	cell_indexes = getCellIndexes(image, CELL_SIZE)
	print("Number of cells of test image:", len(cell_indexes))

	#print(getCellIndexes(test_image, CELL_SIZE))
	gradient_values = [[(0,0) for j in range(image.shape[J])] for i in range(image.shape[I])]
	print(createOrientationBinning(image, NUM_BINS, cell_indexes[0], gradient_values))


if __name__ == "__main__":
	main()