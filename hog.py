import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import datetime
from os.path import isfile, join
from PIL import Image
from numpy import linalg as LA
from sklearn import svm
from sklearn.externals import joblib

TRAIN_POS_PATH="INRIAPerson/train_64x128_H96/pos"
TRAIN_NEG_PATH="INRIAPerson/train_64x128_H96/neg"
TEST_POS_PATH="INRIAPerson/test_64x128_H96/neg"
TEST_NEG_PATH="INRIAPerson/test_64x128_H96/neg"
CELL_SIZE=(8, 8) # 8x8 shape
CELLS_PER_BLOCK=(2, 2) # 2x2 cells in a block
NUM_BINS=18 # 18 bins over 360 degrees
DETECTION_WINDOW=(128,64)
EXPORT_FILENAME="svm.pkl"
EXPORT_FILENAME_HARD="svm_hard.pkl"
I=0
J=1

def computeCenteredXGradient(grayscaleImage):
	height = grayscaleImage.shape[0]
	width = grayscaleImage.shape[1]
	gradientXImage = np.zeros(DETECTION_WINDOW, dtype=int)
	
	centeredKernel = np.array([-1, 0, 1])
	uncenteredKernel = np.array([-1, 1])
	
	start_height = int((height - DETECTION_WINDOW[I]) / 2)
	end_height = start_height + DETECTION_WINDOW[I]

	for i in range(start_height, end_height):
		start_width = int((width - DETECTION_WINDOW[J]) / 2)
		end_width = start_width + DETECTION_WINDOW[J]
	
		for j in range(start_width, end_width):
			if j == start_width:
				gradientXImage[i-start_height][j-start_width] = np.dot(grayscaleImage[i][j: j + 2], uncenteredKernel)
			elif j == end_width - 1:
				gradientXImage[i-start_height][j-start_width] = np.dot(grayscaleImage[i][j - 1: j + 1], uncenteredKernel)
			else:
				gradientXImage[i-start_height][j-start_width] = np.dot(grayscaleImage[i][j - 1: j + 2], centeredKernel)
	return gradientXImage

def computeCenteredYGradient(grayscaleImage):
	grayscaleImage = np.transpose(grayscaleImage)
	height = grayscaleImage.shape[0]
	width = grayscaleImage.shape[1]
	gradientYImage = np.zeros((DETECTION_WINDOW[J], DETECTION_WINDOW[I]), dtype=int)

	centeredKernel = np.array([-1, 0, 1])
	uncenteredKernel = np.array([-1, 1])

	start_height = int((height - DETECTION_WINDOW[J]) / 2)
	end_height = start_height + DETECTION_WINDOW[J];

	for i in range(start_height, end_height):
		start_width = int((width - DETECTION_WINDOW[I]) / 2)
		end_width = start_width + DETECTION_WINDOW[I]
	
		for j in range(start_width, end_width):
			if j == start_width:
				gradientYImage[i-start_height][j-start_width] = np.dot(grayscaleImage[i][j: j + 2], uncenteredKernel)
			elif j == end_width - 1:
				gradientYImage[i-start_height][j-start_width] = np.dot(grayscaleImage[i][j - 1: j + 1], uncenteredKernel)
			else:
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

def readDataset(positive_train_path, negative_train_path):
	result = {"pos":[], "neg":[]}
	positive_files = [f for f in os.listdir(positive_train_path) if isfile(join(positive_train_path, f))]
	negative_files = [f for f in os.listdir(negative_train_path) if isfile(join(negative_train_path, f))]
	positive_files = positive_files #[0:100]
	negative_files = negative_files #[0:10]

	for image_name in positive_files:
		image_path = positive_train_path + os.sep + image_name
		image = np.array(Image.open(image_path).convert("L"))
		result["pos"].append(image)
	for image_name in negative_files:
		image_path = negative_train_path + os.sep + image_name
		image = np.array(Image.open(image_path).convert("L"))
		result["neg"].append(image)
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

def getHogDescriptor(cells_matrix):
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

def getFeaturesNumber():
	cells_per_detection_window = (int(DETECTION_WINDOW[I]/CELL_SIZE[I]), int(DETECTION_WINDOW[J]/CELL_SIZE[J]))
	blocks_per_detection_window = (cells_per_detection_window[I] - CELLS_PER_BLOCK[I] + 1, \
		cells_per_detection_window[J] - CELLS_PER_BLOCK[J] + 1)
	overlapping_cells_per_window = blocks_per_detection_window[I] * blocks_per_detection_window[J] * \
		CELLS_PER_BLOCK[I] * CELLS_PER_BLOCK[J]
	return overlapping_cells_per_window * NUM_BINS

# Returns A list of indexes of windowses from the initial image, given a window size and shift value
def getSubWindows(image, subWindowSize = DETECTION_WINDOW, shift = 8):
	height = image.shape[0]
	width = image.shape[1]
	heightRange = int(np.ceil((height - subWindowSize[0] + 1) / shift))
	widthRange = int(np.ceil((width - subWindowSize[1] + 1) / shift))
	subWindows = [[(0, 0) for i in range(2)] for j in range(widthRange * heightRange)]
	for i in range(0, height - subWindowSize[0] + 1, shift):
		for j in range(0, width - subWindowSize[1] + 1, shift):
			topLeft = (i, j)
			bottomRight = (i + subWindowSize[0], j + subWindowSize[1])
			subWindowsRow = int(i / shift)
			subWindowsCol = int(j / shift)
			subWindows[subWindowsRow * widthRange + subWindowsCol][0] = topLeft
			subWindows[subWindowsRow * widthRange + subWindowsCol][1] = bottomRight
	return subWindows

# Given a svm classifier and a set of negative examples, construct windows in each negative example and predict the
# result of the classifier on those. The ones that will give us a positive class will be added to the hard negatives
# list which is returned
def get_hard_negatives(svm, negative_set):
	hard_negatives = []
	descriptors = []
	print("Getting hard features")
	i = 0
	for image in negative_set:
		print(i, image.shape)
		i += 1
		subWindowsIndexes = getSubWindows(image)
		percentIndexes = 0.1
		subWindowsIndexes = random.sample(subWindowsIndexes, int(len(subWindowsIndexes) * percentIndexes))
		# Create sub image
		for index in subWindowsIndexes:
			top_left = index[0]
			bottom_right = index[1]
			sub_image = np.array(image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]])
			# Save the descriptor of the sub image.
			descriptor = train_image(sub_image)
			descriptors.append(descriptor)

	print("Testing on ", len(descriptors), "negative inputs")
	svm_result = svm.predict(descriptors)
	print("Result size:", len(svm_result))
	for i in range(len(svm_result)):
		if svm_result[i] == "pos":
			hard_negatives.append(descriptors[i])
	return hard_negatives

def train_image(image):
	height = image.shape[0]
	width = image.shape[1]
	gradientXImage = computeCenteredXGradient(image)
	gradientYImage = computeCenteredYGradient(image)	
	gradientImage = computeCenteredGradient(gradientXImage, gradientYImage)
	
	cells_matrix = getOrientationBinMatrix(gradientImage)
	hog_descriptor = getHogDescriptor(cells_matrix)
	return hog_descriptor

def svm_classify(svm, data, classes, message="RBF"):
	good = 0
	result = svm.predict(data)
	assert(len(result) == len(classes))
	for i in range(len(result)):
		if result[i] == classes[i]:
			good += 1
	print("[", message, "] Accuracy on training set: ", good / len(result), sep="")

def prepare_data_for_svm(dataset, random_negative_windows_count=10):
	data = []
	classes = []
	
	print("Preparing descriptors for", len(dataset["pos"]), "positive images and", len(dataset["neg"]), "negative")

	print("Getting descriptor for positive images")
	for image in dataset["pos"]:
		descriptor = train_image(image)
		data.append(descriptor)
		classes.append("pos")

	print("Getting descriptor for negative images")
	i = 0
	for image in dataset["neg"]:
		i += 1
		descriptor = train_image(image)
		# Get random windows in each negative image for training.
		print(i, image.shape)
		subWindowsIndexes = getSubWindows(image)
		subWindowsIndexes = random.sample(subWindowsIndexes, random_negative_windows_count)
		for index in subWindowsIndexes:
			top_left = index[0]
			bottom_right = index[1]
			sub_image = np.array(image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]])
			descriptor = train_image(sub_image)
			data.append(descriptor)
			classes.append("neg")
	return (data, classes)

def train(dataset):
	(data, classes) = prepare_data_for_svm(dataset)
	# Train initial SVM
	C = 1.0
	print("Training initial SVM with", len(dataset["pos"]), "positives and", \
		len(data) - len(dataset["pos"]), "negatives")

	# Testing all 4 types of SVMs and printing the accuracy by testing on the training set
	rbf_svc = svm.SVC(kernel="rbf", gamma=0.7, C=C).fit(data, classes)
	svm_classify(rbf_svc, data, classes, "RBF")
	linear_svc = svm.SVC(kernel="linear").fit(data, classes)
	svm_classify(linear_svc, data, classes, "Linear")
	polynomial_svc = svm.SVC(kernel="poly").fit(data, classes)
	svm_classify(polynomial_svc, data, classes, "Polynomial")
	sigmoid_svc = svm.SVC(kernel="sigmoid").fit(data, classes)
	svm_classify(sigmoid_svc, data, classes, "Sigmoid")

	# Export the trained RBF SVM to current.
	print("Exporting SVM to", EXPORT_FILENAME)
	joblib.dump(rbf_svc, EXPORT_FILENAME) 

	# Get hard negatives and add them to the negative training set and then re-train the SVM with them as well.
	hard_negatives = get_hard_negatives(rbf_svc, dataset["neg"])
	print("Returned", len(hard_negatives), "hard negatives. Adding them to negative dataset")	
	data.extend(hard_negatives)
	classes.extend(["neg"] * len(hard_negatives))
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(data, classes)
	test_on_training_set("RBF", rbf_svc, data, classes)

	# Export the re-trained SVM with hard negatives.
	print("Exporting SVM to", EXPORT_FILENAME_HARD)
	joblib.dump(rbf_svc, EXPORT_FILENAME_HARD)

	return rbf_svc

def main():
	dataset = readDataset(TRAIN_POS_PATH, TRAIN_NEG_PATH)
	print("Read test dataset of", len(dataset["pos"]), "positive images and", len(dataset["neg"]), "negative images")

	if len(sys.argv) == 2:
		svm_classifier = joblib.load(sys.argv[1])
		print("SVM classifier imported from:", sys.argv[1])
	else:
		svm_classifier = train(dataset)

	# Read the testing dataset and run it.
	dataset = readDataset(TEST_POS_PATH, TEST_NEG_PATH)
	(data, classes) = prepare_data_for_svm(dataset)
	svm_classify(svm_classifier, data, classes)

if __name__ == "__main__":
	main()
