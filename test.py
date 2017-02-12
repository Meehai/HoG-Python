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
from skimage.feature import hog

def our_hog_normalize_block(block, method, eps=1e-5):
	if method == 'L1':
		out = block / (np.sum(np.abs(block)) + eps)
	elif method == 'L1-sqrt':
		out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
	elif method == 'L2':
		out = block / np.sqrt(np.sqrt(np.sum(block ** 2)) ** 2 + eps ** 2)
	elif method == 'L2-Hys':
		out = block / np.sqrt(np.sqrt(np.sum(block ** 2)) ** 2 + eps ** 2)
		out = np.minimum(out, 0.2)
		out = out / np.sqrt(np.sqrt(np.sum(block ** 2)) ** 2 + eps ** 2)
	else:
		raise ValueError('Selected block normalization method is invalid.')

	return out

def our_cell_hog(magnitude, orientation, orientation_start, orientation_end, cell_columns, cell_rows, \
	column_index, row_index, size_columns, size_rows, range_rows_start, range_rows_stop, range_columns_start, range_columns_stop):
    #cdef int cell_column, cell_row, cell_row_index, cell_column_index
    #cdef float total = 0.

    total = float(0)

    for cell_row in range(range_rows_start, range_rows_stop):
        cell_row_index = row_index + cell_row
        if (cell_row_index < 0 or cell_row_index >= size_rows):
            continue

        for cell_column in range(range_columns_start, range_columns_stop):
            cell_column_index = column_index + cell_column
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[cell_row_index, cell_column_index]
                    >= orientation_start
                    or orientation[cell_row_index, cell_column_index]
                    < orientation_end):
                continue

            total += magnitude[cell_row_index, cell_column_index]
    return total / (cell_rows * cell_columns)

def our_hog_histogram(gradient_columns, gradient_rows, cell_columns, cell_rows, size_columns, size_rows, number_of_cells_columns, number_of_cells_rows, \
	number_of_orientations, orientation_histogram):
 
	magnitude = np.hypot(gradient_columns, gradient_rows)
	#orientation = np.arctan2(gradient_rows, gradient_columns) * (180 / np.pi) % 180
	orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180
	#cdef int i, x, y, o, yi, xi, cc, cr, x0, y0, \
	#   range_rows_start, range_rows_stop, \
	#    range_columns_start, range_columns_stop
	#cdef float orientation_start, orientation_end, \
	#    number_of_orientations_per_180

	y0 = int(cell_rows / 2)
	x0 = int(cell_columns / 2)
	cc = cell_rows * number_of_cells_rows # number of pixels per lines
	cr = cell_columns * number_of_cells_columns # no of px per col
	range_rows_stop = int(cell_rows / 2)
	range_rows_start = -range_rows_stop
	range_columns_stop = int(cell_columns / 2)
	range_columns_start = -range_columns_stop
	number_of_orientations_per_180 = 180. / number_of_orientations

	for i in range(number_of_orientations):
		# isolate orientations in this range
		orientation_start = number_of_orientations_per_180 * (i + 1)
		orientation_end = number_of_orientations_per_180 * i
		x = x0
		y = y0
		yi = 0
		xi = 0

		#print(cc, cr)
		#print (cc/ cell_rows, cr / cell_columns)
		while y < cc:
			xi = 0
			x = x0

			while x < cr:
				orientation_histogram[yi, xi, i] = \
					our_cell_hog(magnitude, orientation,
							 orientation_start, orientation_end,
							 cell_columns, cell_rows, x, y,
							 size_columns, size_rows,
							 range_rows_start, range_rows_stop,
							 range_columns_start, range_columns_stop)
				xi += 1
				x += cell_columns

			yi += 1
			y += cell_rows

def our_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
		block_norm='L1', transform_sqrt=False,
		feature_vector=True, normalise=None):

	if image.dtype.kind == 'u':
		# convert uint image to float
		# to avoid problems with subtracting unsigned numbers in np.diff()
		image = image.astype('float')

	#print("Image shape", image.shape)
	gx = np.empty(image.shape, dtype=np.double)
	gx[:, 0] = 0
	gx[:, -1] = 0
	gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
	gy = np.empty(image.shape, dtype=np.double)
	gy[0, :] = 0
	gy[-1, :] = 0
	gy[1:-1, :] = image[2:, :] - image[:-2, :]
	#print(gx, gy)

	sy, sx = image.shape
	cx, cy = pixels_per_cell
	bx, by = cells_per_block

	#n_cellsx = int(sx // cx)  # number of cells in x
	#n_cellsy = int(sy // cy)  # number of cells in y
	n_cellsx = int(np.floor(sx // cx))  # number of cells in x
	n_cellsy = int(np.floor(sy // cy))  # number of cells in y
	#print("N cells:", n_cellsx, n_cellsy)

	# compute orientations integral images
	orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))

	our_hog_histogram(gx, gy, cx, cy, sx, sy, n_cellsx, n_cellsy, orientations, orientation_histogram)
	#print("noi:", orientation_histogram[0])
	#_hoghistogram.hog_histograms(gx, gy, cx, cy, sx, sy, n_cellsx, n_cellsy, orientations, orientation_histogram)


	# now compute the histogram for each cell
	hog_image = None
	n_blocksx = (n_cellsx - bx) + 1
	n_blocksy = (n_cellsy - by) + 1
	#print("N_blocks_y", n_blocksy, "n_blocks_x", n_blocksx, ". by", by, "bx", bx, "no orientations", orientations)
	normalized_blocks = np.zeros((n_blocksy, n_blocksx,
								  by, bx, orientations))

	eps = 1e-8
	index = 0
	for x in range(n_blocksx):
		for y in range(n_blocksy):
			block = orientation_histogram[y:y + by, x:x + bx, :]
			#normalized_blocks[y, x, :] = \
			#	our_hog_normalize_block(block, method=block_norm)
			eps = 1e-5
			normalized_blocks[y, x, :] = block / np.sqrt(block.sum() ** 2 + eps)

			'''
			for z in range(block.shape[0]):
				for q in range(block.shape[1]):
					for t in range(block.shape[2]):
						index += 1
						if index == 33558:
							indx = x * bx + z
							indy = y * by + q
							print("HerE:", indx,indy, block[z][q])
			'''
	if feature_vector:
		normalized_blocks = normalized_blocks.ravel()

	return normalized_blocks

def main():
	image = np.array(Image.open(sys.argv[1]).convert("L"))
	image = image[0:8,0:8]
	original_hog = hog(image, cells_per_block=(1,1))
	result = our_hog(image, cells_per_block=(1,1))
	diff = np.abs(original_hog - result)
#	argDiff = np.argmax(diff)
#	print(argDiff, original_hog[argDiff], result[argDiff])
	print(np.max(diff))
#	print(diff)
	#print(original_hog[100:200])
	#print(original_hog[0:9])
	#print(len(original_hog))

if __name__ == "__main__":
	main()