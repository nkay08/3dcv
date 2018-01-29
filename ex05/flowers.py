import cv2 as cv
import numpy as np
import scipy.io as io
import os

# Global varialbles

"""
Since the dataset is of high resolution we will work with reduced version.
You can choose the reduction factor using the SCALE_FACTOR variable.
"""
SCALE_FACTOR = 3

"""
Since our stereo matching  algorithms is going to be slow we will reconstruct only part of the scene.
You can limit the section to be reconstructed using the following two tuples
-- For the whole image use
H_BOUND = (0, 660)
W_BOUND = (0, 960)

-- To experiment faster use the a smaller range like
H_BOUND = (350, 650)
W_BOUND = (60, 230)
"""

H_BOUND = (400, 650)
W_BOUND = (40, 230)

# Minimum and Maximum disparies(dataset specific parameters)
DISPARITY_BOUND = (int(31/SCALE_FACTOR), int(257/SCALE_FACTOR))
# Disparity offset: pixel difference in c_x1-c_x2 of the two cameras
DOFFS = (1307.839/SCALE_FACTOR - 1176.728/SCALE_FACTOR)
# Focal length
F = 3997.684/SCALE_FACTOR
BASELINE=193.001

K_LEFT = np.asarray([[3997.684/SCALE_FACTOR, 0, 1176.728/SCALE_FACTOR], [0, 3997.684/SCALE_FACTOR, 1011.728/SCALE_FACTOR], [0, 0, 1]])
K_RIGHT = np.asarray([[3997.684/SCALE_FACTOR, 0, 1307.839/SCALE_FACTOR], [0, 3997.684/SCALE_FACTOR, 1011.728/SCALE_FACTOR], [0, 0, 1]])

SIMILARITY_MERTIC = 'ncc'
# SIMILARITY_MERTIC = 'ssd'

# Outlier Filtering Threshold,
# You can test other values, too
# This is usually a parameter which you have to select carefully for each dataset
#
O_F_THRESHOLD = 3

# Patch Size
# TODO: Experiment with other values like 3, 13, 17 and observe the result
K_SIZE = 7

# Give a folder name for your experiment to keep results of each trial separate
EXP_NAME = 'my_exp/'
os.system('mkdir ./' + EXP_NAME)

# The following three variables will be used by the ply creation function,
pre_text1 = """ply
format ascii 1.0"""
pre_text2 = "element vertex "
pre_text3 = """property float x
property float y
property float z
end_header"""
def ply_creator(input_, filename):
	assert (input_.ndim==2),"Pass 3d points as NumPointsX3 array "
	pre_text22 = pre_text2 + str(input_.shape[0])
	pre_text11 = pre_text1
	pre_text33 = pre_text3
	fid = open(filename + '.ply', 'w')
	fid.write(pre_text11)
	fid.write('\n')
	fid.write(pre_text22)
	fid.write('\n')
	fid.write(pre_text33)
	fid.write('\n')
	for i in range(input_.shape[0]):
		# Check if the depth is not set to zero
		if input_[i,2]!=0:
			for c in range(2):
				fid.write(str(input_[i,c]) + ' ')
			fid.write(str(input_[i,2]))
			if i!=input_.shape[0]-1:
				fid.write('\n')
	fid.close()
	return True
"""
sub_pixel_crop() and pixel_interpolate(): are provided incase you want to do sub pixel matching.
sub_pixel_crop() will allow you to crop a patch centered at a floating point location like (10.89, 3.88).
Even if you are not cropping at sub pixel locations you can use them.
"""
def pixel_interpolate(img, h_idx, w_idx):
	"""
	- img: input image
	- h_idx, w_idx: center of the patch ( in the vertical and horizontal dimensions, respectively)
				  h_idx & w_idx could be integer of floating values.
	- pixel_out: output pixel after applying bilinear interpolation
	"""
	pixel_out = np.zeros((3,))
	h_s = np.int(h_idx)
	w_s = np.int(w_idx)
	for h in range(h_s, h_s+2):
		for w in range(w_s, w_s+2):
			weight = (1-np.abs(h_idx-h))*(1-np.abs(w_idx-w))
			pixel_out += weight*img[h,w,:]
	return pixel_out

def sub_pixel_crop(img, h_center, w_center, K_SIZE):
	"""
	- img: input image
	- h_center, w_center: center of the patch ( in the vertical and horizontal dimensions, respectively)
				  h_center & w_center could be integer of floating values.
	- K_SIZE: kernel width/ and height/
	- crop_out: output patch of size kxk
	"""
	crop_out = np.zeros((K_SIZE, K_SIZE, 3))
	offset = np.floor(K_SIZE/2)
	h_idxs = np.linspace(h_center-offset, h_center+offset, K_SIZE)
	w_idxs = np.linspace(w_center-offset, w_center+offset, K_SIZE)
	for h in range(len(h_idxs)):
		for w in range(len(w_idxs)):
			crop_out[h,w,:] = pixel_interpolate(img, h_idxs[h], w_idxs[w])
	return crop_out

def copy_make_border(img, K_SIZE):
	"""
	Patches/windows centered at the border of the image need additional padding of size K_SIZE/2
	This function applies cv.copyMakeBorder to extend the image by K_SIZE/2 in top, bottom, left and right part of the image
	"""
	offset = np.int(K_SIZE/2.0)
	return cv.copyMakeBorder(img, top=offset, bottom=offset, left=offset, right=offset, borderType=cv.BORDER_REFLECT)

def write_depth_to_image(depth, f_name):
	"""
	This function writes depth map to f_name
	"""
	#assert (input_.ndim==2),"Depth map should be a 2D array "
	max_depth = np.max(depth)
	min_depth = np.min(depth)
	depth_v1 = 255 - 255*((depth-min_depth)/max_depth)
	#depth_v2 = 255*((depth-min_depth)/max_depth)
	cv.imwrite(f_name, depth_v1)
	return True

def depth_to_3d(Z, y, x):
	"""
	Given the pixel position(y,x) and depth(Z)
	It computes the 3d point in world coordinates,
	first back-project the point from homogeneous image space to 3D,  by multiplying it with inverse of the camera intrinsic matrix,  inv(K)
	Then scale it so that you will get the point at depth equal to Z.
	the scale the vector
	"""
	X_W = Z*np.dot(np.linalg.inv(K_LEFT),np.asarray([[x],[y],[1]]))
	X_W = X_W.reshape(3,1)
	# returns a list [X, Y, Z]
	return [X_W[0,0], X_W[1,0], X_W[2,0]]

def is_outlier(score_vector):
	"""
	O_F_THRESHOLD is the Outlier Filtering Threshold.
	For 'ncc' metric: if more than O_F_THRESHOLD number of disparity values have ncc score greater of equal to 0.8*max_score, the function returns True.
	For 'ssd' metric: if more than O_F_THRESHOLD number of disparity values have ssd score less than of equal to 1.4*min_score, the function returns True.
	The above thresholds are chosen by the TAs. Feel free, to try out different values.
	- The input to this function should be a 1D array of ncc scores computed for all disparities.
	"""
	assert (score_vector.ndim==1), "Pass 1D array of vector of scores "
	if SIMILARITY_MERTIC=='ncc':
		max_score = np.max(score_vector)
		if np.sum(score_vector>=0.8*max_score)>=O_F_THRESHOLD:
			return True
		else:
			return False
	if SIMILARITY_MERTIC=='ssd':
		min_distance = np.min(score_vector)
		min_score=min_distance
		if np.sum(score_vector<=1.4*min_score)>=O_F_THRESHOLD:
			return True
		else:
			return False
	return True

def disparity_to_depth(disparity):
	"""
	Converts disparity to depth.
	"""
	inv_depth = (disparity+DOFFS)/(BASELINE*F)
	return 1/inv_depth

def ncc(patch_1, patch_2):
	"""
	TODO
	1. Normalise the input patch by subtracting its mean and dividing by its standard deviation.
	Perform the normalisation for each color channel of the input patch separately.
	2. Reshape each normalised image patch into a 1D feature vector and then
	compute the dot product between the resulting normalised feature vectors.
	"""
	raise NotImplementedError

def ssd(feature_1, feature_2):
	"""
	TODO
	Compute the sum of square difference between the input features
	"""

	raise NotImplementedError

def stereo_matching(img_left, img_right, K_SIZE, disp_per_pixel):
	"""
	TODO
	This is the main function that you have to write.
	For the region of the left image delimited by H_BOUND and H_BOUND, this section should do the following.
		1. Compute Depth-Map and save it as a gray scale
		2. Create point cloud by giving depth map to depth_to_3d() function.
		3. Using ply_creator(), save point cloud:
	Note that: the left and right_images are padded; by int(K_SIZE/2) on the left, right, top and bottom.
	"""
	# TASKs:
	# Iterate over every pixel of the left image (with in the region bounded by H_BOUND and W_BOUND), (lets denote a sample as P_x_y)
		# For every P_x_y:
		 # compute disparity using ncc cost metric
		 # Convert disparity to depth, use disparity_to_depth()
		 # Convert depth to point cloud, use depth_to_3d()
	# Save point clouds as .ply and depth as .png files
	#
	#for every row in im_left
	dispMap=np.zeros((H_BOUND[1]-H_BOUND[0],W_BOUND[1]-W_BOUND[0]))
	for i in range(H_BOUND[0],H_BOUND[1]):
		#for every column in im_left
		#->for every pixel in im_left within bounds
		for j0 in range(W_BOUND[0],W_BOUND[1]):
			P0 = img_left[i,j0]

			#for every pixel in im_left look for pixel in right image on same row

			for j1 in range(W_BOUND[0],W_BOUND[1]):
				P1 = img_right[i,j1]
				ssd= ssd(P0,P1)
	#cv.imshow("disp",dispMap)
	#cv.waitKey(0)
	#raise NotImplementedError
	return True

def pixelValue(pixel):
	weightR= 0.3
	weightG=0.6
	weightB=0.11
	return(weightR * pixel[0]+weightG * pixel[1]+ weightB * pixel[2])

def main():
	# Set parameters
	# Use 1: for pixel-wise disparity
	# Optional: you can try to use sub-pixel level disparity, using, disp_per_pixel = n, n>1
	# Do not try this until you see the result for n=1
	disp_per_pixel = 1
	# Read images and expand borders
	# load Left image
	l_file = './data/flowers_perfect/im0.png'
	l_im = cv.imread(l_file)
	resized_l_img = cv.pyrDown(l_im, SCALE_FACTOR)
	left_img = copy_make_border(resized_l_img, K_SIZE)
	# load Right image
	r_file = './data/flowers_perfect/im1.png'
	r_im = cv.imread(r_file)
	resized_r_img = cv.pyrDown(r_im, SCALE_FACTOR)
	right_img = copy_make_border(resized_r_img, K_SIZE)
	# TODO: #1 fill in the stereo_matching() function called below
	dummy = stereo_matching(left_img, right_img, K_SIZE, disp_per_pixel)
main()
