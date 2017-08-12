
# coding: utf-8

# In[2]:


# Imports

import numpy as np
import cv2
import glob # Used to read in image files of a particular pattern
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
import random
import pickle
import collections # Used to store a recent window of good fits
import math # Used for nan detection
import sys # For progress indicator
import time # For time difference measurements

# SciKitLearn
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# Packages below needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from scipy.ndimage.measurements import label
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage.measurements import label


# In[3]:



# Shared Constants
# Constant for separating log statements( if needed )
const_separator_line = "--------------------------------"

# Constant representing a random seed for Data Splitting
const_data_split_seed = 42

# Constant representing location of the trained model
# The model name(s) are unique as per the color spaces they were trained to operate upon
const_persist_data_rgb2 = './model/persist_data_rgb2.p'
const_persist_data_rgb = './model/persist_data_rgb.p'
const_persist_data_hsv = './model/persist_data_hsv.p'
const_persist_data_hls = './model/persist_data_hls.p'
const_persist_data_ycrcb = './model/persist_data_ycrcb.p'
const_persist_data_yuv = './model/persist_data_yuv.p'
const_persist_data_luv = './model/persist_data_yuv.p'

# Constants representing paths of test images
const_test_straight1 = './test_images/input/straight_lines1.jpg'
const_test_straight1_output = './test_images/output/straight_lines1.jpg'

const_test_straight2 = './test_images/input/straight_lines2.jpg'
const_test_straight2_output = './test_images/output/straight_lines2.jpg'

const_test_image_1 = './test_images/input/test1.jpg'
const_test_image_1_output = './test_images/output/test1.jpg'

const_test_image_2 = './test_images/input/test2.jpg'
const_test_image_2_output = './test_images/output/test2.jpg'

const_test_image_3 = './test_images/input/test3.jpg'
const_test_image_3_output = './test_images/output/test3.jpg'

const_test_image_4 = './test_images/input/test4.jpg'
const_test_image_4_output = './test_images/output/test4.jpg'

const_test_image_5 = './test_images/input/test5.jpg'
const_test_image_5_output = './test_images/output/test5.jpg'

const_test_image_6 = './test_images/input/test6.jpg'
const_test_image_6_output = './test_images/output/test6.jpg'

# Constants representing paths of a project video
const_project_video = './project_video/input/project_video.mp4'
const_project_video_output = './project_video/output/project_video_output.mp4'

# Constants representing paths of a test video
const_test_video = './test_video/input/test_video.mp4'
const_test_video_output = './test_video/output/test_video_output.mp4'

# Constants representing paths of model to train and test
## Non-Vehicles
nonvehicles_gti = glob.glob('./project_data/non-vehicles/GTI/image*.png')
nonvehicles_extras = glob.glob('./project_data/non-vehicles/Extras/extra*.png')

## Vehicles
vehicles_gti_far = glob.glob('./project_data/vehicles/GTI_Far/image*.png')
vehicles_gti_left = glob.glob('./project_data/vehicles/GTI_Left/image*.png')
vehicles_gti_middleclose = glob.glob('./project_data/vehicles/GTI_MiddleClose/image*.png')
vehicles_gti_right = glob.glob('./project_data/vehicles/GTI_Right/image*.png')
vehicles_gti_kitti = glob.glob('./project_data/vehicles/KITTI_extracted/*.png')


# In[47]:


# Useful functions to selectively turn on / off logging at different levels

const_info_log_enabled = False
def infoLog(logMessage, param_separator=None):
    if const_info_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)

const_debug_log_enabled = True
def debugLog(logMessage, param_separator=None):
    if const_debug_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)
        
const_warning_log_enabled = True
def warningLog(logMessage, param_separator=None):
    if const_warning_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)
        
const_error_log_enabled = True
def errorLog(logMessage, param_separator=None):
    if const_error_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)



# In[48]:


# Convenience function for reading an image from a path
# Provides a consistent read mechanism
# Returns an RGB image

def loadImageForPath(imagePath):
    image =  mpimg.imread(imagePath)
    return image


# In[49]:


# Convenience function to get the scale ( min, max ) for an image 
def get_image_scale(imageOrPath, isImagePath = False):
    # If the image is a path, then read the image from Path
    image = None
    if(isImagePath == True):
        image = loadImageForPath(imageOrPath) # Reads image as an RGB image
    else:
        image = imageOrPath
        
    scale_min = np.amin(image)
    scale_max = np.amax(image)
    
    # Print out Image Scale
    infoLog("Image Scale - Min : " + str(scale_min))
    infoLog("Image Scale - Max : " + str(scale_max))

    return (scale_min, scale_max)


# In[50]:


# Create a flattened array of Vehicle and Non-Vehicle Data

cars = vehicles_gti_far + vehicles_gti_left + vehicles_gti_middleclose + vehicles_gti_right + vehicles_gti_kitti
notcars = nonvehicles_gti + nonvehicles_extras


# 
# # Exploring various parameters to train the classifier
# 
# * I used the note book section below for exploration of the various parameters.
# 

# In[53]:


# This is the block where we train the SVM classifier, 
# commenting this out because we perform the training within "svm_train.py"

# car_features = extract_features(cars, color_space=color_space, 
#                         spatial_size=spatial_size, hist_bins=hist_bins, 
#                         orient=orient, pix_per_cell=pix_per_cell, 
#                         cell_per_block=cell_per_block, 
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                         hist_feat=hist_feat, hog_feat=hog_feat)

# notcar_features = extract_features(notcars, color_space=color_space, 
#                         spatial_size=spatial_size, hist_bins=hist_bins, 
#                         orient=orient, pix_per_cell=pix_per_cell, 
#                         cell_per_block=cell_per_block, 
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                         hist_feat=hist_feat, hog_feat=hog_feat)

# X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# # Fit a per-column scaler
# X_scaler = StandardScaler().fit(X)
# # Apply the scaler to X
# scaled_X = X_scaler.transform(X)

# # Define the labels vector
# y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# # Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(
#     scaled_X, y, test_size=0.2, random_state=rand_state)

# print('Using:',orient,'orientations',pix_per_cell,
#     'pixels per cell and', cell_per_block,'cells per block')
# print('Feature vector length:', len(X_train[0]))
# # Use a linear SVC 
# svc = LinearSVC()
# # Check the training time for the SVC
# t=time.time()
# svc.fit(X_train, y_train)
# t2 = time.time()
# print(round(t2-t, 2), 'Seconds to train SVC...')
# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# # Check the prediction time for a single sample
# t=time.time()

# # Save to Pickle
# persist_data_dict = { "svc": svc, "scaler" : X_scaler, "orient" : orient, "pix_per_cell" : pix_per_cell, "cell_per_block" : cell_per_block, "spatial_size" : spatial_size, "hist_bins" : hist_bins, "color_space": color_space }
# pickle.dump( persist_data_dict, open( const_persist_data, "wb" ) )

# # Load a test image
# image = mpimg.imread(const_test_image_1)
# draw_image = np.copy(image)

# # Uncomment the following line if you extracted training
# # data from .png images (scaled 0 to 1 by mpimg) and the
# # image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255

# image_height = image.shape[0]
# y_start_stop_value = [int(image_height/2), image_height] # Min and max in y to search in slide_window()

# windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_value, 
#                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))

# hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
#                         spatial_size=spatial_size, hist_bins=hist_bins, 
#                         orient=orient, pix_per_cell=pix_per_cell, 
#                         cell_per_block=cell_per_block, 
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                         hist_feat=hist_feat, hog_feat=hog_feat)                       

# window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

# plt.imshow(window_img)

# print(X.shape)


# # Perform training within "svm_train.py"
# 
# * I was doing the initial training within the notebook, but then after sometime the notebooks have a tendency to become fairly slow.
# * Whenever I was training the model, with more than 6100 data points within the notebook, I was running out of memory.
# * Because of the above, I created the python file to perform the training. 
# * In the section below, we restore the parameters with which we had performed the training in the above file. 

# In[51]:


# Restore the persisted parameters from Disk and run the pipeline on the same test image as above to ensure consistent results

persist_data_dict = pickle.load( open(const_persist_data_rgb, "rb" ) )
svc = persist_data_dict["svc"]
X_scaler = persist_data_dict["scaler"]
orient = persist_data_dict["orient"]
pix_per_cell = persist_data_dict["pix_per_cell"]
cell_per_block = persist_data_dict["cell_per_block"]
spatial_size = persist_data_dict["spatial_size"]
hist_bins = persist_data_dict["hist_bins"]
color_space = persist_data_dict["color_space"]
hog_channel = persist_data_dict["hog_channel"]
spatial_feat = persist_data_dict["spatial_feat"]
hist_feat = persist_data_dict["hist_feat"]
hog_feat = persist_data_dict["hog_feat"]

# Print these out for a quick sanity check

debugLog(const_persist_data_rgb)
debugLog(orient)
debugLog(pix_per_cell)
debugLog(cell_per_block)
debugLog(spatial_size)
debugLog(hist_bins)
debugLog(color_space)
debugLog(hog_channel)
debugLog(spatial_feat)
debugLog(hist_feat)
debugLog(hog_feat)


# In[52]:


# Define a function to return HOG features and visualization
# http://www.learnopencv.com/histogram-of-oriented-gradients/
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    bins_range = get_image_scale(img)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    print(len(imgs))
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    



# In[54]:


# Use the parameters restored from Disk ( after training do to a test run )

# Load a test image
image = mpimg.imread(const_test_image_1)
draw_image = np.copy(image)

image_shape = image.shape
debugLog("Image Shape : " + str(image_shape))
image_width = image_shape[1]
image_height = image.shape[0]

y_start_stop_value = [int(image_height/2), image_height] # Min and max in y to search in slide_window()

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_value, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

plt.imshow(window_img)


# In[55]:


def convert_color(img, color_space='RGB'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            infoLog("convert_color : color space is HSV")
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: 
        return np.copy(img)   

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     print(hog1.shape)

    bbox_list = list()
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
#             hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features_to_scale = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
#             print(test_features_to_scale.shape)
            test_features = X_scaler.transform(test_features_to_scale)    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return draw_img, bbox_list
    
    

# Use the parameters restored from Disk ( after training do to a test run )

# const_test_straight1
# const_test_straight2
# const_test_image_1
# const_test_image_2
# const_test_image_3
# const_test_image_4
# const_test_image_5
# const_test_image_6

img = mpimg.imread(const_test_image_6)

ystart = 400
ystop = 656
scale = 1
    
out_img, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)

plt.imshow(out_img)

for bbox in box_list:
    infoLog("Left top : " + str(bbox[0]) + " || Right bottom : " + str(bbox[1]) )


# In[56]:


image = mpimg.imread(const_test_image_1)
heat = np.zeros_like(image[:,:,0]).astype(np.float)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def get_bbox_list_from_labels(labels):
    cars_bbox_list = list()
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cars_bbox_list.append(bbox)
    # Return the bbox_list
    return cars_bbox_list

def get_centroid_list_from_bbox_list(bbox_list):
    centroids_list = list()
    # Iterate through all detected cars
    for bbox in bbox_list:
        (left_top_x, left_top_y) = bbox[0]
        (right_bottom_x, right_bottom_y) = bbox[1]
        centroid_x = (left_top_x + right_bottom_x) / 2
        centroid_y = (left_top_y + right_bottom_y) / 2
        centroid = (centroid_x, centroid_y)
        centroids_list.append(centroid)
    # Return the bbox_list
    return centroids_list

def draw_bbox_list_on_image(img, bbox_list, indices_list_to_draw = None, draw_color = (0,0,255), draw_thickness = 6):
    
    if indices_list_to_draw == None:   
        # Iterate through all detected cars
        for bbox in bbox_list:
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], draw_color, draw_thickness)
        # Return the image
    else:
        for indexvalue in indices_list_to_draw:
            bbox = bbox_list[indexvalue]
            cv2.rectangle(img, bbox[0], bbox[1], draw_color, draw_thickness)

    return img

def draw_labeled_bboxes(img, labels):
    cars_bbox_list = list()
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        cars_bbox_list.append(bbox)
    # Return the image
    return img, cars_bbox_list

# Add heat to each box in box list
heat = add_heat(heat,box_list)
heat_image_scale = get_image_scale(heat)
debugLog("Heat Image Scale : " + str(heat_image_scale))
    
# Detect the leval of 'hotness' for the image
max_heat = heat_image_scale[1]
    

# Rejection multiplier to pick all cells upto (thresholding_ratio*max_heat)
# If our detection pipeline is very sensitive, then this ration needs to be close to one, else
# it needs to be closer to 0
thresholding_ratio = 0.3

dynamic_threshold = int(thresholding_ratio*max_heat) # Do the thresholding dynamically

debugLog("dynamic threshold is : " + str(dynamic_threshold))
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat, dynamic_threshold)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
print(labels[1], 'cars found')

cars_bbox_list = get_bbox_list_from_labels(labels)
centroid_list = get_centroid_list_from_bbox_list(cars_bbox_list)
draw_img = draw_bbox_list_on_image(np.copy(image), cars_bbox_list)

# draw_img, cars_bbox_list = draw_labeled_bboxes(np.copy(image), labels)

debugLog(centroid_list)
debugLog(cars_bbox_list)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()


# In[57]:


const_measurements_fontsize = 1
const_measurements_fontcolor = (255,255,255)
const_measurements_fontcolor_true = (0,0,255)
const_measurements_fontcolor_false = (255,0,0)


# Convenient method to add a frame number to an image
# This can be useful to track specific frame(s) of a video for debugging
def add_frame_number_to_image(param_img, current_frame_number, true_detections_list, false_detections_list):
    return_img = param_img.copy()
    
    true_detections_count = 0
    false_detections_count = 0
    
    if true_detections_list != None:
        true_detections_count = len(true_detections_list)
        
    if false_detections_list != None:
        false_detections_count = len(false_detections_list)

    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(return_img, 'Frame Number = %d' % current_frame_number, (50, 50), font, const_measurements_fontsize, const_measurements_fontcolor, 2)
    
    cv2.putText(return_img, 'True Detections Count = %d' % true_detections_count, (50, 100), font, const_measurements_fontsize, const_measurements_fontcolor_true, 2)
    
    cv2.putText(return_img, 'False Detections Count = %d' % false_detections_count, (50, 150), font, const_measurements_fontsize, const_measurements_fontcolor_false, 2)

    return return_img


# In[58]:


# Clear history across the video pipeline

global_centroid_history_list = list() # Each entry here contains a list of car centroids as detected per frame
global_current_frame_number = 0

def reset_history():
    debugLog("Clearing any previous state.")
    global global_centroid_history_list
    global global_current_frame_number
    global_centroid_history_list = list()
    global_current_frame_number = 0
    
def update_history(centroid_list):
    debugLog("Updating history - total number of frames captured : " + str(len(global_centroid_history_list)) )
    global global_centroid_history_list
    global global_current_frame_number
    global_centroid_history_list.append(centroid_list)
    global_current_frame_number = global_current_frame_number + 1
    
def get_all_historical_centroids():
    infoLog("Fetching history - total number of frames captured : " + str(len(global_centroid_history_list)) )
    global global_centroid_history_list
    return global_centroid_history_list
    


# In[59]:


const_max_history_frames = 5 # Constant representing maximum number of historical frames to consider

const_detection_threshold_x = 60 # Threshold for the difference between x co ordinate of centroid of current frame to historical frames
const_detection_threshold_y = 60 # Threshold for the difference between y co ordinate of centroid of current frame to historical frames

# Method which gets the last 'history_depth' number of centroids from centroid_history_list.
# The returned list is a flattened list of centroids ( unline centroid_history_list which by itself contains list of centroids of detections per frame )
def get_historical_centroids(centroid_history_list, history_depth):
    infoLog(centroid_history_list)
    centroid_history_list_len = len(centroid_history_list)
    history_depth_for_lookup = history_depth
    if centroid_history_list_len < history_depth:
        history_depth_for_lookup = centroid_history_list_len
    list_of_centroid_lists = centroid_history_list[-history_depth_for_lookup:]
    infoLog(list_of_centroid_lists)
    return_centroid_list = list()
    for centroids_list in list_of_centroid_lists:
        for centroid in centroids_list:
            return_centroid_list.append(centroid)
    return return_centroid_list

# This method compares the centroid detections of the current frame to all the centroids from the previous frame
# To determine proximity, we use, const_detection_threshold_x and const_detection_threshold_y
# For each current point, we count the number of historical points that it is close to, and prepare a list of strength for all such points
# We will then apply thresholding to see if the current point is close to at-least past 'X' points, and in that case declare that as a legitimate detection
# Points that do not meet this criteria will be rejected
def compare_current_centroids_with_historical_centroids(current_centroid_list, flattened_historical_centroids):
    proximity_list_for_centroids = list()
    for centroid in current_centroid_list:
        proximity_count_for_current_centroid = 0
        for historical_centroid in flattened_historical_centroids:
            abs_delta_x = abs(historical_centroid[0] - centroid[0])
            abs_delta_y = abs(historical_centroid[1] - centroid[1])
            if abs_delta_x <= const_detection_threshold_x and abs_delta_y <= const_detection_threshold_y:
                proximity_count_for_current_centroid = proximity_count_for_current_centroid + 1
        proximity_list_for_centroids.append(proximity_count_for_current_centroid)
    return proximity_list_for_centroids

def generate_centroid_statistics(centroid_history_list, current_frame_centroids):
    debugLog("Current frame detection centroids : " + str(current_frame_centroids))
    flattened_list_of_centroid_history = get_historical_centroids(centroid_history_list, const_max_history_frames)
    debugLog("Historical centroids : " + str(flattened_list_of_centroid_history))
    proximity_list_for_current_centroids = compare_current_centroids_with_historical_centroids(current_frame_centroids, flattened_list_of_centroid_history)
    return proximity_list_for_current_centroids
    
def separate_centroids_into_true_false(proximity_list_for_current_centroids):
    if proximity_list_for_current_centroids == None:
        return None, None
    
    max_proximity_value = max(proximity_list_for_current_centroids)
    infoLog(max_proximity_value)
    
    indices_list_true_detections = list()
    indices_list_false_detections = list()
    
    for index, proximity in enumerate(proximity_list_for_current_centroids):
        if proximity == max_proximity_value:
            indices_list_true_detections.append(index)
        else: 
            indices_list_false_detections.append(index)
    return indices_list_true_detections, indices_list_false_detections
    

    


# In[60]:


# Defining the pipeline

def pipeline(imageOrPath, isImagePath = False):
    
    global global_current_frame_number
    
    debugLog("-----------------------------------------------------------------------------------------------")
    
    debugLog("Current Frame Number : " + str(global_current_frame_number))

    # If the image is a path, then read the image from Path
    image = None
    if(isImagePath == True):
        image = mpimg.imread(imageOrPath) # Reads image as an RGB image
    else:
        image = imageOrPath
        
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Clear out matplotlib plot frame for each run
    plt.clf()
            
    # Detect Cars
    out_img, box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
                 
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)
    
    # Detect the leval of 'hotness' for the image
    heat_image_scale = get_image_scale(heat)
    debugLog("Heat Image Scale : " + str(heat_image_scale))
    max_heat = heat_image_scale[1]
    
    # Do the thresholding dynamically
    dynamic_threshold = int(thresholding_ratio*max_heat)

    debugLog("Dynamic threshold for this frame is : " + str(dynamic_threshold))
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, dynamic_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    num_cars_detected = labels[1]
    debugLog('Number of Cars found in current frame : ' +  str(num_cars_detected))    
    
    cars_bbox_list = get_bbox_list_from_labels(labels)
    
    current_frame_centroids_list = get_centroid_list_from_bbox_list(cars_bbox_list)
    
    infoLog("Detected Cars Bounding Box List : " + str(cars_bbox_list))

    proximity_list_for_current_frame_centroids = generate_centroid_statistics(get_all_historical_centroids(), current_frame_centroids_list)
    debugLog(proximity_list_for_current_frame_centroids)
    
    true_detections_indices_list, false_detections_indices_list = separate_centroids_into_true_false(proximity_list_for_current_frame_centroids)
    
    debugLog("True Detections Indices List : " + str(true_detections_indices_list))
    debugLog("False Detections Indices List : " + str(false_detections_indices_list))

    draw_img_true = draw_bbox_list_on_image(np.copy(image), cars_bbox_list, true_detections_indices_list)
    
    draw_img_true_and_false = draw_bbox_list_on_image(draw_img_true, cars_bbox_list, false_detections_indices_list, (255, 0,0), 1)
    
    # Items to log / keep a record of - num_cars_detected, cars_bbox_list, heat_image_scale
    draw_img_true_and_false = add_frame_number_to_image(draw_img_true_and_false, global_current_frame_number, true_detections_indices_list, false_detections_indices_list)
    
    update_history(current_frame_centroids_list)
    
    debugLog("-----------------------------------------------------------------------------------------------")

    return draw_img_true_and_false

# Pipeline as executed on a test image

reset_history()
detected_cars_img = pipeline(const_test_image_1, True)
plt.imshow(detected_cars_img)
cv2.imwrite(const_test_image_1_output, detected_cars_img)


# In[61]:


# Pipeline as executed on Test Image #2 

reset_history()
detected_cars_img = pipeline(const_test_image_2, True)
plt.imshow(detected_cars_img)
cv2.imwrite(const_test_image_2_output, detected_cars_img)


# In[131]:


# Pipeline as executed on Test Image #3 

reset_history()
detected_cars_img = pipeline(const_test_image_3, True)
plt.imshow(detected_cars_img)
cv2.imwrite(const_test_image_3_output, detected_cars_img)




# In[132]:



# Pipeline as executed on Test Image #4 

reset_history()
detected_cars_img = pipeline(const_test_image_4, True)
plt.imshow(detected_cars_img)
cv2.imwrite(const_test_image_4_output, detected_cars_img)



# In[133]:


# Pipeline as executed on Test Image #5 

reset_history()
detected_cars_img = pipeline(const_test_image_5, True)
plt.imshow(detected_cars_img)
cv2.imwrite(const_test_image_5_output, detected_cars_img)



# In[134]:


# Pipeline as executed on Test Image #6 

reset_history()
detected_cars_img = pipeline(const_test_image_6, True)
plt.imshow(detected_cars_img)
cv2.imwrite(const_test_image_6_output, detected_cars_img)



# In[135]:


# Pipeline as executed on Test Image #7 

reset_history()
detected_cars_img = pipeline(const_test_straight1, True)
plt.imshow(detected_cars_img)
cv2.imwrite(const_test_straight1_output, detected_cars_img)



# In[136]:


# Pipeline as executed on Test Image #8 

reset_history()
detected_cars_img = pipeline(const_test_straight2, True)
plt.imshow(detected_cars_img)
cv2.imwrite(const_test_straight2_output, detected_cars_img)




# In[ ]:


# Pipeline as executed on the Test Video

reset_history()

clip1 = VideoFileClip(const_test_video)
output_clip1 = clip1.fl_image(pipeline)
output1 = const_test_video_output
get_ipython().magic('time output_clip1.write_videofile(output1, audio=False)')


# In[62]:


# Pipeline as executed on the Project Video

reset_history()

clip2 = VideoFileClip(const_project_video)
output_clip2 = clip2.fl_image(pipeline)
output2 = const_project_video_output
get_ipython().magic('time output_clip2.write_videofile(output2, audio=False)')


# In[ ]:



