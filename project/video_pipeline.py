
####################################################################################

# Parameters
# Max Heat
# Heat threshold
# Depth of History
# Eject False Positives from List
# Number of past frames to consider
# Distance from previous centroids

####################################################################################
# Imports

import numpy as np
import cv2
import glob # Used to read in image files of a particular pattern
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

####################################################################################

# Shared Constants
# Constant for separating log statements( if needed )
const_separator_line = "--------------------------------"

# Constant representing a random seed for Data Splitting
const_data_split_seed = 42

# Constant representing location of the trained model
# The model name(s) are unique as per the color spaces they were trained to operate upon
const_persist_data_rgb3 = './model/persist_data_rgb3.p'
const_persist_data_rgb2 = './model/persist_data_rgb2.p'
const_persist_data_rgb = './model/persist_data_rgb.p'

# Constants representing paths of test images

const_test_images_output_folder = "./test_images/output/"
const_project_video_output = './project_video/output/'
const_test_video_output = './test_video/output/'

const_extension_jpg = ".jpg"
const_extension_mp4 = ".mp4"

const_test_straight1_prefix = "straight_lines1"
const_test_straight1 = './test_images/input/straight_lines1.jpg'
const_test_straight1_output = './test_images/output/straight_lines1.jpg'

const_test_straight2_prefix = 'straight_lines2'
const_test_straight2 = './test_images/input/straight_lines2.jpg'
const_test_straight2_output = './test_images/output/straight_lines2.jpg'

const_test_image_1_prefix = 'test1'
const_test_image_1 = './test_images/input/test1.jpg'
const_test_image_1_output = './test_images/output/test1.jpg'

const_test_image_2_prefix = 'test2'
const_test_image_2 = './test_images/input/test2.jpg'
const_test_image_2_output = './test_images/output/test2.jpg'

const_test_image_3_prefix = 'test3'
const_test_image_3 = './test_images/input/test3.jpg'
const_test_image_3_output = './test_images/output/test3.jpg'

const_test_image_4_prefix = 'test4'
const_test_image_4 = './test_images/input/test4.jpg'
const_test_image_4_output = './test_images/output/test4.jpg'

const_test_image_5_prefix = 'test5'
const_test_image_5 = './test_images/input/test5.jpg'
const_test_image_5_output = './test_images/output/test5.jpg'

const_test_image_6_prefix = 'test6'
const_test_image_6 = './test_images/input/test6.jpg'
const_test_image_6_output = './test_images/output/test6.jpg'

# Constants representing paths of a project video
const_project_video_prefix = 'project_video'
const_project_video = './project_video/input/project_video.mp4'
const_project_video_output = './project_video/output/project_video_output.mp4'

const_project_video_short = './project_video/input/project_video_short.mp4'
const_project_video_short_output = './project_video/output/project_video_short_output.mp4'

const_project_video_short2 = './project_video/input/project_video_short2.mp4'
const_project_video_short2_output = './project_video/output/project_video_short2_output.mp4'

const_project_video_short3 = './project_video/input/project_video_short3.mp4'
const_project_video_short3_output = './project_video/output/project_video_short3_output.mp4'

const_project_video_short4 = './project_video/input/project_video_short4.mp4'
const_project_video_short4_output = './project_video/output/project_video_short4_output.mp4'

# Constants representing paths of a test video
const_test_video_prefix = 'test_video'
const_test_video = './test_video/input/test_video.mp4'
const_test_video_output = './test_video/output/test_video_output.mp4'

const_max_history_frames = 15 # Constant representing maximum number of historical frames to consider
const_detection_threshold_x = 30 # Threshold for the difference between x co ordinate of centroid of current frame to historical frames
const_detection_threshold_y = 15 # Threshold for the difference between y co ordinate of centroid of current frame to historical frames

const_measurements_fontsize = 1
const_measurements_fontcolor = (255,255,255)
const_measurements_fontcolor_true = (0,0,255)
const_measurements_fontcolor_false = (255,0,0)

global_centroid_history_list = list() # Each entry here contains a list of car centroids as detected per frame
global_current_frame_number = 0

####################################################################################

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

####################################################################################

def getFileName(folder, fileprefix, filepostfix):
    epoch_time_string = str(time.time())
    filename = folder + fileprefix + "_" + epoch_time_string + filepostfix
    return filename

####################################################################################

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

####################################################################################

# Restore the persisted parameters from Disk and run the pipeline on the same test image as above to ensure consistent results

persist_data_dict = pickle.load( open(const_persist_data_rgb3, "rb" ) )
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

####################################################################################

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

####################################################################################

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

####################################################################################


# Convenient method to add a frame number to an image
# This can be useful to track specific frame(s) of a video for debugging
def add_frame_number_to_image(param_img, current_frame_number, true_detections_list, false_detections_list, max_heat, dynamic_threshold):
    return_img = param_img.copy()
    
    true_detections_count = 0
    false_detections_count = 0
    
    if true_detections_list != None:
        true_detections_count = len(true_detections_list)
        
    if false_detections_list != None:
        false_detections_count = len(false_detections_list)

    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(return_img, 'Frame Number = %d' % current_frame_number, (50, 50), font, const_measurements_fontsize, const_measurements_fontcolor, 2)
    
    cv2.putText(return_img, 'True Detections Count = %d' % true_detections_count, (50, 75), font, const_measurements_fontsize, const_measurements_fontcolor_true, 2)
    
    cv2.putText(return_img, 'False Detections Count = %d' % false_detections_count, (50, 100), font, const_measurements_fontsize, const_measurements_fontcolor_false, 2)

    cv2.putText(return_img, 'Maximum Heat = %d' % max_heat, (50, 125), font, const_measurements_fontsize, const_measurements_fontcolor, 2)

    cv2.putText(return_img, 'Dynamic Threshold = %d' % dynamic_threshold, (50, 150), font, const_measurements_fontsize, const_measurements_fontcolor, 2)

    return return_img

####################################################################################

# Clear history across the video pipeline

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

####################################################################################

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
    print(" Max of proximity list for this frame : " + str(max(proximity_list_for_centroids)))
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
        if proximity == max_proximity_value and max_proximity_value > 3:
            indices_list_true_detections.append(index)
        else: 
            indices_list_false_detections.append(index)
    return indices_list_true_detections, indices_list_false_detections

####################################################################################

ystart = 400
ystop = 656
scale = 1
thresholding_ratio = 0.1

####################################################################################

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
    heat_thresholded = apply_threshold(heat, dynamic_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat_thresholded, 0, 255)

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
    draw_img_true_and_false = add_frame_number_to_image(draw_img_true_and_false, global_current_frame_number, true_detections_indices_list, false_detections_indices_list, max_heat, dynamic_threshold)
    
    update_history(current_frame_centroids_list)

    # update_history(true_detections_indices_list)
    
    debugLog("-----------------------------------------------------------------------------------------------")

    return draw_img_true_and_false

####################################################################################

# Pipeline as executed on a test image

# debugLog("Pipeline as executed on a test image")

# reset_history()
# detected_cars_img = pipeline(const_test_image_1, True)
# plt.imshow(detected_cars_img)
# filename = getFileName(const_test_images_output_folder,const_test_image_1_prefix,const_extension_jpg)
# print(filename)
# cv2.imwrite(filename, detected_cars_img)
# plt.show()

# ####################################################################################

# # Pipeline as executed on the test video

# debugLog("Pipeline as executed on the test video")

# reset_history()

# clip1 = VideoFileClip(const_test_video)
# output_clip1 = clip1.fl_image(pipeline)
# output1 = const_test_video_output
# output_clip1.write_videofile(output1, audio=False)

# # ####################################################################################

# # # Pipeline as executed on the short project video

# debugLog("Pipeline as executed on the short project video")

# reset_history()

# clip2 = VideoFileClip(const_project_video_short)
# output_clip2 = clip2.fl_image(pipeline)
# output2 = const_project_video_short_output
# output_clip2.write_videofile(output2, audio=False)

# # ####################################################################################

# # # Pipeline as executed on the short project video #2

# debugLog("Pipeline as executed on the short project video #2")

# reset_history()

# clip3 = VideoFileClip(const_project_video_short2)
# output_clip3 = clip3.fl_image(pipeline)
# output3 = const_project_video_short2_output
# output_clip3.write_videofile(output3, audio=False)


# ####################################################################################

# # Pipeline as executed on the short project video #3

# debugLog("Pipeline as executed on the short project video #3")

# reset_history()

# clip4 = VideoFileClip(const_project_video_short3)
# output_clip4 = clip4.fl_image(pipeline)
# output4 = const_project_video_short3_output
# output_clip4.write_videofile(output4, audio=False)

####################################################################################

# Pipeline as executed on the short project video #4

# debugLog("Pipeline as executed on the short project video #4")

# reset_history()

# clip5 = VideoFileClip(const_project_video_short4)
# output_clip5 = clip5.fl_image(pipeline)
# output5 = const_project_video_short4_output
# output_clip5.write_videofile(output5, audio=False)

####################################################################################

# Pipeline as executed on the short project video #4

debugLog("Pipeline as executed on the project video")

reset_history()

clip6 = VideoFileClip(const_project_video)
output_clip6 = clip6.fl_image(pipeline)
output6 = const_project_video_output
output_clip6.write_videofile(output6, audio=False)

####################################################################################
