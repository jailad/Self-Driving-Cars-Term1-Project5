# Writeup for Self-Driving-Cars-Term1-Project5 - Vehicle Detection and Tracking

---

[//]: # (Image and Video References)
[input_image1]: ./project/project_data/non-vehicles/GTI/image46.png
[input_image2]: ./project/project_data/non-vehicles/GTI/image141.png
[input_image3]: ./project/project_data/vehicles/GTI_Left/image0060.png
[input_image4]: ./project/project_data/vehicles/GTI_Left/image0133.png

[pipeline_py]: ./project/project.py
[pipeline_notebook]: ./project/project.ipynb
[pipeline_html]: ./project/project.html
[pipeline_used_for_training]: ./project/svm_train.py
[older_pipeline_notebook]: ./project/project_old.ipynb

[output_image1]: ./project/output_images/sliding_window.png
[output_image2]: ./project/output_images/heatmap_thresholding.png
[output_image3]: ./project/output_images/bounded_boxes.png
[output_image4]: ./project/output_images/sliding_window2.png
[output_image5]: ./project/output_images/detections.png

* A submission by Jai Lad

# Table of contents

1. [Objective(s)](#objective)
2. [Project Rubric](#rubric)
3. [Key File(s)](#keyfiles)
4. [Detection Pipeline](#pl)
5. [Video Implementation](#vi)
6. [Challenges and Comments](#chal)

<BR><BR>
---

<BR><BR>
---

## Objective(s) <a name="objective"></a> :

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

<BR><BR>
---

## Project Rubric <a name="rubric"></a> :

The project rubric is available below:

* [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

<BR><BR>
---

## Key File(s) <a name="keyfiles"></a> :

* [readme.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/README.md) - The accompanying Readme file, with setup details.
* [project.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.ipynb) - [Jupyter](http://jupyter.org/) notebook used to implement the pipeline.
* [project.py](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.py) - Python script version of the above notebook. ( useful for referencing specific file numbers )
* [project.html](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.py) - HTML version of the above notebook. 
* [svm_train.py](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/svm_train.py) - Python script used to train the SVM classifier ( useful for referencing specific file numbers )
* [project_old.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.ipynb) - [Jupyter](http://jupyter.org/) notebook used to implement the initial pipeline. It was used to conduct experiments for various parameters and color spaces.
* [writeup.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/writeup.md) - Report writeup file
* [output_images folder](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/tree/master/project/output_images) - Folder with various images as generated from the pipeline.
* [test_video](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/test_video/output/test_video_output.mp4) - Video of the output from the pipeline for the project video. Alternative Youtube link is [here](https://youtu.be/1VDsRGCD_1g).
* [pipeline_video](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project_video/output/project_video_output.mp4) - Video of the output from the pipeline for the project video. Alternative Youtube link is [here](https://youtu.be/ik_of0XFF1g).
* [trained models](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/tree/master/project/model) - Folder with a few trained Models which were generated from the training pipeline


<BR><BR>
---

## Detection Pipeline <a name="pl"></a> :

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained [here](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.py#L363-L414).  

I started by reading in all the `vehicle` and `non-vehicle` images.  

Example(s) of the `non-vehicle` class:

![alt text][input_image1] 
![alt text][input_image2] 

Example(s) of the `vehicle` class:

![alt text][input_image3] 
![alt text][input_image4]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and this was a manual process. The results are summarized below. The term FP below means False Positive. These experiments were performed in [project_old.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.ipynb) and are summarized as the last cell of the above notebook.

* I tweaked one parameter at a time, recorded the results and see which ones gives the best results. 
    * Samples - Increased number of samples to 8000
    * Color Space :
        * All else being equal, HSV performed poorly relative to RGB in terms of false positives.
        * LUV was better than HSV but poorer than RGB.
        * HLS was better than HSV but poorer than RGB and LUV.
        * YUV comparable to RGB
        * YCrCb was slightly worse than RGB.
        * Based on the above results, I decided to proceed with RGB for the time - being. Alternative(s) could be YUV.
    * HoG orientations :
        * Started off with 9 - three false positives - FP.
        * 10 - eight FP.
        * 11 - two FP.
        * 13 - five FP.
        * 8 - four FP.
        * 7 - two FP.
        * 6 - eight FP.
        * Based on the above, I decided to proceed with 7 HoG orientations ( two FP ). Alternatives could be 11, or 9.
    * Pixel Per Cell :
        * Baseline - 8 - 2 FP.
        * 16 - 4 FP.
        * 32 - 6 FP.
        * 6 - 2FP.
        * 5 - 3 FP.
        * 4 - 5 FP.
        * Decision: 1. Go with 8 2. Backup(s) - 6.
    * Cells per Block :
        * Baseline - 2 cells per block - 2 FP.
        * 4 - 3 FP.
        * 8 - 3 FP.
        * 1 - 2 FP.
        * Decision : 1 cell per block. 2. Backup(s) - 2.
    * HoG Channel :
         * Baseline - 0 - 2 FP.
         * 1 - 11 FP.
         * 2 - 11 FP.
         * ALL - 4 FP.
         * Decision - 0 2. Backup(s) - All.
    * Spatial Size and hist_bins :
         * 16 x 16 and 16 - 1 FP.
         * 32 x 32 and 16 - 5 FP.
         * 8 x 8 and 8 - 1 FP.
         * 4 x 4 and 4 - 5 FP.
         * Decision - 16 x 16 and 16 2. Backup(s) - 8 x 8 and 8.
    * Only Spatial Features On:
        * 6 FP.
    * Only Histogram Features On:
        * 20 FP.
    * Only HoG Features On:
        * 4 FP. Also positive detection was not that strong.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the following parameters:

* color_space = 'RGB'
* orient = 9  # HOG orientations
* pix_per_cell = 8 # HOG pixels per cell
* cell_per_block = 2 # HOG cells per block
* hog_channel = "ALL"
* spatial_size = (32, 32) # Spatial binning dimensions
* hist_bins = 32    # Number of histogram bins
* spatial_feat = True # Spatial features on or off
* hist_feat = True # Histogram features on or off
* hog_feat = True # HOG features on or off

The training was performed in [this](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/svm_train.py) file, and the resultant model(s) were saved [here](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/tree/master/project/model).

### Sliding Window Search

I implemented the sliding window search [here](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.py#L649-L718).

Ultimately I searched on one scale using RGB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

![alt text][output_image1]

Here is an example image of the bounding boxes.

![alt text][output_image5]

<BR><BR>
---

## Video Implementation <a name="vi"></a> :

* Here is a link to the output from my pipeline on the [test video](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/test_video/output/test_video_output.mp4). Alternative Youtube link is [here](https://youtu.be/1VDsRGCD_1g).

* Here is a link to the output from my pipeline on the [project video](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project_video/output/project_video_output.mp4) . Alternative Youtube link is [here](https://www.youtube.com/watch?v=SnXz4bTzu1c).


* I implemented the false positive rejection [here](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.py#L913-L1002). I essentially tracked the detection centroids of the past five frames, and then rejected the centroids which were not within a threshold range of the centroids detected for the past few frames.

* I performed duplicate detection rejection [here](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.py#L1036-L1053). At the scale and window size at which I was searching, the detection filter was very sensitive, and would yield in many many good detections for genuine cars. At the same time, I did not want to hard-code a heat threshold since it could vary each frame. To make it dynamic, I detected the maximum heat intensity of a frame, and then applied a fraction to it [here](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.py#L1045).


<BR><BR>
---

## Challenges and Comments <a name="chal"></a>:

* The process of finding optimal detection parameters was very manual. I would like to find a better way to automate this process. One good aspect of this process was that it really drives home the point of feature engineering capabilities of neural network(s), as I reflect on the process of crafting the 8000 column+ feature vector for this project.

* Jupyter environment was running out of memory easily when I tried to train the model with the entire set of available features ( length of a single feature > 8000 ), and when I tried to use all available samples. Therefore, for the longest time, I struggled with training the pipeline with the complete set of features and samples. This was resolved by migrating the training code to a Python pipeline. I have since made this my default workflow, as in, initial EDA ( exploratory data analysis ) and code validation in Jupyter for a small data set, and subsequent migration to a Python notebook when running the code on large scale data.

* SVM model(s) train much faster than Neural Network(s) !

* SVM model(s) also have a smaller on-Disk footprint relative to Neural Networks. The [model](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/model/persist_data_rgb2.p) for this project stands at 266 KB.

* To handle the potential difference in terms of pixel strengths ( e.g. between o - 1 versus 0 - 255), I captured the min max range(s) of pixel strengths [dynamically](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.py#L163-L178).

* The detection capacity of the pipeline at far off distances is not good, presumably because of the underlying data ( or the scale at which I am searching).


