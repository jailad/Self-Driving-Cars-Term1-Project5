# Read me for Self-Driving-Cars-Term1-Project5 - Vehicle Detection and Tracking

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

* [Objective(s)](#objective)
* [Key File(s)](#keyfiles)
* [Challenges and Comments](#chal)

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

## Key File(s) <a name="keyfiles"></a> :

* [writeup.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/writeup.md) - Report writeup file.
* [output_images folder](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/tree/master/project/output_images) - Folder with various images as generated from the pipeline.
* [test_video](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/test_video/output/test_video_output.mp4) - Video of the output from the pipeline for the project video. Alternative Youtube link is [here](https://youtu.be/1VDsRGCD_1g).
* [pipeline_video](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project_video/output/project_video_output.mp4) - Video of the output from the pipeline for the project video. Alternative Youtube link is [here](https://youtu.be/ik_of0XFF1g).
* [project.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.ipynb) - [Jupyter](http://jupyter.org/) notebook used to implement the pipeline.
* [trained models](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/tree/master/project/model) - Folder with a few trained Models which were generated from the training pipeline


<BR><BR>
---

## Challenges and Comments <a name="chal"></a>:

* The process of finding optimal detection parameters was very manual. I would like to find a better way to automate this process. One good aspect of this process was that it really drives home the point of feature engineering capabilities of neural network(s), as I reflect on the process of crafting the 8000 column+ feature vector for this project.

* Jupyter environment was running out of memory easily when I tried to train the model with the entire set of available features ( length of a single feature > 8000 ), and when I tried to use all available samples. Therefore, for the longest time, I struggled with training the pipeline with the complete set of features and samples. This was resolved by migrating the training code to a Python pipeline. I have since made this my default workflow, as in, initial EDA ( exploratory data analysis ) and code validation in Jupyter for a small data set, and subsequent migration to a Python notebook when running the code on large scale data.

* SVM model(s) train much faster than Neural Network(s) !

* SVM model(s) also have a smaller on-Disk footprint relative to Neural Networks. The [model](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/model/persist_data_rgb2.p) for this project stands at 266 KB.

* To handle the potential difference in terms of pixel strengths ( e.g. between o - 1 versus 0 - 255), I captured the min max range(s) of pixel strengths [dynamically](https://github.com/jailad/Self-Driving-Cars-Term1-Project5/blob/master/project/project.py#L163-L178).

* The detection capacity of the pipeline at far off distances is not good, presumably because of the underlying data ( or the scale at which I am searching).


