# EGAE
Automation of LIME using ensemble-based genetic algorithm explainer: A case study on melanoma detection dataset

Due to space limitation in github the data needed to run EGAE including the saved weights for the model, training and test sets and their labels (in the numpy.ndarray format), and the ground truths delineated by experts for accuracy calculation of 8 sample images in the paper are [here](https://drive.google.com/drive/folders/1341NsT56HIh4DyB6R0ViuxPtDD1AWfdg?usp=sharing).

The 8 sample images are selected from 600 images in the X1_test for evaluation. The record number of the selected sample images according to their location in X1_test and appearance in Table 3 of the paper are given based on the following pattern:
(image number in Table 3 of the paper = image number in the X1_test = the ground truth file). As such,
image 1 = 100 = i100.npy,
image 2 = 277 = i277.npy,
image 3 = 74 = i74.npy,
image 4 = 5 = i5.npy,
image 5 = 574 = i574.npy,
image 6 = 91 = i91.npy,
image 7 = 208 = i208.npy,
image 8 = 11 = i11.npy,

INSTRUCTION:

1- Open EGAE.py.
2- Load the training and test data and all the ground truths images with the prediction model saved weights using the the variable explorer panel.
3- Run lines 1-86 to train the model with weights and calculation the yhat.
4- In line 91 specify which image you want to explain using the image number in X1_test. For example, assign variable "Figure" to 100 if you want to test image 1 in Table 3 of the paper.
5- Run from line 88-595 to let EGAE calculate the results.
6- Run  lines 602-668 to generate consensus voting image.
7- Run lines 673-685 to generate majority voting image.

##########ACCURACY CALCULATION###########
1- Save the consensus voting image (c1) and majority voting image (m1) in lines 693-694.
2- Change the "initial" and "second" variables in lines 716 and 717 accordingly. For example, if you want to calculate accuracy for consensus voting the values of "initial" and "second" should be i100.flatten() and c1.flatten()
3- Run lines 720-726 to calculate the Euclidean distance. The less this distance is, the more accurate the explanation is.
You can also run LIME separately (lines 770-779) with a predefined number of superpixels in the input image and top features (num_features parameter) to be seen. The num_samples parameter also needs to be determined in advance by user. Then, you can also calculate the accuracy of LIME explanation with the ground truth. Number of superpixels in the input image should be specified manually in segmentation_fn.
########## PERFORMANCE GRAPH#############
1- Run lines 734-743
########## SPARSITY PLOTS################
1- Run lines 752-756

