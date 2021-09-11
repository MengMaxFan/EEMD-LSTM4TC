# EEMD-LSTM4TC
This repository is the official implementation of "Predicting Tropical Cyclone Wave Height Using Buoy Data".

## Model Structure
![model](https://user-images.githubusercontent.com/17058892/132942282-1e65e599-23e1-4ba8-b171-1c7666b3dd90.png)

## Requirements
  Keras, PyEMD
## Training
  To run the expriments in this paper,run this commend:
  ```
  python mian.py
  ```
  
These files are the main experiments of the paper. If you want to test the performance of a particular TC in generating wave height, specify the test set as the particular TC and set the remainder of the sequence as the training set.

Note that the data is interpolated and sampled, and the original data can be downloaded from the ["China Scientific Data"](http://csdata.org) website. Dataset DOI：
10.11922/sciencedb.924 .
If you want to use the real-time data for training and prediction in practice, you only need to replace the files under the csv file with your own data, but you need to pre-process the data to a 30-minute interval.

