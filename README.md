# mhealth
## A federated learning sandbox  (To be finished)
Converted activity recognition dataset ["MHealth"](https://www.kaggle.com/datasets/gaurav2022/mobile-health) to a federated learning (FL) dataset for FL simulation.  Currently, main.py does not contain the federated learning routing.  The learning and evaluation routine only works in centralized (non-federated) mode.

Performance was tested in centralized mode (non-federated) better than the current best solution based on LSTM posted on Kaggle (last time I checked in 2021).  This was meant to be a quick sandbox from many years ago so I did not get to posting it back on Kaggle.  My apologies.

What was done to the dataset:
* Made dataset federated in non-IID or IID mode
* Pre-process data with overlapping windows.  

The design decisions made were to be quick, rather unoptimized (to be done later during FL design stage), and still to be able to do well using a very simple model (MLPClassifier with 3 layers), < 15k parameters.  This showed to be very performant in centralized mode, which was a necessary evaluation step for federated learning.  The reason is that we assume wireless bandwidth is the most expensive resouces in this to-be-simulated device-based test.  Model / memory usage for this model and pre-processing method is also moderate and arguably practical.  