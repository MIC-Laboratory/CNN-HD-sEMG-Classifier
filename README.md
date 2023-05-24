# CNN-HD-sEMG-Classifier
This is a project is about to use MobilenetV2 to classifify HD-sEMG. The project test on CSL-HD sEMG dataset and ICE Lab HD sEMG dataset
## Dependency
- python=3.9
- torchvision
- pytorch-cuda=11.8
- pytorch
- torchaudio
- matplotlib
- scikit-learn
- tqdm
### Quick dependency installation
```
conda env create -f environment.yml
```
## Dataset
### CSL HD-sEMG Dataset
#### Introduction
- Cognitive Systems Lab (CSL) investigates the usefulness of inertial sensing (IMU) and electromyography (EMG) for the detection of gestures 
- Inertial sensors detect movement through acceleration and yaw rate sensors
- The group’s site provides two sets of data
    - mmGest - IMU and EMG data from 5 different subjects in 5 separate sessions
    - csl-hdemg - high-density EMG recordings of finger movements
#### Overview of Data
- EMG data for 5 subjects
- Each subject performed 5 sessions recorded on different days
27 gestures in each gesture set, 10 trials for each gesture per session
One of those gestures is an “idle” gesture, which was repeated for 30 trials
- Sampling Rate - 2048 Hz
- Data saved in a 192 x N matrix
- Every row is a channel
#### Electrodes
- Every 8th channel does not contain meaningful data due to differential amplification in the bipolar recordings and should be ignored
1st channel is the differential signal of electrodes 1 and 2, 2nd channel is the differential channel of electrodes 2 and 3, etc ...
- Bipolar electrode arrangement with differential amplifier
- Suppresses signals common to two electrodes
- Essentially, differential amplification subtracts the potential voltage at one electrode with the potential voltage at another and then amplifies the difference
#### Electrodes Placement
- Electrodes 1, 9, 17, ..., 185 are located near the proximal end
- Electrodes 8, 16, ..., 192 are located on the distal end

### ICE Lab Dataset



## Quick Start
The config file controls all the hyper parameters and data path of the training program, feel free to modify it. 
```
python train.py
```