# Smart Assistant Interaction through Visual Gesture Recognition using a Kinect Sensor

Project for Vision and Cognitive Services course @ UniPD
Master's Course in Computer Science, A.Y. 2020-2021.

## Disclaimer

This repo is distributed under GPLv3 license, and this means that you can use and modify the code as you prefer; nevertheless, we don't assume any kind of responsibility on the use you will do of it, you're responsible for yourself and yourself only.

## Abstract

In the last decade, the raise of smart assistant devices played a key role for the everyday life as they help to retrieve information from the web, manage other home devices and execute routines independently. 
Even though the vocal interaction is getting more and more efficient, not always the commands are interpreted correctly.
Moreover, looking at deaf people and/or people with speech disabilities, the effective interaction is drastically reduced, thus limiting the use of this kind of devices.
The main goal of this project is to use the visible body parts of a person to interact with smart devices like Google Home through human pose estimation and gesture recognition.
During our work we compared two different solutions – Google MediaPipe and NiTE2 middleware for Kinect –, and for both of these use-cases we tried different classification algorithms: multinomial logistic regression, ridge regression, random forest, support vector machines and multilayer perceptron.
Through intensive trial and error experiments we were able to get F1-scores in a range between 0.71 and 0.97 for the different use-cases and classification algorithms, with an average accuracy never below 0.70.

## Installation

Install all the dependencies using `pip install -r requirements.txt`

## Usage

`python3 main.py [action] {option1} {option2}`

### Acquire

- `acquire-mediapipe {gesture}`: acquire `gesture` from webcam (Kinect but RGB) for mediapipe recognition + sklearn training
- `acquire-dataset`: automatically acquire data from *APE dataset*
- `acquire-kinect {gesture}`: acquire `gesture` from Kinect (RGBD) for NiTE2 recognition + sklearn training
- `aseqtrain`: acquire every gesture from webcam (Kinect but RGB) sequentially (debugging purposes)

### Train

- `train {mode}`: train sklearn in one of two `modes`:
    -   `mediapipe`
    -   `nite`

### Run

- `run {mode} {classification}`: run program in one of two `modes`:
    -   `mediapipe`
    -   `nite`

    #### With different `classification` algorithm:
    - `lr`: Logistic Regression
    - `rc`: Ridge
    - `rf`: Random Forest
    - `gb`: Gradient Boosting
