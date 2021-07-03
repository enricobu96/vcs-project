# vcs-project
Project for Vision and Cognitive Services course @ UniPD

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
