# vcs-project
Project for Vision and Cognitive Services course @ UniPD

## Installation

Install all the dependencies using `pip install -r requirements.txt`

## Usage

`python3 main.py [action] {option}`
With:
`action`:
    - `acquire-mediapipe {gesture}`: acquire `gesture` from webcam (Kinect but RGB) for mediapipe recognition + sklearn training
    - `acquire-dataset`: automatically acquire data from *APE dataset*
    - `acquire-kinect {gesture}`: acquire `gesture` from Kinect (RGBD) for NiTE2 recognition + sklearn training
    - `aseqtrain`: acquire every gesture from webcam (Kinect but RGB) sequentially (debugging purposes)
    - `train {mode}`: train sklearn in one of two `modes` [`mediapipe` or `nite`]
    - `run {mode}`: run program in one of two `modes` [`mediapipe` or `nite`]