## RDE automation

## Overview
The tool predicts the quality of experiments from electrode images in rotating disk electrode methods. It contributes to the future automation of electrochemical experiments.

## Description
Rotating disk electrode (RDE) technique is the essential tool to study the activity, stability, and other fundamental properties of electrocatalysts. High-quality RDE experimentation demands to evenly coat the catalyst layer on the electrode surface, which strongly relies on experience and lacks necessary quality control currently. The inadequate evaluation to ensure the quality of RDE experimentation aside from conventional expertise-relied judgment reduces the efficiency, challenges data interpretation, and hinders future automation of RDE experimentation. Here we propose a simple, easy-to-execute and non-destructive method through a combined microscopy photographing and artificial intelligence based decision-making process to assess the quality of as-prepared electrode. We develop a convolutional neural network-based method that uses the microscopic photos of as-prepared electrodes to directly evaluate the sample quality. In the study of electrodes for oxygen reduction reaction, the model achieved an accuracy of over 80% in predicting sample qualities. Our method enables the removal of low-quality samples prior to the actual RDE test, thereby ensuring high-quality electrochemical experimentation and offering the path towards high-quality automated electrochemical experimentation. This approach is applicable to various electrochemical systems and highlights the great applicability of artificial intelligence in automated experimentation.

## Usage
#1. To take a micro-scope image of a catalyst coated rotating electrode.(Please position the electrode at the center of the image and capture it from directly above, ensuring that the entire electrode is visible. Be careful to avoid overexposure caused by light reflections.)
#2. To do a RDE experiment and calculate the kouteckey-levich slope from the liner sweep voltammogram with various rotating speeds.
#3. To save both the micro-scope image in the foleder(Image) and the slope value in csv file(ORR.csv) acordingly.
#4. To run the code.

## Install
Python version: 3.8.5 
OpenCV version: 4.10.0
TensorFlow version: 2.13.0
Keras version: 2.13.1
NumPy version: 1.22.0
scikit-learn version: 1.3.2
Pillow (PIL) version: 10.4.0
Matplotlib version: 3.7.5
SciPy version: 1.10.1


