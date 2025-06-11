# Sketch2Data: Recovering Data from Hand-Drawn Infographics (Project page)



![Python](https://img.shields.io/badge/Python->=3.8-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.12.0-Red?logo=pytorch)

This repository provides the implementation for our sketch2data, a method to recover data values from glyph-based hand-drawn infographics.

<img src="./teaser.png" slign="center">





 We use *Thoughts* design as an example.
<img src="./thoughts.png" slign="center">


## Data preparation: procedurally generate glyphs

~~~bash
cd glyph_design
~~~

 you may install the following packages (and their dependencies) manually (Due to the use of cairosvg, the code runs on MAC OS.)
:

- numpy, glob, xml, cairosvg
- PIL, shapely, random, shutil, svgutils



Use the following command to generate detection dataset. The generated images are at `../glyph_detection/thoughts_detection_dataset`

~~~bash
python generate_detection_glyphs.py
~~~

And the following command to generate recognition dataset. The generated images are at `../glyph_rec/thoughts_rec_dataset`
~~~bash
python generate_rec_glyphs.py
~~~
And the following command to generate individual glyphs without deformation. The generated images are at `../glyph_rec/thoughts_rec_dataset/default_glpyh`
~~~bash
python generate_default_glyphs.py
~~~


## Glyph detection

 The code is tested on window 10. You may install the following packages (and their dependencies) 
:
- pathlib, tqdm, yaml, ultralytics, PIL

~~~bash
cd glyph_detection
~~~
Split the dataset and prepare the dataset into format required by ultralytics. 
~~~bash
python prepare_dataset.py
~~~

### Train model

Please download the yolov8n.pt from https://docs.ultralytics.com/models/yolov8/#overview, and put it at `pretrained_yolov8n_model`. Run the following command to train the yolov8 detection model:

~~~bash
python train_thoughts.py 
~~~

### Test model and visualize the detected glyphs. 
The detected glyphs/files are saved as `./thoughts_detection_results`

~~~bash
python test_thougths.py
~~~


## Glyph recognition
 The code is tested on window 10. You may install the following packages (and their dependencies) 
:
- PyTorch>=1.12.0, torchvision>=0.13.0
- Pandas, PIL, numpy, scipy, re, typing


~~~bash
cd glyph_rec
~~~
Split the dataset and generate csv files for 10 different settings.
~~~bash
python prepare_dataset.py
~~~

### Train model

Run an ensemble of neural networks (10), each trained on a different subset of the dataset.
~~~bash
python train_test_thoughts.py 
~~~

### Evaluate the results and visualize the reconstructed infographic. 
The reconstructed design is saved at `./thoughts_reconstruction.png`
~~~bash
python evaluate.py
~~~

## Acknowledgments
The code in glyph_design/svgpathtools was slightly modified from https://pypi.org/project/svgpathtools/.

## Citation
If you use this code for your research, please cite our paper:

~~~bibtex
@article{QI2025104251,
title = {Sketch2Data: Recovering data from hand-drawn infographics},
journal = {Computers & Graphics},
pages = {104251},
year = {2025},
issn = {0097-8493},
doi = {https://doi.org/10.1016/j.cag.2025.104251},
url = {https://www.sciencedirect.com/science/article/pii/S0097849325000925},
author = {Anran Qi and Theophanis Tsandilas and Ariel Shamir and Adrien Bousseau}}
~~~
