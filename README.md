# Black-Box Attacks on Neural Networks

## Abstract
The paper[1] discusses an algorithm which allows us to craft an adversarial attack on black box networks (for classification). The attacker has no knowledge of internals or training data of the victim.
<br><br>
The solution presented treats the black box as an oracle and gets the output for several inputs and trains a substitute model on this data. Then adversarial samples are created by a white box attack on this substituted model. These adversarial samples work well to attack on the black box.
<br><br>
In this project, by the time of midterm review, we implemented this algorithm on MNIST dataset. Now, we have tried to implement this on object detection on COCO dataset.
We separate the bounding boxes of object detected as new images, create adversarial example from them, and stitch the adversarial examples into the original image. 

## Requirements
```
python >=3.5 -> python == 3.10
numpy
torch
torchvision
matplotlib
tensorflow -> delete
tensorboard -> delete
terminaltables -> delete
pillow
tqdm
libtiff -> changed to tifffile
```

## Setup Instructions
1. Install the requirements
2. Clone This directory
```
git clone https://github.com/darth-c0d3r/black_box_attacks
```
3. Get [COCO dataset](http://images.cocodataset.org/zips/val2017.zip)
#### For attack on object detection
* Create substitute model
```
python3 main_script.py --yolo
```
It asks to save the model. Give it a proper name. The model is saved in the folder ```saved_models/```
* Create adversarial samples
```
python3 main_script.py --adv
```
It asks for which substitute model to use, num_samples to be generated. This generates the adversarial samples and stores them in directory ```adv_samples/```
* Stitch the adversarial examples generated to original image
```
python3 main_script.py --stitch
```
You can see the images in ```stitched_images/``` directory
* Test them with black box model
```
python3 main_script.py --yolotest
```
#### For attack on MNIST dataset
```
cd black_box_attack_classification
```
* Create black box model
```
python3 main_script.py --bb
```
It asks to save the model. Give it a proper name. The mode lis saved in the folder ```saved_models/```
* Create substitute model
```
python3 main_script.py --sub
```
It asks which black box model to use. Give it the model name from ```saved_models/```. 
It asks to save the model. Give it a proper name. The mode lis saved in the folder ```saved_models/```
* Create adversarial samples
```
python3 main_script.py --adv
```
It asks for which substitute model to use, num_samples to be generated. This generates the adversarial samples and stores them in directory ```adv_samples/```
* Test them with black box model
```
python3 main_script.py --test
```

## Results


## Additional Details
#### * [dataset.py](dataset.py)
This file creates different datasets.
#### * [main_script.py](main_script.py)
This file contains various options to run different steps in the project
#### * [oracle.py](oracle.py)
This file gets output from the oracle for a given input
#### * [predict.py](predict.py)
This gets predictions from the black box for inputs given
#### * [model.py](model.py)
This creates model
#### * [stitch_photos.py](stitch_photos)
This contains code to stitch the adversarial sample generated to original photo
#### * [train_substitute.py](train_substitute.py):
This file implements the Substitute DNN training algorithm given in paper[1]. <br>
For oracle *Õ*, a maximum number *max<sub>ρ</sub>* of substitute training epochs, a substitute architecture *F* and initial training set *S<sub>0</sub>*.
<br>
**Input**: *Õ*, *max<sub>ρ</sub>* , *S<sub>0</sub>* , *λ*
<br>
1:	Define architecture F
<br>
2:	**for** ρ ∈ 0 .. max<sub>ρ</sub> − 1 **do**
<br>
3:		D ← {(x, Õ(x)) : x ∈ S<sub>ρ</sub>}  *// Label the substitute training*
<br>
4:		0<sub>F</sub> ← train(F, D)  *// Train F on D to evaluate parameters θ<sub>F</sub>*
<br>
5:		S<sub>(ρ+1)</sub> ← {x + λ · sgn(J<sub>F</sub> [Õ(x)]) : x ∈ S<sub>ρ</sub>} ∪ S<sub>ρ</sub> *// Perform Jacobian-based dataset augmentation*
<br>
6:		**end for**
<br>
7:	**return** θ<sub>F</sub>
<br><br>
The function ```create_dataset()``` creates dataset out of the samples generated and ```augment_dataset()``` function augments it to the current dataset.
#### * [train.py](train.py)
This file trains the model
#### * [utilities.py](utilities.py)
This file contains helper functions 
#### * [whitebox.py](whitebox.py)
This file creates adversarial samples based on the white box (substitute) model



## References
### Papers
1. [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/pdf/1602.02697.pdf)
2. [On the Robustness of Semantic Segmentation Models to Adversarial Attacks](https://arxiv.org/pdf/1711.09856.pdf)

### Pre-trained model used in object detection
* [Minimal PyTorch implementation of YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

### Dataset used
* [COCO 2017 Val images](http://images.cocodataset.org/zips/val2017.zip)
