# Black-Box Attacks on Neural Networks

## Abstract
The paper[1] discusses an algorithm which allows us to craft an adversarial attack on black box networks. The attacker has no knowledge of internals or training data of the victim.
<br><br>
The solution presented treats the black box as an oracle and gets the output for several inputs and trains a substitute model on this data. Then adversarial samples are created by a white box attack on this substituted model. These adversarial samples work well to attack on the black box.
<br><br>
In this project, by the time of midterm review, we implemented this algorithm on MNIST dataset. Now, we tried to implement this on object detection on COCO dataset.
We separate the bounding boxes of object detected as new images, create adversarial example fro mit, and stitch the adversarial example to original image. 

## Requirements
```
python >=3.5
numpy
torch
torchvision
matplotlib
tensorflow
tensorboard
terminaltables
pillow
tqdm
libtiff
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
It asks to save the model. Give it a proper name. The mode lis saved in the folder ```saved_models/```
* Create adversarial samples
```
python3 main_script.py --adv
```
It asks for which substitute model to use, num_samples to be generated. This generates the adversarial samples and stores them in directory ```adv_samples/```
* Stitch the adversarial examples generated to otriginal image
```
python3 main_script.py --stitch
```
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
### Folders and Files
#### * [dataset.py](dataset.py)
#### * [main_script.py](main_script.py)
#### * [oracle.py](oracle.py)
#### * [predict.py](predict.py)
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
#### * [utilities.py](utilities.py)
#### * [whitebox.py](whitebox.py)



## References
### Papers
1. [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/pdf/1602.02697.pdf)
2. [On the Robustness of Semantic Segmentation Models to Adversarial Attacks](https://arxiv.org/pdf/1711.09856.pdf)

### Pre-trained model used in object detection
* [Minimal PyTorch implementation of YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

### Dataset used
* [COCO 2017 Val images](http://images.cocodataset.org/zips/val2017.zip)
