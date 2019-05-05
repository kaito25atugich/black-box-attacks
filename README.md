# Black-Box Attacks on Neural Networks

## Abstract
The paper[1] discusses an algorithm which allows us to craft an adversarial attack on black box networks. The attacker has no knowledge of internals or training data of the victim.
<br><br>
The solution presented treats the black box as an oracle and gets the output for several inputs and trains a substitute model on this data. Then adversarial samples are created by a white box attack on this substituted model. These adversarial samples work well to attack on the black box.
<br><br>
In this project, by the time of midterm review, we implemented this algorithm on MNIST dataset. Now, we tried to implement this on object detection on COCO dataset.

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

## SetUp & Instructions


## Results


## Additional Details
### Algorithms
#### Substitute DNN Training: 
For oracle *Õ*, a maximum number max *ρ* of substitute training epochs, a substitute architecture *F* , and an initial training set *S<sub>0</sub>*.
<br>
Input: *Õ*, *max<sub>ρ</sub>* , *S<sub>0</sub>* , *λ*
<br>
1:	Define architecture F
<br>
2:	**for** ρ ∈ 0 .. max<sub>ρ</sub> − 1 **do**
<br>
3:		*// Label the substitute training*
<br>
4:		D ← {(x, Õ(x)) : x ∈ S<sub>ρ</sub>}
<br>
5:		*// Train F on D to evaluate parameters θ<sub>F</sub>*
<br>
6:		0<sub>F</sub> ← train(F, D)
<br>
7:		*// Perform Jacobian-based dataset augmentation*
<br>
8:		S<sub>(ρ+1)</sub> ← {x + λ · sgn(J<sub>F</sub> [Õ(x)]) : x ∈ S<sub>ρ</sub>} ∪ S<sub>ρ</sub>
<br>
9:		**end for**
<br>
10:	**return** θ<sub>F</sub>


## References
### Papers
1. [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/pdf/1602.02697.pdf)
2. [On the Robustness of Semantic Segmentation Models to Adversarial Attacks](https://arxiv.org/pdf/1711.09856.pdf)

### Pre-trained model used in object detection
* [Minimal PyTorch implementation of YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

### Dataset used
* [COCO 2017 Val images](http://images.cocodataset.org/zips/val2017.zip)
