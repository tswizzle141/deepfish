# deepfish
#### [Specially thanks to Mr.@IssamLaradji for the code] We have done this project for the Student Scientific Research Award held annually by Hanoi University of Science and Technology, and we have been honored with the Third Prize.
## Our tasks
* Weakly-supervised segmentation: With training images including fully-masked and point-level images (see the [DeepFish Dataset](https://alzayats.github.io/DeepFish/)).
* Fishes localization and counting
## Our contributions
* We keep the architecture from @IssamLaradji (with affinity and random walk module) except for proposing new baseline and new loss function.
![Proposed Architecture](https://github.com/tswizzle141/deepfish/blob/main/train/affinity-based%20architecture.png)
* In the baseline, we have changed the FCN8-VGG16 to FCN8-wide-ResNet50. A Progressive Atrous Spatial Pyramid Pooling (PASPP) is added between the encoder and the decoder of the FCN8-base.
![PASPP](https://github.com/tswizzle141/deepfish/blob/main/train/PASPP.png)
![Proposed Baseline](https://github.com/tswizzle141/deepfish/blob/main/train/fcn%20backbone.png)
* In the loss function, we combine the LCFCN loss function and our Focal-Cross-Entropy loss function for weakly-supervised segmentation:
$$L_{proposed\_loss} = L_{Focal-CE} + L_{LCFCN}$$
with:
$$L_{LCFCN}(S,T) = L_I(S,T) + L_P(S,T) + L_S(S,T) + L_P(S,T)$$
$$L_{Focal-CE} = (1-L_{CE})^{\gamma}$$
$\gamma>0$ is parameter; $L_{CE}$ is the conventional Cross-Entropy:
$$L_{CE} = -tln(p) - (1-t)ln(1-p)$$
## Results
We have trained all cases end-to-end by ourselves, with optimizer Nesterov-Adam, initial learning rate of 1e-5.
![table1](https://github.com/tswizzle141/deepfish/blob/main/train/1.jpg)
![table2](https://github.com/tswizzle141/deepfish/blob/main/train/2.jpg)
![table3](https://github.com/tswizzle141/deepfish/blob/main/train/3.jpg)
