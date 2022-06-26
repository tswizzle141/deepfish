# deepfish
#### [Specially thanks to Mr.@IssamLaradji for the code] We have done this project for the Student Scientific Research Award held annually by Hanoi University of Science and Technology, and we have been honored with the Third Prize.
## Our tasks
* Weakly-supervised segmentation: With training images including fully-masked and point-level images (see the [DeepFish Dataset](https://alzayats.github.io/DeepFish/)).
## Our contributions
* We keep the architecture from @IssamLaradji except for proposing new baseline and new loss function.
![Proposed Architecture](https://github.com/tswizzle141/deepfish/blob/main/train/affinity-based%20architecture.png)
* In the baseline, we have changed the FCN8-VGG16 to FCN8-wide-ResNet50. A Progressive Atrous Spatial Pyramid Pooling (PASPP) is added between the encoder and the decoder of the FCN8-base.
![PASPP](https://github.com/tswizzle141/deepfish/blob/main/train/PASPP.png)
![Proposed Baseline](https://github.com/tswizzle141/deepfish/blob/main/train/fcn%20backbone.png)
* In the loss function, we combine the LCFCN loss function and our Focal-Cross-Entropy loss function for weak-segmentation
## Results
