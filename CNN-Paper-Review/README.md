# Paper Review:  D2-Net

* **Title**: D2-Net: A Trainable CNN for Joint Description and Detection of Local Features
* **Authors**: Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, Torsten Sattler
* **[Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Attention-Guided_Unified_Network_for_Panoptic_Segmentation_CVPR_2019_paper.pdf)**
* **Tags**: Joint Feature Description and Detection, Correspondence, Convolutional Neural Network
* **Year**: 2019

# Summary

* **What:**
The authors propose a CNN architecture for simultaneous dense feature description and detection in order to find reliable pixel-level correspondences under difficult imaging conditions. 
  ![Summary](assets/summary.png?raw=true "D2Net")

* **How:**
  * It's a "single-shot" pipeline.
   <!-- * `Spatial Transformer` allows the spatial manipulation of the data (any feature map or particularly input image). This differentiable module can be inserted into any CNN, giving neural networks the ability to actively spatially transform feature maps, conditional on the feature map itself.
   * The action of the spatial transformer is conditioned on individual data samples, with the appropriate behavior learned during training for the task in question.
   * No additional supervision or modification of the optimization process is required.
   * Spatial manipulation consists of cropping, translation, rotation, scale, and skew.
   ![Example](images/STN/stn_example2.png?raw=true "Example") ![Example2](images/STN/stn_example.png?raw=true "Example2")
   * STN structure:
        1. `Localization net`: predicts parameters of the transform `theta`. For 2d case, it's 2 x 3 matrix. For 3d case, it's 3 x 4 matrix.
        2. `Grid generator`: Uses predictions of `Localization net` to create a sampling grid, which is a set of points where the input map should be sampled to produce the transformed output.
        3. `Sampler`: Produces the output map sampled from the input feature map at the predicted grid points. -->
              
* **Results:**

  * **Aachen Day-Night localization dataset**

  * **InLoc indoor localization dataset**
  
    <!-- * **Street View House Numbers multi-digit recognition**:
      ![SVHN Results](images/STN/stn_svhn_results.png?raw=true   "SVHN Results")
    * **Distored MNIST**:
      ![Distorted MNIST Results](images/STN/stn_distored_mnist_results.png?raw=true "Distorted MNIST")
    * **CUB-200-2011 birds dataset**:
      ![Birds Classification Results](images/STN/stn_birds_results.png?raw=true "Birds Classification Results")
    * **MNIST addition**:
      ![MNIST addition Results](images/STN/stn_mnist_addition_results.png?raw=true "MNIST addition Results")   -->

# CNN Visualization: D2-Net
## Architecture
![Architecture](assets/architecture.png?raw=true "D2Net")

## Blocks description
![Modules](assets/modules.png?raw=true "D2Net")