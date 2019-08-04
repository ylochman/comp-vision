# Paper Review:  D2-Net

* **Title**: D2-Net: A Trainable CNN for Joint Description and Detection of Local Features
* **Authors**: Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, Torsten Sattler
* **[Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Attention-Guided_Unified_Network_for_Panoptic_Segmentation_CVPR_2019_paper.pdf)**
* **Tags**: Joint Feature Description and Detection, Correspondence, Convolutional Neural Network
* **Year**: 2019

# Summary

* **What:**

  * The authors proposed a CNN architecture for simultaneous dense feature description and detection in order to find reliable pixel-level correspondences under difficult imaging conditions. 
  * D2-Net obtains state-of-the-art performance on **Aachen Day-Night** (outdoor) and **InLoc** (indoor) localization datasets.
  * The method can be integrated into image matching and 3D reconstruction pipelines.

* **How:**
  * It's a "single-shot" detect-and-describe (D2) approach. A VGG-16 (up to the `conv4_3` layer) backbone is fine-tuned for extracting feature maps:

  ![Summary](assets/summary.png?raw=true "D2Net")

  * Local descriptors (d_ij) are obtained by traversing n feature maps (l2-normalized across channels) at a spatial position (i,j)
  * Detections (scores -- s_ij) are obtained by performing a soft versions of non-local-maximum suppression on a feature map (soft local-maximum score α) + non-maximum suppression across each descriptor (ratio-to-maximum score per descriptor β).
  * Also, during the inference authors propose to create image pyramids for 3 scales: 0.5, 1, 2; then pass through the network and sum the feature maps (using bilinear interpolation for larger iamges and masking already detected regions to prevent re-detection) 
  * The objective corresponds to the 
  repeatability of the detector and the distinctiveness of the descriptor. It is an extended triplet margin ranking loss.

* **Results:**

  * **HPatches**

![HPatches Results](assets/res_hpatches.png?raw=true   "HPatches Results")

  * **Aachen Day-Night localization dataset**

![Aachen Dataset Results](assets/res_aachen.png?raw=true   "Aachen Dataset Results")

  * **InLoc indoor localization dataset**
  
![InLoc Dataset Results](assets/res_inloc.png?raw=true   "InLoc Dataset Results")


# CNN Visualization: D2-Net
## Architecture
![Architecture](assets/architecture.png?raw=true "D2Net")

## Blocks description
![Modules](assets/modules.png?raw=true "D2Net")