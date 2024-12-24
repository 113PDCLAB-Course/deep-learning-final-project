# deep-learning-final-project
### 1. Promblem
#### (1) Time Consuming
Confirming the presence of a brain tumor requires considerable time and effort.
#### (2) Analyze
People cannot understand CT scans or medical instructions, explanations can be generated using LLMs.
### 2. Purpose
Our purpose is to detected whether there's tomor or not.
### 3. Method
#### (1) Data processing
The following pre-processing was applied to each image:
- Auto-orientation of pixel data (with EXIF-orientation stripping)
- Resize to 512x512(Stretch) scale pixel values **between 0 and 1**, ensuring data consistency.
- Convert the original annotation data into **binary masks** to identify tumor regions.
- Employs a data generator for batch processing, while implementing data augmentation techniques to expand the training set.

#### (2) Feature we choose
The dataset includes 2146 images.Tumors are annotated in COCO Segmentation format, these 2146 pictures are all pictures with tumors. We use all labels because we are figuring out the classification task.
- test data: 215 images
- train data: 1502 images
- validation data: 429 images

#### (3) Model
We choose **3** models to compare, UNet, ResNext50, ResUNet++. And we analyze these three models below :
- **UNet**

  - **Main feature:** Classic model designed for medical image segmentation. Adopts an encoder-decoder architecture and uses skip connections in detailed features are introduced during the decoding process.

**Advantage:** The computing cost is low, the architecture is simple and easy to deploy.

**Limitation:** Limited performance in capturing global context and handling small targets.

- **ResNext50**

ResNext50 combine the architecture of ResNet and Next, uses **grouped convolution** to reduce parameter count, which helps divide input channels into groups for parallel processing, reducing computational complexity. Another thing is that ResNext50 add **incorporates skip connections**, allowing direct information flow between layers, helping prevent gradient vanishing and enabling better feature preservation. It also mmploy **Batch Normalization and ReLU activation functions**, provide stabilize training and prevent vanishing gradients.
- **ResUNet++**

**Main feature:** Enhanced 3D encoding-decoding Model. Use pre-training ResNet50 backbone and 3D dense convolution blocks and volumes product transpose layer.

**Application areas:** Multimodal MRI brain tumor segmentation.

**Advantage:** Handling multimodal volume numbers excellent performance, amd improve segmentation accuracy and efficiency.

#### (4) Evaluation 
test_loss, test_accuracy, Accuracy Curve, Loss Curve, Confusion Matrix
#### (5) Training Strategy
We mainly focus on the data enhancement part (data preprocessing part code)
### 4. Execution
#### (1) Execution Environment
![image](./environment.jpg)

### 5. Challenge we faced
- RAM too small

**solution:**
- Early stop

**solution:** 
- Disconnect weight file clear
  
**solution:** save it to Google Drive
### 6. Conclusion
#### (1) Result

