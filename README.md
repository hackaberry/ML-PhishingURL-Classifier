# ML-PhishingURL-Classifier

In this project, I explored the use of fully connected neural networks (or multi-layer perceptrons) to classify websites as **phishing** or **legitimate** based on 30 features from a public dataset.

---

## Dataset

The dataset used in this project comes from the UC Irvine Machine Learning repository and can be found here: https://archive.ics.uci.edu/dataset/327/phishing+websites

The dataset consists of **11,055** samples each with 30 features to describe the website.
Additionally, each sample is labeled a **'-1' for phishy** or a **'1' for legit**.

Labels were remapped to:
- `0` → Phishy  
- `1` → Legit  
for ease of handling.

---

## Model Architectures

| Model     | Hidden Layers | Neurons per Layer |
|-----------|----------------|------------------|
| ModelV0   | 3              | 5                |
| ModelV1   | 6              | 5                |
| ModelV2   | 3              | 10               |
| ModelV3   | 6              | 10               |

---

## Training Setup

All models were trained under the same setup to be fair to each architecture and accurately evaluate them:
- **Loss Function**: `BCEWithLogitsLoss`  
- **Optimizer**: `SGD`  
- **Learning Rate**: 0.1
- **Activation**: `ReLU`  
- **Output Activation**: `sigmoid`
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report  
- **Train Split**: 80/20
- **Batch Size**: 64  
- **Epochs**: 100  
- **Hardware**: CPU (device-agnostic code is provided and allows for CUDA-enabled devices)
- **Note**: Each model architecture was trained 5 times. The best of each were kept for evaluation


---

## Findings
Below is a summary of each model along with its metrics:
```
ModelV0 Accuracy: 93.8%
ModelV0 Confusion Matrix:
[[ 904   92]
 [  45 1170]]
ModelV0 Classification Report:
              precision    recall  f1-score   support

      Phishy       0.95      0.91      0.93       996
       Legit       0.93      0.96      0.94      1215

    accuracy                           0.94      2211
   macro avg       0.94      0.94      0.94      2211
weighted avg       0.94      0.94      0.94      2211

==================================

ModelV1 Accuracy: 93.22%
ModelV1 Confusion Matrix:
[[ 942   54]
 [  96 1119]]
ModelV1 Classification Report:
              precision    recall  f1-score   support

      Phishy       0.91      0.95      0.93       996
       Legit       0.95      0.92      0.94      1215

    accuracy                           0.93      2211
   macro avg       0.93      0.93      0.93      2211
weighted avg       0.93      0.93      0.93      2211

==================================

ModelV2 Accuracy: 94.53%
ModelV2 Confusion Matrix:
[[ 921   75]
 [  46 1169]]
ModelV2 Classification Report:
              precision    recall  f1-score   support

      Phishy       0.95      0.92      0.94       996
       Legit       0.94      0.96      0.95      1215

    accuracy                           0.95      2211
   macro avg       0.95      0.94      0.94      2211
weighted avg       0.95      0.95      0.95      2211

==================================

ModelV3 Accuracy: 95.21%
ModelV3 Confusion Matrix:
[[ 943   53]
 [  53 1162]]
ModelV3 Classification Report:
              precision    recall  f1-score   support

      Phishy       0.95      0.95      0.95       996
       Legit       0.96      0.96      0.96      1215

    accuracy                           0.95      2211
   macro avg       0.95      0.95      0.95      2211
weighted avg       0.95      0.95      0.95      2211

==================================
```
As we can see, ModelV3 performed the best and was able to predict with an accuracy of 95% whether a website was an attempt to phish or was legit. ModelV0 received a large number of False Positives, while ModelV1 received a large number of False Negatives. Overall, the metrics for ModelV3 look the best in all areas.

---

## Conclusions:
Deeper models performed better only when also paired with more neurons. Mini-batches of size 64 used for training also led consistently to better convergence compared to full-batch training.