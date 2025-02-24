# Project Proposal

## 1️⃣ Preprocessing the Data
- Data should not be deleted, and values such as mean, median, and outliers should not be replaced or removed in any way.  
- **Normalization** for numerical features.
- Since this is a classification task, data balancing is required using **SMOTE**.

## 2️⃣ Training Deep Learning Models
Train classification models such as **1D-CNN, RNN, DNN, LSTM, and BiLSTM** on the data.

- The **label** column should be categorized according to air quality levels:

```python
 def categorize_air_quality(value): 
     if 0 <= value <= 50:
         return 'Good' 
     elif 51 <= value <= 100:
         return 'Moderate'
     elif 101 <= value <= 150:
         return 'Unhealthy for Sensitive'
     elif 151 <= value <= 200:
         return 'Unhealthy'
     elif 201 <= value <= 400:
         return 'Hazardous'
     else:
         return 'Unknown'
```

## 3️⃣ Evaluating Model Accuracy
- Use metrics like **accuracy, recall, precision, and F-score**.
- The model with the highest accuracy is selected for improvement using **optimization techniques** such as:
  1. Weight Pruning
  2. Quantization
  3. Clustering
  4. Weight Clipping
  5. Knowledge Distillation
  6. Post-Training Quantization
- Avoid any techniques that could cause issues.

## 4️⃣ Re-evaluating the Best Model
- The best model after optimization is selected again.
- Perform classification and note the accuracy.
- Predict **temperature and relative humidity** using specific columns and measure prediction accuracy using **MSE, RMSE, R², and MAE**.

## 5️⃣ Visualization and Comparison
- Compare models before and after optimization using charts, diagrams, and tables.
- Examples: Accuracy and loss curves before and after **SMOTE**, outlier removal, etc.

## 6️⃣ Project Implementation
- The project is implemented using **Google Colab**.
- A simplified workflow diagram is included.
