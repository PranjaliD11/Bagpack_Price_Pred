# 🎒 Backpack Price Prediction

A regression problem part of Kaggle's Playground Series where the goal was to predict backpack prices using machine learning techniques. This project ranked in the **top 28%** on the leaderboard.

---

## 📌 Overview

This project involves building a regression model to predict the prices of backpacks based on various product-related features such as brand, material, size, style, and color. The dataset provided contains **3.99M+ rows**, making it rich but also challenging due to the presence of **numerous missing values**.

The target variable is **`Price`**, and all other columns are treated as independent features. A combination of data preprocessing, imputation, encoding, and scaling techniques were applied to prepare the data for modeling.

---

## 🔄 Workflow

### 1. **Handling Missing Values**
- **Numeric Features**:  
  - Used **box plots** to check for outliers.  
  - Since no significant outliers were found, **mean imputation** was used.
- **Categorical Features**:  
  - Used `SimpleImputer` with `strategy="constant"` and `fill_value="Unknown"`.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
```

---

### 2. **Feature Engineering & Encoding**
- **Manual Mapping**:
```python
size_mapping = {
    'Unknown': 0,
    'Small': 1,
    'Medium': 2,
    'Large': 3
}
train_merge['Size'] = train_merge['Size'].replace(size_mapping)
```

- **Label Encoding**:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat_cols = ['Brand', 'Material', 'Style', 'Color']

for feature in cat_cols:
    train_merge[feature] = le.fit_transform(train_merge[feature])
    test[feature] = le.transform(test[feature])
```

---

### 3. **Feature Scaling**
- Applied **MinMaxScaler** to bring all features into the (0,1) range, which helped improve the **Root Mean Square Error (RMSE)**.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_scale = [feature for feature in train_merge if feature not in ['id', 'Price']]
```

---

### 4. **Modeling**
- **Models Used**:
  - Linear Regression
  - XGBoost Regressor ✅ *(Best Performer)*

- **Best RMSE Achieved**: **38.85** with XGBoost

---

## 📂 Files Included

- `notebook.ipynb`: Jupyter Notebook with full code implementation
- `train.csv`: Training data provided by Kaggle
- `test.csv`: Testing data for prediction
- `submission.csv`: Final submission file used on Kaggle

---

Feel free to fork or clone this repository to explore the dataset and try your own modeling approach!

