# ecommerce-fraud-detection-groupC

Machine learning project for detecting fraudulent transactions in e-commerce data using multiple classification models and cross-validation techniques.

## 📊 Dataset

Download the dataset from Kaggle and place it in the root directory:
- **Source**: https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions?resource=download
- **Filename**: `Fraudulent_E-Commerce_Transaction_Data_merge.csv`

## 🚀 Project Structure

### Main Notebooks

1. **`miglior compromesso.ipynb`** - Baseline ANN Implementation
   - Neural Network Architecture: [128, 64, 32]
   - 34 Engineered Features
   - **Performance**:
     - Sensitivity: **81.78%** ⭐ (Primary metric for fraud detection)
     - Precision: 70.18%
     - F1 Score: 75.49%
     - Accuracy: 73.41%

2. **`model_comparison.ipynb`** - Comprehensive Model Comparison ✨ NEW
   - Implements and compares 5+ ML models:
     - ✅ Logistic Regression (L2 regularization)
     - ✅ Random Forest (with hyperparameter tuning)
     - ✅ Support Vector Machine (RBF kernel)
     - ✅ XGBoost (Gradient Boosting)
     - ✅ Naive Bayes (GaussianNB)
   - 3-fold Cross-validation for all models
   - Comprehensive metrics evaluation
   - Business impact analysis
   - Visual comparisons and recommendations

### Utility Files

- **`utils.jl`** - Core utility functions from Unit 5
  - One-hot encoding
  - Min-max normalization
  - Confusion matrix calculations
  - Cross-validation utilities
  - ANN building and training functions

- **`model_utils.jl`** - Model comparison utilities
  - Comprehensive metrics calculation
  - Results aggregation for CV
  - Business impact analysis
  - Comparison table generation

## 🎯 Feature Engineering (34 Features)

The project uses extensive feature engineering to improve fraud detection:

### Time-based Features
- Hour of transaction
- Is_Night flag (< 6 AM)
- Is_Weekend flag
- Hour_Risk flag (high-risk hours)
- Is_Early_Morning flag

### Risk Indicators
- Amount_per_AccountAge ratio
- High_Value_Flag (95th percentile)
- Very_High_Value_Flag (99th percentile)
- High_Qty_Flag (quantity > 5)
- Very_High_Qty_Flag (quantity > 10)
- Unit_Price calculation
- High_Unit_Price_Flag

### Account-based Features
- New_Account_Flag (< 30 days)
- Very_New_Account_Flag (< 7 days)
- Young_Customer_Flag (< 25 years)
- Senior_Customer_Flag (> 65 years)

### Composite Features
- Risk_Score (aggregated risk signals)
- One-hot encoded: Payment Method, Product Category, Device Used

## 📈 Model Comparison Results

Run `model_comparison.ipynb` to see the complete comparison table including:
- All performance metrics (Accuracy, Sensitivity, Specificity, Precision, F1, F2)
- Confusion matrices
- Business metrics (frauds detected, money saved, review costs)
- Visual comparisons
- Recommendations for production deployment

## 🔧 Requirements

```julia
# Core packages
CSV
DataFrames
Flux
Statistics
Random
Dates
StatsBase

# ML packages (for model comparison)
MLJ
MLJLinearModels
MLJDecisionTreeInterface
MLJLIBSVMInterface
XGBoost
NaiveBayes

# Visualization
Plots
StatsPlots
```

## 🚀 Usage

1. **Download the dataset** from the Kaggle link above

2. **Run the baseline ANN model**:
   ```julia
   # Open and run miglior compromesso.ipynb
   ```

3. **Compare all models**:
   ```julia
   # Open and run model_comparison.ipynb
   ```

## 📊 Key Metrics

For fraud detection, **Sensitivity (Recall)** is the primary metric because:
- Minimizes False Negatives (missed frauds)
- Cost of missing fraud >> cost of false alarm
- Business priority: catch as many frauds as possible

## 🏆 Best Model

The best model for production is determined by the highest sensitivity while maintaining acceptable precision. See `model_comparison.ipynb` for the complete analysis and recommendation.

## 📝 Notes

- Dataset is balanced 50/50 (fraud/legitimate) for training
- 3-fold stratified cross-validation used throughout
- All models use the same preprocessing pipeline for fair comparison
- Hyperparameter tuning performed for each model to maximize sensitivity

## 👥 Team

Group C - Machine Learning Project
