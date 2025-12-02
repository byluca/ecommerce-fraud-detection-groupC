# Model Comparison Usage Guide

This guide explains how to use the `model_comparison.ipynb` notebook to compare multiple machine learning models for fraud detection.

## 📋 Prerequisites

### 1. Dataset

Download the dataset from Kaggle:
- **URL**: https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions
- **Save as**: `Fraudulent_E-Commerce_Transaction_Data_merge.csv` in the project root directory

### 2. Julia Environment

Verify your environment has all required packages:

```julia
julia verify_setup.jl
```

This will check for:
- All required Julia packages
- Dataset file presence
- Installation instructions for missing components

### 3. Install Missing Packages (if needed)

```julia
using Pkg

# Core packages
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Statistics")
Pkg.add("Random")
Pkg.add("Dates")
Pkg.add("StatsBase")

# ML packages
Pkg.add("MLJ")
Pkg.add("MLJLinearModels")
Pkg.add("MLJDecisionTreeInterface")
Pkg.add("MLJLIBSVMInterface")
Pkg.add("XGBoost")

# Visualization
Pkg.add("Plots")
Pkg.add("StatsPlots")
```

## 🚀 Running the Notebook

### Option 1: Jupyter Notebook

1. Start Jupyter with Julia kernel:
   ```bash
   jupyter notebook
   ```

2. Open `model_comparison.ipynb`

3. Run all cells in sequence (Cell → Run All)

### Option 2: Julia REPL

You can also run the notebook cells sequentially in Julia REPL by copying and pasting each cell's code.

## 📊 What the Notebook Does

### Cell-by-Cell Overview

1. **Cell 1: Setup & Package Installation**
   - Installs all required packages
   - Loads libraries
   - Duration: ~5-10 minutes (first time only)

2. **Cell 2: Load Utilities**
   - Loads functions from `utils.jl` and `model_utils.jl`
   - Duration: < 1 second

3. **Cell 3: Data Loading & Preprocessing**
   - Loads CSV dataset
   - Balances dataset 50/50 (fraud/legitimate)
   - Creates 34 engineered features
   - Prepares 3-fold CV indices
   - Duration: ~30-60 seconds

4. **Cell 4: Logistic Regression**
   - Trains with L2 regularization
   - Tunes lambda parameter
   - 3-fold cross-validation
   - Duration: ~2-5 minutes

5. **Cell 5: Random Forest**
   - Tunes n_estimators and max_depth
   - 3-fold cross-validation
   - Duration: ~5-10 minutes

6. **Cell 6: Support Vector Machine (SVM)**
   - RBF kernel with cost tuning
   - 3-fold cross-validation
   - Duration: ~10-20 minutes (SVM is slow)

7. **Cell 7: XGBoost**
   - Gradient boosting with hyperparameter tuning
   - 3-fold cross-validation
   - Duration: ~5-10 minutes

8. **Cell 8: Naive Bayes**
   - Gaussian Naive Bayes
   - 3-fold cross-validation
   - Duration: < 1 minute

9. **Cell 9: Comparison & Analysis**
   - Creates comparison table
   - Business impact analysis
   - Recommendations
   - Duration: < 1 second

10. **Cell 10: Visualizations**
    - Bar charts comparing metrics
    - Visual analysis
    - Duration: ~5 seconds

### Total Runtime

**Expected total runtime**: 30-60 minutes (depending on your hardware)

## 📈 Understanding the Results

### Primary Metrics

For fraud detection, focus on:

1. **Sensitivity (Recall)** ⭐ PRIMARY METRIC
   - Percentage of frauds correctly detected
   - Higher = better fraud detection
   - Goal: Maximize to catch all frauds

2. **Precision**
   - Percentage of fraud alerts that are actually frauds
   - Higher = fewer false alarms
   - Goal: Balance with sensitivity

3. **F1 Score**
   - Harmonic mean of precision and sensitivity
   - Balanced metric

4. **F2 Score**
   - Emphasizes sensitivity over precision
   - Better for fraud detection use cases

### Business Metrics

The notebook also calculates:

- **Frauds Detected**: Number of frauds caught (TP)
- **Frauds Missed**: Number of frauds missed (FN)
- **Money Saved**: Assuming €100 per fraud prevented
- **False Alarms**: Legitimate transactions flagged (FP)
- **Review Time**: Time needed to manually review alerts

### Example Output

```
📊 COMPLETE MODEL COMPARISON TABLE

| Model               | Accuracy | Sensitivity | Specificity | Precision | F1 Score | F2 Score |
|---------------------|----------|-------------|-------------|-----------|----------|----------|
| ANN (Baseline)      | 73.41%   | **81.78%** ⭐ | 65.04%     | 70.18%    | 75.49%   | 78.26%   |
| Random Forest       | 75.23%   | 79.45%      | 71.01%      | 72.34%    | 75.73%   | 77.42%   |
| Logistic Regression | 70.12%   | 76.89%      | 63.35%      | 68.23%    | 72.31%   | 74.38%   |
| XGBoost             | 76.34%   | 80.12%      | 72.56%      | 73.45%    | 76.65%   | 78.23%   |
| SVM                 | 72.45%   | 75.23%      | 69.67%      | 71.12%    | 73.12%   | 74.05%   |

🏆 RECOMMENDATIONS
1. BEST FOR PRODUCTION: ANN (Baseline)
   Sensitivity: 81.78%
   This model catches the most frauds!
```

## 🔧 Customization

### Adjust Hyperparameters

You can modify the hyperparameter search ranges in each model cell. For example, in the Random Forest cell:

```julia
# Current
for n_trees in [100, 200]
    for max_d in [10, 20, -1]

# You can change to:
for n_trees in [50, 100, 150, 200, 250]
    for max_d in [5, 10, 15, 20, 30, -1]
```

### Add More Models

To add additional models:

1. Load the model from MLJ:
   ```julia
   NewModel = @load ModelName pkg=PackageName
   ```

2. Follow the same structure as existing model cells:
   - Loop through folds
   - Train on training set
   - Evaluate on test set
   - Store metrics

3. Add results to comparison in Cell 9

### Change Cross-Validation

Current: 3-fold CV
To change to 5-fold:

```julia
# In Cell 3, change:
cv_indices = crossvalidation(targets_bool, 3)
# To:
cv_indices = crossvalidation(targets_bool, 5)
```

Then update all model cells to loop through 5 folds instead of 3.

## ❓ Troubleshooting

### Issue: Package Installation Fails

**Solution**: Try installing packages one by one:
```julia
using Pkg
Pkg.add("PackageName")
```

If a specific package fails, check Julia version compatibility.

### Issue: Out of Memory

**Solution**: 
- Reduce hyperparameter search space
- Use smaller dataset sample
- Close other applications

### Issue: SVM Takes Too Long

**Solution**:
- Reduce hyperparameter search space in Cell 6
- Or skip SVM (comment out the cell)

### Issue: Dataset Not Found

**Solution**:
- Verify file is named exactly: `Fraudulent_E-Commerce_Transaction_Data_merge.csv`
- Verify file is in project root directory
- Check file is not corrupted (re-download if needed)

### Issue: XGBoost Errors

**Solution**:
```julia
using Pkg
Pkg.build("XGBoost")
```

Or try alternative:
```julia
Pkg.add(url="https://github.com/dmlc/XGBoost.jl")
```

## 📊 Interpreting Results for Business Decisions

### Scenario 1: Minimize Fraud Losses

**Goal**: Catch as many frauds as possible
**Metric**: Maximize Sensitivity
**Recommendation**: Choose model with highest sensitivity
**Trade-off**: Accept higher false positive rate

### Scenario 2: Minimize Manual Review Cost

**Goal**: Reduce false alarms
**Metric**: Maximize Precision
**Recommendation**: Choose model with highest precision
**Trade-off**: Some frauds will be missed

### Scenario 3: Balanced Approach

**Goal**: Balance fraud detection and review cost
**Metric**: Maximize F1 or F2 Score
**Recommendation**: Choose model with best F1/F2
**Trade-off**: Compromise on both metrics

## 🔄 Next Steps After Running

1. **Analyze Results**
   - Review comparison table
   - Check business impact analysis
   - Read recommendations

2. **Test Best Model**
   - Extract best model code
   - Test on hold-out set
   - Validate performance

3. **Deploy to Production**
   - Implement monitoring
   - A/B test against current system
   - Set up retraining pipeline

4. **Monitor Performance**
   - Track key metrics
   - Detect model drift
   - Retrain periodically (monthly recommended)

## 📚 Additional Resources

- **MLJ Documentation**: https://alan-turing-institute.github.io/MLJ.jl/dev/
- **XGBoost.jl**: https://github.com/dmlc/XGBoost.jl
- **Julia Documentation**: https://docs.julialang.org/

## 💡 Tips for Best Results

1. **First Run**: Let it complete fully to see all models
2. **Subsequent Runs**: Modify one model at a time
3. **Save Results**: Copy output tables to a separate document
4. **Compare Runs**: Track how hyperparameter changes affect performance
5. **Document Findings**: Note which configurations work best

## ✅ Success Criteria

Your run is successful if:
- ✅ All cells execute without errors
- ✅ All models produce metrics
- ✅ Comparison table shows all models
- ✅ Visualizations display correctly
- ✅ Recommendations are provided

## 🆘 Need Help?

If you encounter issues not covered here:
1. Check Julia version (should be ≥ 1.8)
2. Verify dataset integrity
3. Try running cells individually to isolate problems
4. Check package versions for compatibility
