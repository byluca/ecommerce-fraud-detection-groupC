# Quick Start Example

This is a simplified example showing how to run a single model from the comparison notebook.
Use this to quickly test your setup before running the full comparison.

## Prerequisites

```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "Statistics", "Random", "MLJ", "MLJLinearModels"])
```

## Quick Test: Logistic Regression

```julia
# Load packages
using CSV, DataFrames, Statistics, Random, MLJ

# Load utilities
include("utils.jl")
include("model_utils.jl")

# Load a small sample of data for testing
println("Loading dataset...")
df = CSV.read("Fraudulent_E-Commerce_Transaction_Data_merge.csv", DataFrame)

# Take a small sample (1000 frauds + 1000 legitimate)
target_col = "Is Fraudulent"
fraud_sample = df[df[:, target_col] .== 1, :][1:min(1000, sum(df[:, target_col] .== 1)), :]
legit_sample = df[df[:, target_col] .== 0, :][1:min(1000, sum(df[:, target_col] .== 0)), :]

df_sample = vcat(fraud_sample, legit_sample)
df_sample = df_sample[shuffle(1:size(df_sample, 1)), :]

println("Sample size: $(size(df_sample, 1)) transactions")

# Simple preprocessing (without all 34 features - just basics for testing)
function quick_preprocess(dataframe)
    data = copy(dataframe)
    
    # Drop ID columns
    cols_to_drop = ["Transaction ID", "Customer ID", "Transaction Date",
                    "IP Address", "Shipping Address", "Billing Address", "Customer Location"]
    select!(data, Not(intersect(names(data), cols_to_drop)))
    
    # Simple imputation
    for col in names(data)
        if eltype(data[!, col]) <: Union{Missing, Number}
            if any(ismissing, data[!, col])
                median_val = median(skipmissing(data[!, col]))
                replace!(data[!, col], missing => median_val)
            end
        end
    end
    
    # One-hot encode categoricals
    categorical_cols = ["Payment Method", "Product Category", "Device Used"]
    numeric_df = data[:, setdiff(names(data), categorical_cols)]
    
    for col in categorical_cols
        if col in names(data)
            encoded = oneHotEncoding(data[!, col])
            new_names = ["$(col)_$(i)" for i in 1:size(encoded, 2)]
            encoded_df = DataFrame(encoded, new_names)
            numeric_df = hcat(numeric_df, encoded_df)
        end
    end
    
    return numeric_df
end

println("Preprocessing...")
df_processed = quick_preprocess(df_sample)

# Prepare arrays
input_cols = setdiff(names(df_processed), [target_col])
X = Matrix{Float64}(df_processed[:, input_cols])
y = Bool.(vec(df_processed[:, target_col]))

println("Features: $(size(X, 2))")
println("Samples: $(size(X, 1))")

# Simple train/test split (80/20)
n = size(X, 1)
n_train = Int(floor(0.8 * n))
indices = shuffle(1:n)
train_idx = indices[1:n_train]
test_idx = indices[n_train+1:end]

X_train = X[train_idx, :]
y_train = y[train_idx]
X_test = X[test_idx, :]
y_test = y[test_idx]

# Normalize
norm_params = calculateMinMaxNormalizationParameters(X_train)
X_train_norm = normalizeMinMax(X_train, norm_params)
X_test_norm = normalizeMinMax(X_test, norm_params)

println("\nTraining Logistic Regression...")

# Load and train model
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
model = LogisticClassifier(lambda=0.1)
mach = machine(model, X_train_norm, y_train)
MLJ.fit!(mach, verbosity=1)

println("\nMaking predictions...")
y_pred = MLJ.predict_mode(mach, X_test_norm)

# Calculate metrics
metrics = calculate_comprehensive_metrics(y_test, y_pred)

# Print results
print_model_results("Logistic Regression (Quick Test)", metrics)

println("\n✅ Quick test complete!")
println("\nIf this worked, you're ready to run the full model_comparison.ipynb notebook!")
```

## Expected Output

```
Loading dataset...
Sample size: 2000 transactions
Preprocessing...
Features: 20
Samples: 2000

Training Logistic Regression...
[ Info: Training Machine{LogisticClassifier,...}.

Making predictions...

======================================================================
📊 RESULTS: Logistic Regression (Quick Test)
======================================================================
Accuracy:    XX.XX%
Sensitivity: XX.XX% ⭐ [PRIMARY METRIC]
Specificity: XX.XX%
Precision:   XX.XX%
F1 Score:    XX.XX%
F2 Score:    XX.XX%

🎯 FRAUD DETECTION PERFORMANCE:
   ✅ Detected: XXX
   ❌ Missed:   XXX
   💰 Saved:    €XXXXX
   ⚠️  False Alarms: XXX (~X.Xh review)

✅ Quick test complete!

If this worked, you're ready to run the full model_comparison.ipynb notebook!
```

## Troubleshooting

### If you get "Package not found" errors:
```julia
using Pkg
Pkg.add("PackageName")
```

### If you get "Dataset not found" errors:
Make sure you downloaded the dataset from Kaggle and saved it as:
`Fraudulent_E-Commerce_Transaction_Data_merge.csv`

### If metrics look strange:
This quick test uses a small sample and simplified features.
The full notebook uses 34 engineered features and all data for better results.

## Next Steps

Once this quick test works:
1. Open `model_comparison.ipynb`
2. Run all cells sequentially
3. Review the complete comparison results

The full notebook will:
- Use all data (not just a sample)
- Create 34 engineered features (not just basics)
- Run 3-fold cross-validation (not just one split)
- Compare 5 models (not just one)
- Provide detailed analysis and visualizations
