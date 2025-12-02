# ============================================================================
#          MODEL COMPARISON UTILITY FUNCTIONS
#          Support functions for model_comparison.ipynb
# ============================================================================

using Statistics, Random, StatsBase

"""
    calculate_comprehensive_metrics(y_true, y_pred)

Calculate comprehensive evaluation metrics for binary classification.
Returns a dictionary with accuracy, sensitivity, specificity, precision, F1, F2, and confusion matrix values.
"""
function calculate_comprehensive_metrics(y_true::Vector{Bool}, y_pred::Vector{Bool})
    TP = sum(y_pred .& y_true)
    TN = sum(.!y_pred .& .!y_true)
    FP = sum(y_pred .& .!y_true)
    FN = sum(.!y_pred .& y_true)
    
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total
    sensitivity = TP + FN > 0 ? TP / (TP + FN) : 0.0  # Recall/TPR
    specificity = TN + FP > 0 ? TN / (TN + FP) : 0.0  # TNR
    precision = TP + FP > 0 ? TP / (TP + FP) : 0.0    # PPV
    npv = TN + FN > 0 ? TN / (TN + FN) : 0.0          # Negative Predictive Value
    
    # F-scores
    f1 = precision + sensitivity > 0 ? 2 * precision * sensitivity / (precision + sensitivity) : 0.0
    beta = 2.0  # F2 emphasizes recall
    f2 = (precision + beta^2 * sensitivity) > 0 ? 
         (1 + beta^2) * precision * sensitivity / (beta^2 * precision + sensitivity) : 0.0
    
    return Dict(
        "accuracy" => accuracy,
        "sensitivity" => sensitivity,
        "specificity" => specificity,
        "precision" => precision,
        "npv" => npv,
        "f1" => f1,
        "f2" => f2,
        "TP" => Int(TP),
        "TN" => Int(TN),
        "FP" => Int(FP),
        "FN" => Int(FN)
    )
end

"""
    print_model_results(model_name, metrics)

Print formatted results for a model.
"""
function print_model_results(model_name::String, metrics::Dict)
    println("\n" * "="^70)
    println("📊 RESULTS: $model_name")
    println("="^70)
    println("Accuracy:    $(round(metrics["accuracy"]*100, digits=2))%")
    println("Sensitivity: $(round(metrics["sensitivity"]*100, digits=2))% ⭐ [PRIMARY METRIC]")
    println("Specificity: $(round(metrics["specificity"]*100, digits=2))%")
    println("Precision:   $(round(metrics["precision"]*100, digits=2))%")
    println("F1 Score:    $(round(metrics["f1"]*100, digits=2))%")
    println("F2 Score:    $(round(metrics["f2"]*100, digits=2))%")
    println("\n🎯 FRAUD DETECTION PERFORMANCE:")
    println("   ✅ Detected: $(metrics["TP"])")
    println("   ❌ Missed:   $(metrics["FN"])")
    println("   💰 Saved:    €$(metrics["TP"] * 100)")
    println("   ⚠️  False Alarms: $(metrics["FP"]) (~$(round(metrics["FP"]*2/60, digits=1))h review)")
end

"""
    aggregate_cv_metrics(metrics_list)

Aggregate metrics from cross-validation folds.
"""
function aggregate_cv_metrics(metrics_list::Vector)
    if isempty(metrics_list)
        return nothing
    end
    
    return Dict(
        "accuracy" => mean([m["accuracy"] for m in metrics_list]),
        "sensitivity" => mean([m["sensitivity"] for m in metrics_list]),
        "specificity" => mean([m["specificity"] for m in metrics_list]),
        "precision" => mean([m["precision"] for m in metrics_list]),
        "npv" => mean([m["npv"] for m in metrics_list]),
        "f1" => mean([m["f1"] for m in metrics_list]),
        "f2" => mean([m["f2"] for m in metrics_list]),
        "TP" => sum([m["TP"] for m in metrics_list]),
        "TN" => sum([m["TN"] for m in metrics_list]),
        "FP" => sum([m["FP"] for m in metrics_list]),
        "FN" => sum([m["FN"] for m in metrics_list]),
        # Standard deviations
        "accuracy_std" => std([m["accuracy"] for m in metrics_list]),
        "sensitivity_std" => std([m["sensitivity"] for m in metrics_list]),
        "specificity_std" => std([m["specificity"] for m in metrics_list]),
        "precision_std" => std([m["precision"] for m in metrics_list]),
        "f1_std" => std([m["f1"] for m in metrics_list]),
        "f2_std" => std([m["f2"] for m in metrics_list])
    )
end

"""
    create_comparison_table(models_dict)

Create a markdown-style comparison table for all models.
"""
function create_comparison_table(models_dict::Dict)
    println("\n" * "="^90)
    println("📊 COMPLETE MODEL COMPARISON TABLE")
    println("="^90)
    
    println("\n| Model               | Accuracy | Sensitivity | Specificity | Precision | F1 Score | F2 Score |")
    println("|---------------------|----------|-------------|-------------|-----------|----------|----------|")
    
    for (name, metrics) in sort(collect(models_dict), by=x->x[2]["sensitivity"], rev=true)
        acc = round(metrics["accuracy"]*100, digits=2)
        sens = round(metrics["sensitivity"]*100, digits=2)
        spec = round(metrics["specificity"]*100, digits=2)
        prec = round(metrics["precision"]*100, digits=2)
        f1 = round(metrics["f1"]*100, digits=2)
        f2 = round(metrics["f2"]*100, digits=2)
        
        # Highlight best sensitivity
        if metrics["sensitivity"] == maximum([m["sensitivity"] for m in values(models_dict)])
            println("| $name | $acc% | **$sens%** ⭐ | $spec% | $prec% | $f1% | $f2% |")
        else
            println("| $name | $acc% | $sens% | $spec% | $prec% | $f1% | $f2% |")
        end
    end
    
    println("\n⭐ Best model by sensitivity (primary metric for fraud detection)")
    println("="^90)
end

"""
    print_business_analysis(models_dict)

Print business impact analysis for all models.
"""
function print_business_analysis(models_dict::Dict)
    println("\n" * "="^80)
    println("💼 BUSINESS IMPACT ANALYSIS")
    println("="^80)
    
    for (name, metrics) in models_dict
        total_frauds = metrics["TP"] + metrics["FN"]
        detection_rate = metrics["TP"] / total_frauds * 100
        money_saved = metrics["TP"] * 100
        review_hours = metrics["FP"] * 2 / 60
        false_alarm_rate = (metrics["FP"] / (metrics["FP"] + metrics["TP"])) * 100
        
        println("\n$name:")
        println("   ✅ Frauds detected: $(metrics["TP"]) / $total_frauds ($(round(detection_rate, digits=1))%)")
        println("   ❌ Frauds missed: $(metrics["FN"])")
        println("   💰 Money saved: €$money_saved")
        println("   ⚠️  False alarms: $(metrics["FP"]) ($(round(false_alarm_rate, digits=1))%)")
        println("   ⏱️  Review time: $(round(review_hours, digits=1)) hours")
    end
    
    println("\n" * "="^80)
end

println("✅ Model comparison utility functions loaded!")
