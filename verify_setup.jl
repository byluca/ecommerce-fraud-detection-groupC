# ============================================================================
#          SETUP VERIFICATION SCRIPT
#          Checks if all required packages are installed for model_comparison.ipynb
# ============================================================================

println("🔍 Verifying Julia environment for model_comparison.ipynb")
println("="^70)

using Pkg

# Required packages
required_packages = [
    "CSV",
    "DataFrames",
    "Statistics",
    "Random",
    "Dates",
    "StatsBase",
    "MLJ",
    "MLJLinearModels",
    "MLJDecisionTreeInterface",
    "MLJLIBSVMInterface",
    "XGBoost",
    "Plots",
    "StatsPlots"
]

println("\n📦 Checking installed packages...\n")

missing_packages = String[]
installed_count = 0

for pkg in required_packages
    try
        # Try to load the package
        eval(Meta.parse("using $pkg"))
        println("   ✅ $pkg")
        installed_count += 1
    catch e
        println("   ❌ $pkg - NOT INSTALLED")
        push!(missing_packages, pkg)
    end
end

println("\n" * "="^70)
println("📊 Summary:")
println("   Installed: $installed_count / $(length(required_packages))")

if isempty(missing_packages)
    println("\n✅ All required packages are installed!")
    println("   You can run model_comparison.ipynb")
else
    println("\n⚠️  Missing packages: $(join(missing_packages, ", "))")
    println("\n📥 To install missing packages, run:")
    println("   using Pkg")
    for pkg in missing_packages
        println("   Pkg.add(\"$pkg\")")
    end
end

# Check for dataset
println("\n" * "="^70)
println("📂 Checking for dataset...")

data_path = "Fraudulent_E-Commerce_Transaction_Data_merge.csv"
if isfile(data_path)
    filesize = stat(data_path).size / (1024 * 1024)  # MB
    println("   ✅ Dataset found: $data_path ($(round(filesize, digits=2)) MB)")
else
    println("   ❌ Dataset NOT found: $data_path")
    println("\n📥 Please download the dataset from:")
    println("   https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions")
    println("   Save it as: $data_path")
end

println("\n" * "="^70)
println("✅ Verification complete!")
println("="^70)
