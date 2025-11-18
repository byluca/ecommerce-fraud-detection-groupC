
# ============================================================================
#              SUPPORT FUNCTIONS FROM UNIT 5
# ============================================================================

using Random
using Statistics
using Flux
using Flux.Losses

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    @assert(all([in(value, classes) for value in feature]))
    numClasses = length(classes)
    @assert(numClasses > 1)
    
    if (numClasses == 2)
        return reshape(feature .== classes[1], :, 1)
    else
        oneHot = falses(length(feature), numClasses)
        for i in 1:numClasses
            oneHot[:, i] .= (feature .== classes[i])
        end
        return oneHot
    end
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)


calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) = 
    (minimum(dataset, dims=1), maximum(dataset, dims=1))

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, 
                          normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
    dataset[:, vec(minValues .== maxValues)] .= 0
    return dataset
end

normalizeMinMax!(dataset::AbstractArray{<:Real,2}) = 
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))

normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) = 
    normalizeMinMax!(copy(dataset), normalizationParameters)

normalizeMinMax(dataset::AbstractArray{<:Real,2}) = 
    normalizeMinMax!(copy(dataset))


function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2)
    @assert(numOutputs != 2)
    
    if numOutputs == 1
        return outputs .>= threshold
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        outputs_bool = falses(size(outputs))
        outputs_bool[indicesMaxEachInstance] .= true
        @assert(all(sum(outputs_bool, dims=2) .== 1))
        return outputs_bool
    end
end


accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = 
    mean(outputs .== targets)

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert size(outputs) == size(targets)
    if size(targets, 2) == 1
        return accuracy(view(outputs, :, 1), view(targets, :, 1))
    else
        return mean(all(targets .== outputs, dims=2))
    end
end

accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = 
    accuracy(outputs .>= threshold, targets)

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert size(outputs) == size(targets)
    if size(targets, 2) == 1
        return accuracy(view(outputs, :, 1), view(targets, :, 1), threshold=threshold)
    else
        return accuracy(classifyOutputs(outputs), targets)
    end
end


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    
    total_patterns = TP + TN + FP + FN
    
    acc = total_patterns > 0 ? (TP + TN) / total_patterns : 0.0
    error_rate = 1.0 - acc
    
    sensitivity = (TP + FN) > 0 ? TP / (TP + FN) : 0.0
    specificity = (TN + FP) > 0 ? TN / (TN + FP) : 0.0
    ppv = (TP + FP) > 0 ? TP / (TP + FP) : 0.0
    npv = (TN + FN) > 0 ? TN / (TN + FN) : 0.0
    
    fscore = 0.0
    if (sensitivity + ppv) > 0
        fscore = 2 * ppv * sensitivity / (sensitivity + ppv)
    end
    
    conf_matrix = [TN FP; FN TP]
    
    return acc, error_rate, sensitivity, specificity, ppv, npv, fscore, conf_matrix
end

confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) =
    confusionMatrix(outputs .>= threshold, targets)

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    num_patterns, num_classes = size(targets)
    @assert size(outputs) == size(targets)
    
    if num_classes == 1
        return confusionMatrix(vec(outputs), vec(targets))
    end
    @assert num_classes != 2
    
    sensitivities = zeros(Float64, num_classes)
    specificities = zeros(Float64, num_classes)
    ppvs = zeros(Float64, num_classes)
    npvs = zeros(Float64, num_classes)
    fscores = zeros(Float64, num_classes)
    
    for i in 1:num_classes
        _, _, sens, spec, ppv, npv, fscore, _ = confusionMatrix(outputs[:, i], targets[:, i])
        sensitivities[i] = sens
        specificities[i] = spec
        ppvs[i] = ppv
        npvs[i] = npv
        fscores[i] = fscore
    end
    
    if weighted
        weights = vec(sum(targets, dims=1)) ./ num_patterns
        sensitivity = sum(sensitivities .* weights)
        specificity = sum(specificities .* weights)
        ppv = sum(ppvs .* weights)
        npv = sum(npvs .* weights)
        fscore = sum(fscores .* weights)
    else
        sensitivity = mean(sensitivities)
        specificity = mean(specificities)
        ppv = mean(ppvs)
        npv = mean(npvs)
        fscore = mean(fscores)
    end
    
    acc = accuracy(outputs, targets)
    error_rate = 1.0 - acc
    
    conf_matrix = zeros(Int, num_classes, num_classes)
    for i in 1:num_patterns
        true_class = findfirst(targets[i, :])
        pred_class = findfirst(outputs[i, :])
        if !isnothing(true_class) && !isnothing(pred_class)
            conf_matrix[true_class, pred_class] += 1
        end
    end
    
    return acc, error_rate, sensitivity, specificity, ppv, npv, fscore, conf_matrix
end

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; 
                threshold::Real=0.5, weighted::Bool=true) =
    confusionMatrix(classifyOutputs(outputs, threshold=threshold), targets; weighted=weighted)

# New version: confusionMatrix for string/categorical arrays with classes vector
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, 
                        classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Convert outputs and targets to one-hot encoded format
    outputs_encoded = oneHotEncoding(outputs, classes)
    targets_encoded = oneHotEncoding(targets, classes)
    
    # Call the existing confusionMatrix for Bool arrays
    return confusionMatrix(outputs_encoded, targets_encoded; weighted=weighted)
end


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                      transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()
    numInputsLayer = numInputs
    
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer]
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]))
        numInputsLayer = numNeurons
    end
    
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    
    return ann
end


function holdOut(N::Int; Pval::Real=0.0, Ptest::Real=0.0)
    @assert N > 0 "N must be greater than 0"
    @assert 0.0 ≤ Pval ≤ 1.0 "Pval must be between 0 and 1"
    @assert 0.0 ≤ Ptest ≤ 1.0 "Ptest must be between 0 and 1"
    @assert Pval + Ptest ≤ 1.0 "The sum of Pval + Ptest cannot exceed 1.0"
    
    indices = randperm(N)
    
    numTest = Int(round(N * Ptest))
    numVal  = Int(round(N * Pval))
    numTrain = N - numTest - numVal
    
    trainIndices = indices[1:numTrain]
    valIndices   = (numVal > 0)  ? indices[numTrain+1:numTrain+numVal] : Int[]
    testIndices  = (numTest > 0) ? indices[numTrain+numVal+1:end] : Int[]
    
    return (trainIndices, valIndices, testIndices)
end


function trainClassANN(topology::AbstractArray{<:Int,1},
                      trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                      validationDataset::Union{Nothing, Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}}=nothing,
                      transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
                      maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
                      maxEpochsVal::Int=20)
    
    (trainInputs, trainTargets) = trainingDataset
    @assert(size(trainInputs,1) == size(trainTargets,1))
    
    ann = buildClassANN(size(trainInputs,2), topology, size(trainTargets,2); 
                        transferFunctions=transferFunctions)
    
    loss(model,x,y) = (size(y,1) == 1) ? 
        Losses.binarycrossentropy(model(x),y) : 
        Losses.crossentropy(model(x),y)
    
    trainingLosses = Float32[]
    validationLosses = Float32[]
    
    numEpoch = 0
    trainingLoss = loss(ann, trainInputs', trainTargets')
    push!(trainingLosses, trainingLoss)
    
    bestAnn = ann
    numEpochsValidation = 0
    
    if validationDataset !== nothing
        (valInputs, valTargets) = validationDataset
        @assert(size(valInputs,1) == size(valTargets,1))
        @assert(size(trainInputs,2) == size(valInputs,2))
        @assert(size(trainTargets,2) == size(valTargets,2))
        
        validationLoss = loss(ann, valInputs', valTargets')
        push!(validationLosses, validationLoss)
        
        bestValLoss = validationLoss
        bestAnn = deepcopy(ann)
    end
    
    opt_state = Flux.setup(Adam(learningRate), ann)
    
    while numEpoch < maxEpochs && 
          trainingLoss > minLoss && 
          numEpochsValidation < maxEpochsVal
        
        Flux.train!(loss, ann, [(trainInputs', trainTargets')], opt_state)
        numEpoch += 1
        
        trainingLoss = loss(ann, trainInputs', trainTargets')
        push!(trainingLosses, trainingLoss)
        
        if validationDataset !== nothing
            validationLoss = loss(ann, valInputs', valTargets')
            push!(validationLosses, validationLoss)
            
            if validationLoss < bestValLoss
                bestValLoss = validationLoss
                bestAnn = deepcopy(ann)
                numEpochsValidation = 0
            else
                numEpochsValidation += 1
            end
        end
    end
    
    if validationDataset !== nothing
        return (bestAnn, trainingLosses, validationLosses)
    else
        return (ann, trainingLosses)
    end
end


function crossvalidation(N::Int64, k::Int64)
    @assert k > 0 "k must be greater than 0"
    @assert N >= k "N must be greater than or equal to k"
    
    base_vector = collect(1:k)
    repeated_vector = repeat(base_vector, Int(ceil(N / k)))
    indices = repeated_vector[1:N]
    shuffle!(indices)
    
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = zeros(Int, length(targets))
    indices[targets] .= crossvalidation(sum(targets), k)
    indices[.!targets] .= crossvalidation(sum(.!targets), k)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    @assert size(targets, 2) != 2 "Use the Bool,1 version for binary classification"
    
    numInstances = size(targets, 1)
    numClasses = size(targets, 2)
    
    @assert all(sum(targets, dims=2) .== 1) "Each instance must belong to exactly one class"
    
    for classIdx in 1:numClasses
        numSamplesInClass = sum(targets[:, classIdx])
        if numSamplesInClass < k
            error("Class $classIdx has only $numSamplesInClass samples, but at least $k are required for k-fold cross-validation.")
        end
    end
    
    indices = zeros(Int, numInstances)
    
    for classIdx in 1:numClasses
        classMask = targets[:, classIdx]
        classIndices = crossvalidation(sum(classMask), k)
        indices[classMask] .= classIndices
    end
    
    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    encoded_targets = oneHotEncoding(targets)
    return crossvalidation(encoded_targets, k)
end


function ANNCrossValidation(topology::AbstractArray{<:Int,1},
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1};
        numExecutions::Int=50,
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        validationRatio::Real=0, maxEpochsVal::Int=20)
    
    inputs, targets = dataset
    classes = unique(targets)
    
    encoded_targets = oneHotEncoding(targets, classes)
    
    isBinary = (size(encoded_targets, 2) == 1)
    confMatrixSize = isBinary ? 2 : size(encoded_targets, 2)
    
    numFolds = maximum(crossValidationIndices)
    
    if isBinary
        numPositive = sum(encoded_targets)
        numNegative = length(encoded_targets) - numPositive
        if numPositive < numFolds || numNegative < numFolds
            @warn "Imbalanced dataset: positives=$numPositive, negatives=$numNegative. There may be issues with $numFolds folds."
        end
    else
        for i in 1:size(encoded_targets, 2)
            classCount = sum(encoded_targets[:, i])
            if classCount < numFolds
                error("Class '$(classes[i])' has only $classCount samples, insufficient for $numFolds folds")
            end
        end
    end
    
    testAccuracies = Float64[]
    testErrorRates = Float64[]
    testSensitivities = Float64[]
    testSpecificities = Float64[]
    testPPVs = Float64[]
    testNPVs = Float64[]
    testF1Scores = Float64[]
    
    globalConfusionMatrix = zeros(Float64, confMatrixSize, confMatrixSize)
    
    for fold in 1:numFolds
        println("\nFold ", fold, "/", numFolds)
        
        testMask = crossValidationIndices .== fold
        trainMask = .!testMask
        
        trainInputs = inputs[trainMask, :]
        trainTargets = encoded_targets[trainMask, :]
        testInputs = inputs[testMask, :]
        testTargets = encoded_targets[testMask, :]
        
        normParams = calculateMinMaxNormalizationParameters(trainInputs)
        trainInputsNorm = normalizeMinMax(trainInputs, normParams)
        testInputsNorm = normalizeMinMax(testInputs, normParams)
        
        execAccuracies = Float64[]
        execErrorRates = Float64[]
        execSensitivities = Float64[]
        execSpecificities = Float64[]
        execPPVs = Float64[]
        execNPVs = Float64[]
        execF1Scores = Float64[]
        
        confusionMatrices = zeros(Float64, confMatrixSize, confMatrixSize, numExecutions)
        
        for exec in 1:numExecutions
            print("  Execution ", exec, "/", numExecutions, "\r")
            flush(stdout)
            
            if validationRatio > 0
                trainSize = size(trainInputsNorm, 1)
                adjustedValidationRatio = validationRatio * numFolds / (numFolds - 1)
                adjustedValidationRatio = min(adjustedValidationRatio, 0.99)
                
                trainIndicesExec, valIndices, _ = holdOut(trainSize; Pval=adjustedValidationRatio)
                
                execTrainInputs = trainInputsNorm[trainIndicesExec, :]
                execTrainTargets = trainTargets[trainIndicesExec, :]
                valInputs = trainInputsNorm[valIndices, :]
                valTargets = trainTargets[valIndices, :]
                
                result = trainClassANN(topology, 
                    (execTrainInputs, execTrainTargets);
                    validationDataset=(valInputs, valTargets),
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs,
                    minLoss=minLoss,
                    learningRate=learningRate,
                    maxEpochsVal=maxEpochsVal)
                
                ann = result[1]
            else
                result = trainClassANN(topology, 
                    (trainInputsNorm, trainTargets);
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs,
                    minLoss=minLoss,
                    learningRate=learningRate)
                
                ann = result[1]
            end
            
            testOutputs = ann(testInputsNorm')'
            
            acc, err, sens, spec, ppv, npv, f1, cm = confusionMatrix(
                testOutputs, testTargets; weighted=true)
            
            push!(execAccuracies, acc)
            push!(execErrorRates, err)
            push!(execSensitivities, sens)
            push!(execSpecificities, spec)
            push!(execPPVs, ppv)
            push!(execNPVs, npv)
            push!(execF1Scores, f1)
            confusionMatrices[:, :, exec] = cm
        end
        
        println()
        
        foldAccuracy = mean(execAccuracies)
        foldErrorRate = mean(execErrorRates)
        foldSensitivity = mean(execSensitivities)
        foldSpecificity = mean(execSpecificities)
        foldPPV = mean(execPPVs)
        foldNPV = mean(execNPVs)
        foldF1 = mean(execF1Scores)
        
        push!(testAccuracies, foldAccuracy)
        push!(testErrorRates, foldErrorRate)
        push!(testSensitivities, foldSensitivity)
        push!(testSpecificities, foldSpecificity)
        push!(testPPVs, foldPPV)
        push!(testNPVs, foldNPV)
        push!(testF1Scores, foldF1)
        
        meanConfMatrix = dropdims(mean(confusionMatrices, dims=3), dims=3)
        globalConfusionMatrix .+= meanConfMatrix
    end
    
    return (mean(testAccuracies), std(testAccuracies)), 
           (mean(testErrorRates), std(testErrorRates)),
           (mean(testSensitivities), std(testSensitivities)),
           (mean(testSpecificities), std(testSpecificities)),
           (mean(testPPVs), std(testPPVs)),
           (mean(testNPVs), std(testNPVs)),
           (mean(testF1Scores), std(testF1Scores)),
           globalConfusionMatrix
end
