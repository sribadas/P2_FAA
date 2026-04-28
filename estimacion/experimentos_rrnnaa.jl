# ==============================================================================
# EXPERIMENTOS RRNNAA - Clasificacion de Niveles de Obesidad
# ==============================================================================

include("p1.jl")

using DataFrames
using Random
using Statistics
using CSV
using StatsBase

Random.seed!(42)

const BASE_DIR = @__DIR__
const DATASET_PATH = joinpath(BASE_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")
const OUTPUT_DIR = joinpath(BASE_DIR, "salidas_rrnnaa")
const DATA_OUTPUT_DIR = joinpath(OUTPUT_DIR, "datos")
const RESULTS_PATH = joinpath(DATA_OUTPUT_DIR, "resultados_rrnnaa.csv")
const CONFUSION_PATH = joinpath(DATA_OUTPUT_DIR, "matrices_confusion_rrnnaa.csv")


function build_onehot_matrix(col_data::AbstractVector{<:AbstractString})
    values = sort(unique(col_data))
    return Float32.(hcat([Float32.(col_data .== value) for value in values]...))
end


function run_ann_cross_validation(
    inputs::AbstractMatrix{<:Real},
    targets_raw::AbstractVector{<:AbstractString},
    cross_val_idx::AbstractVector{<:Integer},
    classes_ordered::AbstractVector{<:AbstractString};
    topology::AbstractVector{<:Int},
    transfer_functions::AbstractVector{<:Function},
    num_executions::Int,
    max_epochs::Int,
    learning_rate::Real,
    validation_ratio::Real,
    max_epochs_val::Int,
    min_loss::Real,
)
    targets = oneHotEncoding(targets_raw, classes_ordered)
    num_folds = maximum(cross_val_idx)

    test_accuracy = Array{Float64,1}(undef, num_folds)
    train_accuracy = Array{Float64,1}(undef, num_folds)
    test_error_rate = Array{Float64,1}(undef, num_folds)
    test_recall = Array{Float64,1}(undef, num_folds)
    test_specificity = Array{Float64,1}(undef, num_folds)
    test_precision = Array{Float64,1}(undef, num_folds)
    test_npv = Array{Float64,1}(undef, num_folds)
    test_f1 = Array{Float64,1}(undef, num_folds)
    test_confusion_matrix = zeros(Float64, length(classes_ordered), length(classes_ordered))

    for num_fold in 1:num_folds
        training_inputs_raw = inputs[cross_val_idx .!= num_fold, :]
        test_inputs_raw = inputs[cross_val_idx .== num_fold, :]
        training_targets = targets[cross_val_idx .!= num_fold, :]
        test_targets = targets[cross_val_idx .== num_fold, :]

        test_accuracy_each_repetition = Array{Float64,1}(undef, num_executions)
        train_accuracy_each_repetition = Array{Float64,1}(undef, num_executions)
        test_error_rate_each_repetition = Array{Float64,1}(undef, num_executions)
        test_recall_each_repetition = Array{Float64,1}(undef, num_executions)
        test_specificity_each_repetition = Array{Float64,1}(undef, num_executions)
        test_precision_each_repetition = Array{Float64,1}(undef, num_executions)
        test_npv_each_repetition = Array{Float64,1}(undef, num_executions)
        test_f1_each_repetition = Array{Float64,1}(undef, num_executions)
        test_confusion_matrix_each_repetition = Array{Float64,3}(undef, length(classes_ordered), length(classes_ordered), num_executions)

        for num_training in 1:num_executions
            if validation_ratio > 0
                validation_fraction = validation_ratio
                (training_indices, validation_indices) = holdOut(size(training_inputs_raw, 1), validation_fraction)

                normalization_parameters = calculateMinMaxNormalizationParameters(training_inputs_raw[training_indices, :])
                current_training_inputs = normalizeMinMax(training_inputs_raw[training_indices, :], normalization_parameters)
                current_validation_inputs = normalizeMinMax(training_inputs_raw[validation_indices, :], normalization_parameters)
                current_test_inputs = normalizeMinMax(test_inputs_raw, normalization_parameters)

                ann, = trainClassANN(
                    topology,
                    (current_training_inputs, training_targets[training_indices, :]);
                    validationDataset = (current_validation_inputs, training_targets[validation_indices, :]),
                    testDataset = (current_test_inputs, test_targets),
                    transferFunctions = transfer_functions,
                    maxEpochs = max_epochs,
                    minLoss = min_loss,
                    learningRate = learning_rate,
                    maxEpochsVal = max_epochs_val,
                )

                train_outputs = collect(ann(Float32.(current_training_inputs'))')
                train_accuracy_each_repetition[num_training] = accuracy(
                    train_outputs,
                    training_targets[training_indices, :],
                )
            else
                normalization_parameters = calculateMinMaxNormalizationParameters(training_inputs_raw)
                current_training_inputs = normalizeMinMax(training_inputs_raw, normalization_parameters)
                current_test_inputs = normalizeMinMax(test_inputs_raw, normalization_parameters)

                ann, = trainClassANN(
                    topology,
                    (current_training_inputs, training_targets);
                    testDataset = (current_test_inputs, test_targets),
                    transferFunctions = transfer_functions,
                    maxEpochs = max_epochs,
                    minLoss = min_loss,
                    learningRate = learning_rate,
                )

                train_outputs = collect(ann(Float32.(current_training_inputs'))')
                train_accuracy_each_repetition[num_training] = accuracy(
                    train_outputs,
                    training_targets,
                )
            end

            outputs = collect(ann(Float32.(current_test_inputs'))')
            (
                test_accuracy_each_repetition[num_training],
                test_error_rate_each_repetition[num_training],
                test_recall_each_repetition[num_training],
                test_specificity_each_repetition[num_training],
                test_precision_each_repetition[num_training],
                test_npv_each_repetition[num_training],
                test_f1_each_repetition[num_training],
                test_confusion_matrix_each_repetition[:, :, num_training],
            ) = confusionMatrix(outputs, test_targets; weighted=false)
        end

        test_accuracy[num_fold] = mean(test_accuracy_each_repetition)
        train_accuracy[num_fold] = mean(train_accuracy_each_repetition)
        test_error_rate[num_fold] = mean(test_error_rate_each_repetition)
        test_recall[num_fold] = mean(test_recall_each_repetition)
        test_specificity[num_fold] = mean(test_specificity_each_repetition)
        test_precision[num_fold] = mean(test_precision_each_repetition)
        test_npv[num_fold] = mean(test_npv_each_repetition)
        test_f1[num_fold] = mean(test_f1_each_repetition)
        test_confusion_matrix .+= mean(test_confusion_matrix_each_repetition, dims=3)[:, :, 1]
    end

    return (
        (mean(test_accuracy), std(test_accuracy)),
        (mean(train_accuracy), std(train_accuracy)),
        (mean(test_error_rate), std(test_error_rate)),
        (mean(test_recall), std(test_recall)),
        (mean(test_specificity), std(test_specificity)),
        (mean(test_precision), std(test_precision)),
        (mean(test_npv), std(test_npv)),
        (mean(test_f1), std(test_f1)),
        test_confusion_matrix,
    )
end


# ==============================================================================
# 1. CARGA Y PREPROCESADO
# ==============================================================================

println("="^60)
println("Cargando dataset...")
println("="^60)

data = CSV.read(DATASET_PATH, DataFrame)

data = data[sample(1:size(data, 1), size(data, 1), replace=false), :]

target_col = "NObeyesdad"
targets_raw = string.(data[:, target_col])
classes_ordered = sort(unique(targets_raw))

num_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
bin_cols = ["FAVC", "SMOKE", "SCC", "family_history_with_overweight"]
cat_cols = ["Gender", "CAEC", "CALC", "MTRANS"]

num_matrix = Float32.(Matrix(data[:, num_cols]))
bin_matrix = Float32.(hcat([Float32.(string.(data[:, c]) .== "yes") for c in bin_cols]...))

cat_matrices = Matrix{Float32}[]
for c in cat_cols
    push!(cat_matrices, build_onehot_matrix(string.(data[:, c])))
end
cat_matrix = hcat(cat_matrices...)

inputs_raw = hcat(num_matrix, bin_matrix, cat_matrix)

println("Dimensiones inputs (sin normalizar): ", size(inputs_raw))
println("Clases: ", join(classes_ordered, ", "))
println("F1 reportado: macro (promedio no ponderado entre clases)")


# ==============================================================================
# 2. VALIDACION CRUZADA
# ==============================================================================

cross_val_idx = crossvalidation(targets_raw, 10)


# ==============================================================================
# 3. ARQUITECTURAS
# ==============================================================================

architectures = [
    Dict("name"=>"[16] σ",         "topology"=>[16],     "tf"=>fill(σ, 1)),
    Dict("name"=>"[32] σ",         "topology"=>[32],     "tf"=>fill(σ, 1)),
    Dict("name"=>"[64] σ",         "topology"=>[64],     "tf"=>fill(σ, 1)),
    Dict("name"=>"[32] relu",      "topology"=>[32],     "tf"=>fill(relu, 1)),
    Dict("name"=>"[64] relu",      "topology"=>[64],     "tf"=>fill(relu, 1)),
    Dict("name"=>"[32,16] σ",      "topology"=>[32, 16], "tf"=>fill(σ, 2)),
    Dict("name"=>"[64,32] σ",      "topology"=>[64, 32], "tf"=>fill(σ, 2)),
    Dict("name"=>"[64,32] relu",   "topology"=>[64, 32], "tf"=>fill(relu, 2)),
    Dict("name"=>"[128,64] σ",     "topology"=>[128, 64],"tf"=>fill(σ, 2)),
    Dict("name"=>"[64,32] σ/tanh", "topology"=>[64, 32], "tf"=>[σ, tanh_fast]),
]


# ==============================================================================
# 4. EXPERIMENTOS
# ==============================================================================

results = DataFrame(
    Arquitectura  = String[],
    Topologia     = String[],
    Acc_media     = Float64[],
    Acc_std       = Float64[],
    TrainAcc_media = Float64[],
    TrainAcc_std   = Float64[],
    F1_media      = Float64[],
    F1_std        = Float64[],
    ErrorRate_med = Float64[],
    ErrorRate_std = Float64[],
)

conf_matrices = Dict{String, Matrix{Float64}}()

for arch in architectures
    name = arch["name"]
    topology = arch["topology"]
    tf = arch["tf"]

    println("\n>>> Arquitectura: $name")

    results_tuple = run_ann_cross_validation(
        inputs_raw,
        targets_raw,
        cross_val_idx,
        classes_ordered;
        topology = topology,
        transfer_functions = tf,
        num_executions = 10,
        max_epochs = 500,
        learning_rate = 0.01,
        validation_ratio = 0.1,
        max_epochs_val = 20,
        min_loss = 0.0,
    )

    (acc_m, acc_s) = results_tuple[1]
    (train_acc_m, train_acc_s) = results_tuple[2]
    (err_m, err_s) = results_tuple[3]
    (f1_m, f1_s) = results_tuple[8]
    conf_mat = results_tuple[9]

    println("  Train Acc: $(round(train_acc_m * 100, digits=2)) ± $(round(train_acc_s * 100, digits=2)) %")
    println("  Accuracy : $(round(acc_m * 100, digits=2)) ± $(round(acc_s * 100, digits=2)) %")
    println("  F1-score : $(round(f1_m * 100, digits=2)) ± $(round(f1_s * 100, digits=2)) %")

    push!(results, (
        name,
        string(topology),
        round(acc_m, digits=4),
        round(acc_s, digits=4),
        round(train_acc_m, digits=4),
        round(train_acc_s, digits=4),
        round(f1_m, digits=4),
        round(f1_s, digits=4),
        round(err_m, digits=4),
        round(err_s, digits=4),
    ))

    conf_matrices[name] = conf_mat
end


# ==============================================================================
# 5. GUARDAR RESULTADOS
# ==============================================================================

mkpath(DATA_OUTPUT_DIR)
CSV.write(RESULTS_PATH, results)

open(CONFUSION_PATH, "w") do f
    println(f, "# Matrices de confusion medias acumuladas por fold y ejecucion")
    println(f, "# No representan una matriz global bruta del conjunto completo")
    println(f, "# Clases: " * join(classes_ordered, ","))
    println(f, "")
    for arch in architectures
        name = arch["name"]
        mat = conf_matrices[name]
        println(f, "# Arquitectura: $name")
        for (i, row) in enumerate(eachrow(mat))
            println(f, classes_ordered[i] * "," * join(string.(Int.(round.(row))), ","))
        end
        println(f, "")
    end
end

println("\n✓ Resultados guardados")


# ==============================================================================
# 6. RESUMEN
# ==============================================================================

println("\n", "="^60)
println("RESUMEN FINAL")
println("="^60)

for row in eachrow(results)
    println(
        rpad(row.Arquitectura, 20), " | ",
        "Train Acc: $(round(row.TrainAcc_media * 100, digits=2)) ± $(round(row.TrainAcc_std * 100, digits=2)) | ",
        "Acc: $(round(row.Acc_media * 100, digits=2)) ± $(round(row.Acc_std * 100, digits=2)) | ",
        "F1: $(round(row.F1_media * 100, digits=2)) ± $(round(row.F1_std * 100, digits=2))"
    )
end

println("="^60)
