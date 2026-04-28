# ==============================================================================
# EXPERIMENTOS DoME - Clasificacion de Niveles de Obesidad
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
const OUTPUT_DIR = joinpath(BASE_DIR, "salidas_dome")
const DATA_OUTPUT_DIR = joinpath(OUTPUT_DIR, "datos")
const RESULTS_PATH = joinpath(DATA_OUTPUT_DIR, "resultados_dome.csv")
const CONFUSION_PATH = joinpath(DATA_OUTPUT_DIR, "matrices_confusion_dome.csv")


function build_onehot_matrix(col_data::AbstractVector{<:AbstractString})
    values = sort(unique(col_data))
    return Float32.(hcat([Float32.(col_data .== value) for value in values]...))
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
# 3. CONFIGURACIONES DoME
# ==============================================================================
#
# DoME es un algoritmo simbolico de regresion que construye un modelo en forma
# de arbol de expresion. El parametro clave es maximumNodes, que controla la
# complejidad maxima del modelo. Para clasificacion multiclase (7 clases) se
# entrena un modelo binario por clase (estrategia one-vs-rest).
#
# Segun los resultados obtenidos hasta ahora, la zona mas prometedora para
# DoME esta en el tramo alto de complejidad: entre 30 y 100 nodos el macro-F1
# mejora de forma sostenida y se concentran las mejores configuraciones.
# Para cumplir la consigna, se prueban 8 valores distintos dentro de ese
# intervalo de mejores resultados.

best_nodes_interval = 30:10:100

dome_configs = [
    Dict(
        "name" => "DoME n=$(nodes)",
        "maximumNodes" => nodes,
    ) for nodes in best_nodes_interval
]


function fit_dome_binary(training_inputs::AbstractMatrix{<:Real}, training_targets::AbstractVector{Bool}, maximum_nodes::Int)
    model, _, _, _ = dome(
        Float64.(training_inputs),
        training_targets;
        maximumNodes = maximum_nodes,
    )
    return model
end


function predict_dome_binary(model, inputs::AbstractMatrix{<:Real})
    outputs = evaluateTree(model, Float64.(inputs))
    if isa(outputs, Real)
        outputs = fill(outputs, size(inputs, 1))
    end
    return outputs
end


function fit_dome_multiclass(
    training_inputs::AbstractMatrix{<:Real},
    training_targets::AbstractVector{<:AbstractString},
    classes_ordered::AbstractVector{<:AbstractString},
    maximum_nodes::Int,
)
    onehot_targets = oneHotEncoding(training_targets, classes_ordered)
    models = [
        fit_dome_binary(training_inputs, vec(onehot_targets[:, num_class]), maximum_nodes)
        for num_class in axes(onehot_targets, 2)
    ]
    return models
end


function predict_dome_multiclass(
    models::AbstractVector,
    inputs::AbstractMatrix{<:Real},
    classes_ordered::AbstractVector{<:AbstractString},
)
    raw_outputs = hcat([predict_dome_binary(model, inputs) for model in models]...)
    outputs_bool = classifyOutputs(raw_outputs; threshold=0)
    predicted = Array{eltype(classes_ordered),1}(undef, size(inputs, 1))

    @assert(all(sum(outputs_bool, dims=2) .== 1))
    for num_class in eachindex(classes_ordered)
        predicted[outputs_bool[:, num_class]] .= classes_ordered[num_class]
    end

    return predicted
end


# ==============================================================================
# 4. EXPERIMENTO DoME CON NORMALIZACION POR FOLD
# ==============================================================================

function run_dome_cross_validation(
    inputs_raw::AbstractMatrix{<:Real},
    targets_raw::AbstractVector{<:AbstractString},
    cross_val_idx::AbstractVector{<:Integer},
    classes_ordered::AbstractVector{<:AbstractString},
    maximum_nodes::Int,
)
    num_folds = maximum(cross_val_idx)

    test_accuracy   = Array{Float64,1}(undef, num_folds)
    train_accuracy  = Array{Float64,1}(undef, num_folds)
    test_error_rate = Array{Float64,1}(undef, num_folds)
    test_recall     = Array{Float64,1}(undef, num_folds)
    test_specificity= Array{Float64,1}(undef, num_folds)
    test_precision  = Array{Float64,1}(undef, num_folds)
    test_npv        = Array{Float64,1}(undef, num_folds)
    test_f1         = Array{Float64,1}(undef, num_folds)
    test_conf_matrix = zeros(Int, length(classes_ordered), length(classes_ordered))

    for num_fold in 1:num_folds
        train_inputs_raw = inputs_raw[cross_val_idx .!= num_fold, :]
        test_inputs_raw  = inputs_raw[cross_val_idx .== num_fold, :]
        train_targets    = targets_raw[cross_val_idx .!= num_fold]
        test_targets     = targets_raw[cross_val_idx .== num_fold]

        norm_params  = calculateMinMaxNormalizationParameters(train_inputs_raw)
        train_inputs = normalizeMinMax(train_inputs_raw, norm_params)
        test_inputs  = normalizeMinMax(test_inputs_raw,  norm_params)

        dome_models = fit_dome_multiclass(
            train_inputs,
            train_targets,
            classes_ordered,
            maximum_nodes,
        )

        test_outputs = predict_dome_multiclass(
            dome_models,
            test_inputs,
            classes_ordered,
        )

        train_outputs = predict_dome_multiclass(
            dome_models,
            train_inputs,
            classes_ordered,
        )

        train_accuracy[num_fold] = mean(train_outputs .== train_targets)

        (
            test_accuracy[num_fold],
            test_error_rate[num_fold],
            test_recall[num_fold],
            test_specificity[num_fold],
            test_precision[num_fold],
            test_npv[num_fold],
            test_f1[num_fold],
            conf_fold,
        ) = confusionMatrix(test_outputs, test_targets, classes_ordered; weighted=false)

        test_conf_matrix .+= conf_fold
    end

    return (
        (mean(test_accuracy),   std(test_accuracy)),
        (mean(train_accuracy),  std(train_accuracy)),
        (mean(test_error_rate), std(test_error_rate)),
        (mean(test_recall),     std(test_recall)),
        (mean(test_specificity),std(test_specificity)),
        (mean(test_precision),  std(test_precision)),
        (mean(test_npv),        std(test_npv)),
        (mean(test_f1),         std(test_f1)),
        test_conf_matrix,
    )
end


# ==============================================================================
# 5. EJECUCION
# ==============================================================================

println("\n", "="^60)
println("Iniciando experimentos DoME...")
println("="^60)

results = DataFrame(
    Nombre         = String[],
    MaxNodes       = Int[],
    Acc_media      = Float64[],
    Acc_std        = Float64[],
    TrainAcc_media = Float64[],
    TrainAcc_std   = Float64[],
    F1_media       = Float64[],
    F1_std         = Float64[],
    ErrorRate_med  = Float64[],
    ErrorRate_std  = Float64[],
)

conf_matrices = Dict{String, Matrix{Int}}()

for cfg in dome_configs
    name  = cfg["name"]
    nodes = cfg["maximumNodes"]
    println("\n>>> Configuracion: $name  (maximumNodes=$nodes)")

    result = run_dome_cross_validation(
        inputs_raw,
        targets_raw,
        cross_val_idx,
        classes_ordered,
        nodes,
    )

    (acc_m,  acc_s)  = result[1]
    (tacc_m, tacc_s) = result[2]
    (err_m,  err_s)  = result[3]
    (f1_m,   f1_s)   = result[8]
    conf_mat         = result[9]

    println("  Train Acc: $(round(tacc_m*100, digits=2)) ± $(round(tacc_s*100, digits=2)) %")
    println("  Accuracy : $(round(acc_m *100, digits=2)) ± $(round(acc_s *100, digits=2)) %")
    println("  F1-score : $(round(f1_m  *100, digits=2)) ± $(round(f1_s  *100, digits=2)) %")

    push!(results, (
        name, nodes,
        round(acc_m,  digits=4), round(acc_s,  digits=4),
        round(tacc_m, digits=4), round(tacc_s, digits=4),
        round(f1_m,   digits=4), round(f1_s,   digits=4),
        round(err_m,  digits=4), round(err_s,  digits=4),
    ))

    conf_matrices[name] = conf_mat
end

best_idx = argmax(results.F1_media)
best_row = results[best_idx, :]


# ==============================================================================
# 6. GUARDAR RESULTADOS
# ==============================================================================

mkpath(DATA_OUTPUT_DIR)
CSV.write(RESULTS_PATH, results)

open(CONFUSION_PATH, "w") do f
    println(f, "# Matrices de confusion acumuladas (suma) sobre 10 folds")
    println(f, "# F1 reportado: macro (no ponderado)")
    println(f, "# Clases: " * join(classes_ordered, ","))
    println(f, "")
    for cfg in dome_configs
        name = cfg["name"]
        mat  = conf_matrices[name]
        println(f, "# Configuracion: $name")
        for (i, row) in enumerate(eachrow(mat))
            println(f, classes_ordered[i] * "," * join(string.(Int.(round.(row))), ","))
        end
        println(f, "")
    end
end

println("\n✓ Resultados guardados en:")
println("  - $RESULTS_PATH")
println("  - $CONFUSION_PATH")
println("Mejor configuracion DoME por macro-F1: ", best_row.Nombre,
    " | Acc=", round(best_row.Acc_media*100, digits=2),
    "% | F1=", round(best_row.F1_media*100,  digits=2), "%")


# ==============================================================================
# 7. RESUMEN
# ==============================================================================

println("\n", "="^60)
println("RESUMEN FINAL - DoME")
println("="^60)
println(rpad("Configuracion", 14), " | ", rpad("MaxNodes", 8),
        " | ", rpad("TrainAcc(%)", 18),
        " | ", rpad("Acc(%)", 18),
        " | ", rpad("F1(%)", 18))
println("-"^85)
for row in eachrow(results)
    println(
        rpad(row.Nombre, 14), " | ",
        rpad(string(row.MaxNodes), 8), " | ",
        rpad("$(round(row.TrainAcc_media*100, digits=2)) ± $(round(row.TrainAcc_std*100, digits=2))", 18), " | ",
        rpad("$(round(row.Acc_media*100,      digits=2)) ± $(round(row.Acc_std*100,      digits=2))", 18), " | ",
        rpad("$(round(row.F1_media*100,       digits=2)) ± $(round(row.F1_std*100,       digits=2))", 18),
    )
end
println("-"^85)
println("Mejor: ", best_row.Nombre,
    " | TrainAcc=$(round(best_row.TrainAcc_media*100, digits=2))",
    " | Acc=$(round(best_row.Acc_media*100,           digits=2))",
    " | F1=$(round(best_row.F1_media*100,             digits=2))")
println("="^85)
