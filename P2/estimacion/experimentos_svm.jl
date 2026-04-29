# ==============================================================================
# EXPERIMENTOS SVM - Clasificacion de Niveles de Obesidad
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
const OUTPUT_DIR = joinpath(BASE_DIR, "salidas_svm")
const DATA_OUTPUT_DIR = joinpath(OUTPUT_DIR, "datos")
const RESULTS_PATH = joinpath(DATA_OUTPUT_DIR, "resultados_svm.csv")
const CONFUSION_PATH = joinpath(DATA_OUTPUT_DIR, "matrices_confusion_svm.csv")


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
# 3. CONFIGURACIONES SVM
# ==============================================================================
#
# Justificacion de kernels para la memoria:
# - linear : baseline interpretable y rapido
# - rbf    : captura fronteras no lineales; gamma es un hiperparametro clave
# - poly   : permite modelar relaciones de orden superior
# - sigmoid: se incluye por completitud, aunque su rendimiento suele ser inferior
#
# Parametros probados:
# - kernel : linear, rbf, poly, sigmoid
# - C      : parametro de regularizacion
# - gamma  : -1.0 para el ajuste automatico del wrapper actual
#            (equivalente a 1 / n_features en esta practica), o valor fijo
# - degree : solo para kernel poly

svm_configs = [
    # ---- Kernel lineal ----
    Dict("name"=>"Linear C=0.1",     "kernel"=>"linear",  "C"=>0.1,   "gamma"=>-1.0, "degree"=>-1, "coef0"=>-1.0),
    Dict("name"=>"Linear C=1",       "kernel"=>"linear",  "C"=>1.0,   "gamma"=>-1.0, "degree"=>-1, "coef0"=>-1.0),
    Dict("name"=>"Linear C=10",      "kernel"=>"linear",  "C"=>10.0,  "gamma"=>-1.0, "degree"=>-1, "coef0"=>-1.0),

    # ---- Kernel RBF ----
    Dict("name"=>"RBF C=1 auto",     "kernel"=>"rbf",     "C"=>1.0,   "gamma"=>-1.0, "degree"=>-1, "coef0"=>-1.0),
    Dict("name"=>"RBF C=10 auto",    "kernel"=>"rbf",     "C"=>10.0,  "gamma"=>-1.0, "degree"=>-1, "coef0"=>-1.0),
    Dict("name"=>"RBF C=100 auto",   "kernel"=>"rbf",     "C"=>100.0, "gamma"=>-1.0, "degree"=>-1, "coef0"=>-1.0),
    Dict("name"=>"RBF C=10 γ=0.01",  "kernel"=>"rbf",     "C"=>10.0,  "gamma"=>0.01, "degree"=>-1, "coef0"=>-1.0),
    Dict("name"=>"RBF C=10 γ=0.1",   "kernel"=>"rbf",     "C"=>10.0,  "gamma"=>0.1,  "degree"=>-1, "coef0"=>-1.0),
    Dict("name"=>"RBF C=10 γ=1",     "kernel"=>"rbf",     "C"=>10.0,  "gamma"=>1.0,  "degree"=>-1, "coef0"=>-1.0),

    # ---- Kernel polinomico ----
    Dict("name"=>"Poly d=2 C=1",     "kernel"=>"poly",    "C"=>1.0,   "gamma"=>-1.0, "degree"=>2,  "coef0"=>0.0),
    Dict("name"=>"Poly d=3 C=1",     "kernel"=>"poly",    "C"=>1.0,   "gamma"=>-1.0, "degree"=>3,  "coef0"=>0.0),
    Dict("name"=>"Poly d=3 C=10",    "kernel"=>"poly",    "C"=>10.0,  "gamma"=>-1.0, "degree"=>3,  "coef0"=>0.0),

    # ---- Kernel sigmoide ----
    Dict("name"=>"Sigmoid C=1 auto", "kernel"=>"sigmoid", "C"=>1.0,   "gamma"=>-1.0, "degree"=>-1, "coef0"=>0.0),
    Dict("name"=>"Sigmoid C=10 auto","kernel"=>"sigmoid", "C"=>10.0,  "gamma"=>-1.0, "degree"=>-1, "coef0"=>0.0),
]


# ==============================================================================
# 4. EXPERIMENTO SVM CON NORMALIZACION POR FOLD
# ==============================================================================
#
# A diferencia de las redes neuronales, SVM es determinista en esta practica,
# por lo que una unica ejecucion por fold es suficiente.

function run_svm_cross_validation(
    inputs_raw::AbstractMatrix{<:Real},
    targets_raw::AbstractVector{<:AbstractString},
    cross_val_idx::AbstractVector{<:Integer},
    classes_ordered::AbstractVector{<:AbstractString},
    hyperparams::Dict,
)
    num_folds = maximum(cross_val_idx)

    test_accuracy = Array{Float64,1}(undef, num_folds)
    train_accuracy = Array{Float64,1}(undef, num_folds)
    test_error_rate = Array{Float64,1}(undef, num_folds)
    test_recall = Array{Float64,1}(undef, num_folds)
    test_specificity = Array{Float64,1}(undef, num_folds)
    test_precision = Array{Float64,1}(undef, num_folds)
    test_npv = Array{Float64,1}(undef, num_folds)
    test_f1 = Array{Float64,1}(undef, num_folds)
    test_conf_matrix = zeros(Int, length(classes_ordered), length(classes_ordered))

    for num_fold in 1:num_folds
        train_inputs_raw = inputs_raw[cross_val_idx .!= num_fold, :]
        test_inputs_raw = inputs_raw[cross_val_idx .== num_fold, :]
        train_targets = targets_raw[cross_val_idx .!= num_fold]
        test_targets = targets_raw[cross_val_idx .== num_fold]

        # La normalizacion se ajusta solo con train y se aplica a test.
        norm_params = calculateMinMaxNormalizationParameters(train_inputs_raw)
        train_inputs = normalizeMinMax(train_inputs_raw, norm_params)
        test_inputs = normalizeMinMax(test_inputs_raw, norm_params)

        model = SVMClassifier(
            kernel = hyperparams["kernel"] == "linear"  ? LIBSVM.Kernel.Linear :
                     hyperparams["kernel"] == "rbf"     ? LIBSVM.Kernel.RadialBasis :
                     hyperparams["kernel"] == "poly"    ? LIBSVM.Kernel.Polynomial :
                     hyperparams["kernel"] == "sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing,
            cost = Float64(hyperparams["C"]),
            gamma = Float64(hyperparams["gamma"]),
            degree = Int32(hyperparams["degree"]),
            coef0 = Float64(hyperparams["coef0"]),
        )

        mach = machine(model, MLJ.table(train_inputs), categorical(train_targets))
        MLJ.fit!(mach, verbosity=0)

        train_outputs = MLJ.predict(mach, MLJ.table(train_inputs))
        test_outputs = MLJ.predict(mach, MLJ.table(test_inputs))

        (
            train_accuracy[num_fold],
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = confusionMatrix(train_outputs, train_targets, classes_ordered; weighted=false)

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
        (mean(test_accuracy), std(test_accuracy)),
        (mean(train_accuracy), std(train_accuracy)),
        (mean(test_error_rate), std(test_error_rate)),
        (mean(test_recall), std(test_recall)),
        (mean(test_specificity), std(test_specificity)),
        (mean(test_precision), std(test_precision)),
        (mean(test_npv), std(test_npv)),
        (mean(test_f1), std(test_f1)),
        test_conf_matrix,
    )
end


# ==============================================================================
# 5. EJECUCION
# ==============================================================================

println("\n", "="^60)
println("Iniciando experimentos SVM...")
println("="^60)

results = DataFrame(
    Nombre = String[],
    Kernel = String[],
    C = Float64[],
    Gamma = Float64[],
    Acc_media = Float64[],
    Acc_std = Float64[],
    TrainAcc_media = Float64[],
    TrainAcc_std = Float64[],
    F1_media = Float64[],
    F1_std = Float64[],
    ErrorRate_med = Float64[],
    ErrorRate_std = Float64[],
)

conf_matrices = Dict{String, Matrix{Int}}()

for cfg in svm_configs
    name = cfg["name"]
    println("\n>>> Configuracion: $name")

    result = run_svm_cross_validation(
        inputs_raw,
        targets_raw,
        cross_val_idx,
        classes_ordered,
        cfg,
    )

    (acc_m, acc_s) = result[1]
    (train_acc_m, train_acc_s) = result[2]
    (err_m, err_s) = result[3]
    (f1_m, f1_s) = result[8]
    conf_mat = result[9]

    println("  Train Acc: $(round(train_acc_m * 100, digits=2)) ± $(round(train_acc_s * 100, digits=2)) %")
    println("  Accuracy : $(round(acc_m * 100, digits=2)) ± $(round(acc_s * 100, digits=2)) %")
    println("  F1-score : $(round(f1_m * 100, digits=2)) ± $(round(f1_s * 100, digits=2)) %")

    push!(results, (
        name,
        cfg["kernel"],
        Float64(cfg["C"]),
        Float64(cfg["gamma"]),
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
    println(f, "# El kernel sigmoide se incluye por completitud, aunque su rendimiento suele ser inferior")
    println(f, "")
    for cfg in svm_configs
        name = cfg["name"]
        mat = conf_matrices[name]
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
println("Mejor configuracion SVM por macro-F1: ", best_row.Nombre,
    " | Acc=", round(best_row.Acc_media * 100, digits=2),
    "% | F1=", round(best_row.F1_media * 100, digits=2), "%")


# ==============================================================================
# 7. RESUMEN
# ==============================================================================

println("\n", "="^60)
println("RESUMEN FINAL - SVM")
println("="^60)
println(rpad("Configuracion", 24), " | ", rpad("TrainAcc(%)", 18), " | ", rpad("Acc(%)", 18), " | ", rpad("F1(%)", 18))
println("-"^90)
for row in eachrow(results)
    println(
        rpad(row.Nombre, 24), " | ",
        rpad("$(round(row.TrainAcc_media * 100, digits=2)) ± $(round(row.TrainAcc_std * 100, digits=2))", 18), " | ",
        rpad("$(round(row.Acc_media * 100, digits=2)) ± $(round(row.Acc_std * 100, digits=2))", 18), " | ",
        rpad("$(round(row.F1_media * 100, digits=2)) ± $(round(row.F1_std * 100, digits=2))", 18),
    )
end
println("-"^90)
println("Mejor configuracion: ", best_row.Nombre,
    " | TrainAcc=$(round(best_row.TrainAcc_media * 100, digits=2))",
    " | Acc=$(round(best_row.Acc_media * 100, digits=2))",
    " | F1=$(round(best_row.F1_media * 100, digits=2))")
println("="^90)
