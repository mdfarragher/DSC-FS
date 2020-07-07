open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data

/// The HeartData record holds one single heart data record.
[<CLIMutable>]
type HeartData = {
    [<LoadColumn(0)>] Age : float32
    [<LoadColumn(1)>] Sex : float32
    [<LoadColumn(2)>] Cp : float32
    [<LoadColumn(3)>] TrestBps : float32
    [<LoadColumn(4)>] Chol : float32
    [<LoadColumn(5)>] Fbs : float32
    [<LoadColumn(6)>] RestEcg : float32
    [<LoadColumn(7)>] Thalac : float32
    [<LoadColumn(8)>] Exang : float32
    [<LoadColumn(9)>] OldPeak : float32
    [<LoadColumn(10)>] Slope : float32
    [<LoadColumn(11)>] Ca : float32
    [<LoadColumn(12)>] Thal : float32
    [<LoadColumn(13)>] Diagnosis : float32
}

/// The HeartPrediction class contains a single heart data prediction.
[<CLIMutable>]
type HeartPrediction = {
    [<ColumnName("PredictedLabel")>] Prediction : bool
    Probability : float32
    Score : float32
}

/// The ToLabel class is a helper class for a column transformation.
[<CLIMutable>]
type ToLabel = {
    mutable Label : bool
}

/// file paths to data files (assumes os = windows!)
let dataPath = sprintf "%s\\processed.cleveland.data.csv" Environment.CurrentDirectory

/// The main application entry point.
[<EntryPoint>]
let main argv =

    // set up a machine learning context
    let context = new MLContext()

    // load training and test data
    let data = context.Data.LoadFromTextFile<HeartData>(dataPath, hasHeader = false, separatorChar = ',')

    // split the data into a training and test partition
    let partitions = context.Data.TrainTestSplit(data, testFraction = 0.2)

    // set up a training pipeline
    let pipeline = 
        EstimatorChain()

            // step 1: convert the label value to a boolean
            .Append(
                context.Transforms.CustomMapping(
                    Action<HeartData, ToLabel>(fun input output -> output.Label <- input.Diagnosis > 0.0f),
                    "LabelMapping"))
    
            // step 2: concatenate all feature columns
            .Append(context.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal"))

            // step 3: set up a fast tree learner
            .Append(context.BinaryClassification.Trainers.FastTree())

    // train the model
    let model = partitions.TrainSet |> pipeline.Fit

    // make predictions and compare with the ground truth
    let metrics = partitions.TestSet |> model.Transform |> context.BinaryClassification.Evaluate

    // report the results
    printfn "Model metrics:"
    printfn "  Accuracy:          %f" metrics.Accuracy
    printfn "  Auc:               %f" metrics.AreaUnderRocCurve
    printfn "  Auprc:             %f" metrics.AreaUnderPrecisionRecallCurve
    printfn "  F1Score:           %f" metrics.F1Score
    printfn "  LogLoss:           %f" metrics.LogLoss
    printfn "  LogLossReduction:  %f" metrics.LogLossReduction
    printfn "  PositivePrecision: %f" metrics.PositivePrecision
    printfn "  PositiveRecall:    %f" metrics.PositiveRecall
    printfn "  NegativePrecision: %f" metrics.NegativePrecision
    printfn "  NegativeRecall:    %f" metrics.NegativeRecall

    // set up a prediction engine
    let predictionEngine = context.Model.CreatePredictionEngine model

    // create a sample patient
    let sample = { 
        Age = 36.0f
        Sex = 1.0f
        Cp = 4.0f
        TrestBps = 145.0f
        Chol = 210.0f
        Fbs = 0.0f
        RestEcg = 2.0f
        Thalac = 148.0f
        Exang = 1.0f
        OldPeak = 1.9f
        Slope = 2.0f
        Ca = 1.0f
        Thal = 7.0f
        Diagnosis = 0.0f // unused
    }

    // make the prediction
    let prediction = sample |> predictionEngine.Predict

    // report the results
    printfn "\r"
    printfn "Single prediction:"
    printfn "  Prediction:  %s" (if prediction.Prediction then "Elevated heart disease risk" else "Normal heart disease risk")
    printfn "  Probability: %f" prediction.Probability

    0 // return value