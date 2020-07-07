open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms

/// The Passenger class represents one passenger on the Titanic.
[<CLIMutable>]
type Passenger = {
    [<LoadColumn(1)>] Label : bool
    [<LoadColumn(2)>] Pclass : float32
    [<LoadColumn(4)>] Sex : string
    [<LoadColumn(5)>] RawAge : string // not a float!
    [<LoadColumn(6)>] SibSp : float32
    [<LoadColumn(7)>] Parch : float32
    [<LoadColumn(8)>] Ticket : string
    [<LoadColumn(9)>] Fare : float32
    [<LoadColumn(10)>] Cabin : string
    [<LoadColumn(11)>] Embarked : string
}

/// The PassengerPrediction class represents one model prediction. 
[<CLIMutable>]
type PassengerPrediction = {
    [<ColumnName("PredictedLabel")>] Prediction : bool
    Probability : float32
    Score : float32
}

/// The ToAge class is a helper class for a column transformation.
[<CLIMutable>]
type ToAge = {
    mutable Age : string
}

/// file path to the train data file (assumes os = windows!)
let trainDataPath = sprintf "%s\\train_data.csv" Environment.CurrentDirectory

/// file path to the test data file (assumes os = windows!)
let testDataPath = sprintf "%s\\test_data.csv" Environment.CurrentDirectory

[<EntryPoint>]
let main argv = 

    // set up a machine learning context
    let context = new MLContext()

    // load the training and testing data in memory
    let trainData = context.Data.LoadFromTextFile<Passenger>(trainDataPath, hasHeader = true, separatorChar = ',', allowQuoting = true)
    let testData = context.Data.LoadFromTextFile<Passenger>(testDataPath, hasHeader = true, separatorChar = ',', allowQuoting = true)

    // set up a training pipeline
    let pipeline = 
        EstimatorChain()

            // step 1: replace missing ages with '?'
            .Append(
                context.Transforms.CustomMapping(
                    Action<Passenger, ToAge>(fun input output -> output.Age <- if String.IsNullOrEmpty(input.RawAge) then "?" else input.RawAge),
                    "AgeMapping"))

            // step 2: convert string ages to floats
            .Append(context.Transforms.Conversion.ConvertType("Age", outputKind = DataKind.Single))

            // step 3: replace missing age values with the mean age
            .Append(context.Transforms.ReplaceMissingValues("Age", replacementMode = MissingValueReplacingEstimator.ReplacementMode.Mean))

            // step 4: replace string columns with one-hot encoded vectors
            .Append(context.Transforms.Categorical.OneHotEncoding("Sex"))
            .Append(context.Transforms.Categorical.OneHotEncoding("Ticket"))
            .Append(context.Transforms.Categorical.OneHotEncoding("Cabin"))
            .Append(context.Transforms.Categorical.OneHotEncoding("Embarked"))

            // step 5: concatenate everything into a single feature column 
            .Append(context.Transforms.Concatenate("Features", "Age", "Pclass", "SibSp", "Parch", "Sex", "Embarked"))

            // step 6: use a fasttree trainer
            .Append(context.BinaryClassification.Trainers.FastTree())

    // train the model
    let model = trainData |> pipeline.Fit

    // make predictions and compare with ground truth
    let metrics = testData |> model.Transform |> context.BinaryClassification.Evaluate

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
    let engine = context.Model.CreatePredictionEngine model

    // create a sample record
    let passenger = {
        Pclass = 1.0f
        Sex = "male"
        RawAge = "48"
        SibSp = 0.0f
        Parch = 0.0f
        Ticket = "B"
        Fare = 70.0f
        Cabin = "123"
        Embarked = "S"
        Label = false // unused!
    }

    // make the prediction
    let prediction = engine.Predict passenger

    // report the results
    printfn "Model prediction:"
    printfn "  Prediction:  %s" (if prediction.Prediction then "survived" else "perished")
    printfn "  Probability: %f" prediction.Probability

    0 // return value