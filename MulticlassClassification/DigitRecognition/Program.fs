open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms

/// The Digit class represents one mnist digit.
[<CLIMutable>]
type Digit = {
    [<LoadColumn(0)>] Number : float32
    [<LoadColumn(1, 784)>] [<VectorType(784)>] PixelValues : float32[]
}

/// The DigitPrediction class represents one digit prediction.
[<CLIMutable>]
type DigitPrediction = {
    Score : float32[]
}

/// file paths to train and test data files (assumes os = windows!)
let trainDataPath = sprintf "%s\\mnist_train.csv" Environment.CurrentDirectory
let testDataPath = sprintf "%s\\mnist_test.csv" Environment.CurrentDirectory

[<EntryPoint>]
let main argv = 

    // create a machine learning context
    let context = new MLContext()

    // load the datafiles
    let trainData = context.Data.LoadFromTextFile<Digit>(trainDataPath, hasHeader = true, separatorChar = ',')
    let testData = context.Data.LoadFromTextFile<Digit>(testDataPath, hasHeader = true, separatorChar = ',')

    // build a training pipeline
    let pipeline = 
        EstimatorChain()

            // step 1: map the number column to a key value and store in the label column
            .Append(context.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality = ValueToKeyMappingEstimator.KeyOrdinality.ByValue))

            // step 2: concatenate all feature columns
            .Append(context.Transforms.Concatenate("Features", "PixelValues"))
            
            // step 3: cache data to speed up training                
            .AppendCacheCheckpoint(context)

            // step 4: train the model with SDCA
            .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy())

            // step 5: map the label key value back to a number
            .Append(context.Transforms.Conversion.MapKeyToValue("Number", "Label"))

    // train the model
    let model = trainData |> pipeline.Fit

    // get predictions and compare them to the ground truth
    let metrics = testData |> model.Transform |> context.MulticlassClassification.Evaluate

    // show evaluation metrics
    printfn "Evaluation metrics"
    printfn "  MicroAccuracy:    %f" metrics.MicroAccuracy
    printfn "  MacroAccuracy:    %f" metrics.MacroAccuracy
    printfn "  LogLoss:          %f" metrics.LogLoss
    printfn "  LogLossReduction: %f" metrics.LogLossReduction

    // grab five digits from the test data
    let digits = context.Data.CreateEnumerable(testData, reuseRowObject = false) |> Array.ofSeq
    let testDigits = [ digits.[5]; digits.[16]; digits.[28]; digits.[63]; digits.[129] ]

    // create a prediction engine
    let engine = context.Model.CreatePredictionEngine<Digit, DigitPrediction> model

    // show predictions
    printfn "Model predictions:"
    printf "  #\t\t"; [0..9] |> Seq.iter(fun i -> printf "%i\t\t" i); printfn ""
    testDigits |> Seq.iter(
        fun digit -> 
            printf "  %i\t" (int digit.Number)
            let p = engine.Predict digit
            p.Score |> Seq.iter (fun s -> printf "%f\t" s)
            printfn "")

    0 // return value