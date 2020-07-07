open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data

/// The SpamInput class contains one single message which may be spam or ham.
[<CLIMutable>]
type SpamInput = {
    [<LoadColumn(0)>] Verdict : string
    [<LoadColumn(1)>] Message : string
}

/// The SpamPrediction class contains one single spam prediction.
[<CLIMutable>]
type SpamPrediction = {
    [<ColumnName("PredictedLabel")>] IsSpam : bool
    Score : float32
    Probability : float32
}

/// This class describes what output columns we want to produce.
[<CLIMutable>]
type ToLabel ={
    mutable Label : bool
}

/// Helper function to cast the ML pipeline to an estimator
let castToEstimator (x : IEstimator<_>) = 
    match x with 
    | :? IEstimator<ITransformer> as y -> y
    | _ -> failwith "Cannot cast pipeline to IEstimator<ITransformer>"

/// file paths to data files (assumes os = windows!)
let dataPath = sprintf "%s\\spam.tsv" Environment.CurrentDirectory

[<EntryPoint>]
let main arv =

    // set up a machine learning context
    let context = new MLContext()

    // load the spam dataset in memory
    let data = context.Data.LoadFromTextFile<SpamInput>(dataPath, hasHeader = true, separatorChar = '\t')

    // use 80% for training and 20% for testing
    let partitions = context.Data.TrainTestSplit(data, testFraction = 0.2)

    // set up a training pipeline
    let pipeline = 
        EstimatorChain()

            // step 1: transform the 'spam' and 'ham' values to true and false
            .Append(
                context.Transforms.CustomMapping(
                    Action<SpamInput, ToLabel>(fun input output -> output.Label <- input.Verdict = "spam"),
                    "MyLambda"))

            // step 2: featureize the input text
            .Append(context.Transforms.Text.FeaturizeText("Features", "Message"))

            // step 3: use a stochastic dual coordinate ascent learner
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression())

    // test the full data set by performing k-fold cross validation
    printfn "Performing cross validation:"
    let cvResults = context.BinaryClassification.CrossValidate(data = data, estimator = castToEstimator pipeline, numberOfFolds = 5)

    // report the results
    cvResults |> Seq.iter(fun f -> printfn "  Fold: %i, AUC: %f" f.Fold f.Metrics.AreaUnderRocCurve)

    // train the model on the training set
    let model = partitions.TrainSet |> pipeline.Fit

    // evaluate the model on the test set
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
    let engine = context.Model.CreatePredictionEngine model

    // create sample messages
    let messages = [
        { Message = "Hi, wanna grab lunch together today?"; Verdict = "" }
        { Message = "Win a Nokia, PSP, or €25 every week. Txt YEAHIWANNA now to join"; Verdict = "" }
        { Message = "Home in 30 mins. Need anything from store?"; Verdict = "" }
        { Message = "CONGRATS U WON LOTERY CLAIM UR 1 MILIONN DOLARS PRIZE"; Verdict = "" }
    ]

    // make the predictions
    printfn "Model predictions:"
    let predictions = messages |> List.iter(fun m -> 
            let p = engine.Predict m
            printfn "  %f %s" p.Probability m.Message)

    0 // return value