open System
open Microsoft.ML
open Microsoft.ML.Data

/// The TaxiTrip class represents a single taxi trip.
[<CLIMutable>]
type TaxiTrip = {
    [<LoadColumn(0)>] VendorId : string
    [<LoadColumn(5)>] RateCode : string
    [<LoadColumn(3)>] PassengerCount : float32
    [<LoadColumn(4)>] TripDistance : float32
    [<LoadColumn(9)>] PaymentType : string
    [<LoadColumn(10)>] [<ColumnName("Label")>] FareAmount : float32
}

/// The TaxiTripFarePrediction class represents a single far prediction.
[<CLIMutable>]
type TaxiTripFarePrediction = {
    [<ColumnName("Score")>] FareAmount : float32
}

// file paths to data files (assumes os = windows!)
let dataPath = sprintf "%s\\yellow_tripdata_2018-12_small.csv" Environment.CurrentDirectory

/// The main application entry point.
[<EntryPoint>]
let main argv =

    // create the machine learning context
    let context = new MLContext()

    // load the data
    let dataView = context.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader = true, separatorChar = ',')

    // split into a training and test partition
    let partitions = context.Data.TrainTestSplit(dataView, testFraction = 0.2)

    // set up a learning pipeline
    let pipeline = 
        EstimatorChain()
    
            // one-hot encode all text features
            .Append(context.Transforms.Categorical.OneHotEncoding("VendorId"))
            .Append(context.Transforms.Categorical.OneHotEncoding("RateCode"))
            .Append(context.Transforms.Categorical.OneHotEncoding("PaymentType"))

            // combine all input features into a single column 
            .Append(context.Transforms.Concatenate("Features", "VendorId", "RateCode", "PaymentType", "PassengerCount", "TripDistance"))

            // cache the data to speed up training
            .AppendCacheCheckpoint(context)

            // use the fast tree learner 
            .Append(context.Regression.Trainers.FastTree())

    // train the model
    let model = partitions.TrainSet |> pipeline.Fit

    // get regression metrics to score the model
    let metrics = partitions.TestSet |> model.Transform |> context.Regression.Evaluate

    // show the metrics
    printfn "Model metrics:"
    printfn "  RMSE:%f" metrics.RootMeanSquaredError
    printfn "  MSE: %f" metrics.MeanSquaredError
    printfn "  MAE: %f" metrics.MeanAbsoluteError

    // create a prediction engine for one single prediction
    let engine = context.Model.CreatePredictionEngine model

    let taxiTripSample = {
        VendorId = "VTS"
        RateCode = "1"
        PassengerCount = 1.0f
        TripDistance = 3.75f
        PaymentType = "CRD"
        FareAmount = 0.0f // To predict. Actual/Observed = 15.5
    }

    // make the prediction
    let prediction = taxiTripSample |> engine.Predict

    // show the prediction
    printfn "\r"
    printfn "Single prediction:"
    printfn "  Predicted fare: %f" prediction.FareAmount

    0 // return value