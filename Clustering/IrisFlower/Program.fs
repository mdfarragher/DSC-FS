open System
open Microsoft.ML
open Microsoft.ML.Data

/// A type that holds a single iris flower.
[<CLIMutable>]
type IrisData = {
    [<LoadColumn(0)>] SepalLength : float32
    [<LoadColumn(1)>] SepalWidth : float32
    [<LoadColumn(2)>] PetalLength : float32
    [<LoadColumn(3)>] PetalWidth : float32
    [<LoadColumn(4)>] Label : int
}

/// A type that holds a single model prediction.
[<CLIMutable>]
type IrisPrediction = {
    PredictedLabel : float32
    Score : float32[]
}

/// file paths to data files (assumes os = windows!)
let dataPath = sprintf "%s\\iris-data.csv" Environment.CurrentDirectory

[<EntryPoint>]
let main argv = 

    // get the machine learning context
    let context = new MLContext();

    // read the iris flower data from a text file
    let data = context.Data.LoadFromTextFile<IrisData>(dataPath, hasHeader = false, separatorChar = ',')

    // split the data into a training and testing partition
    let partitions = context.Data.TrainTestSplit(data, testFraction = 0.2)

    // set up a learning pipeline
    let pipeline = 
        EstimatorChain()

            // step 1: concatenate features into a single column
            .Append(context.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))

            // step 2: use k-means clustering to find the iris types
            .Append(context.Clustering.Trainers.KMeans(numberOfClusters = 3))

    // train the model on the training data
    let model = partitions.TrainSet |> pipeline.Fit 

    // get predictions and compare to ground truth
    let metrics = partitions.TestSet |> model.Transform |> context.Clustering.Evaluate

    // show results
    printfn "Nodel results"
    printfn "   Average distance:   %f" metrics.AverageDistance
    printfn "   Davies Bould index: %f" metrics.DaviesBouldinIndex

    // set up a prediction engine
    let engine = context.Model.CreatePredictionEngine model

    // grab 3 flowers from the dataset
    let flowers = context.Data.CreateEnumerable<IrisData>(partitions.TestSet, reuseRowObject = false) |> Array.ofSeq
    let testFlowers = [ flowers.[0]; flowers.[10]; flowers.[20] ]

    // show predictions for the three flowers
    printfn "Predictions for the 3 test flowers:"
    testFlowers |> Seq.iter(fun f -> 
            let p = engine.Predict f
            printfn "  %i" f.PredictedLabel)

    0 // return value