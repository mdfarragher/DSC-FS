open System
open Microsoft.ML
open Microsoft.ML.Trainers
open Microsoft.ML.Data

/// The MovieRating class holds a single movie rating.
[<CLIMutable>]
type MovieRating = {
    [<LoadColumn(0)>] UserID : float32
    [<LoadColumn(1)>] MovieID : float32
    [<LoadColumn(2)>] Label : float32
}

/// The MovieRatingPrediction class holds a single movie prediction.
[<CLIMutable>]
type MovieRatingPrediction = {
    Label : float32
    Score : float32
}

/// The MovieTitle class holds a single movie title.
[<CLIMutable>]
type MovieTitle = {
    [<LoadColumn(0)>] MovieID : float32
    [<LoadColumn(1)>] Title : string
    [<LoadColumn(2)>] Genres: string
}

// file paths to data files (assumes os = windows!)
let trainDataPath = sprintf "%s\\recommendation-ratings-train.csv" Environment.CurrentDirectory
let testDataPath = sprintf "%s\\recommendation-ratings-test.csv" Environment.CurrentDirectory
let titleDataPath = sprintf "%s\\recommendation-movies.csv" Environment.CurrentDirectory

[<EntryPoint>]
let main argv = 

    // set up a new machine learning context
    let context = new MLContext()

    // load training and test data
    let trainData = context.Data.LoadFromTextFile<MovieRating>(trainDataPath, hasHeader = true, separatorChar = ',')
    let testData = context.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader = true, separatorChar = ',')

    // prepare matrix factorization options
    let options = 
        MatrixFactorizationTrainer.Options(
            MatrixColumnIndexColumnName = "UserIDEncoded",
            MatrixRowIndexColumnName = "MovieIDEncoded",
            LabelColumnName = "Label",
            NumberOfIterations = 20,
            ApproximationRank = 100)

    // set up a training pipeline
    let pipeline = 
        EstimatorChain()

            // step 1: map userId and movieId to keys
            .Append(context.Transforms.Conversion.MapValueToKey("UserIDEncoded", "UserID"))
            .Append(context.Transforms.Conversion.MapValueToKey("MovieIDEncoded", "MovieID"))

            // step 2: find recommendations using matrix factorization
            .Append(context.Recommendation().Trainers.MatrixFactorization(options))

    // train the model
    let model = trainData |> pipeline.Fit

    // calculate predictions and compare them to the ground truth
    let metrics = testData |> model.Transform |> context.Regression.Evaluate

    // show model metrics
    printfn "Model metrics:"
    printfn "  RMSE: %f" metrics.RootMeanSquaredError
    printfn "  MAE:  %f" metrics.MeanAbsoluteError
    printfn "  MSE:  %f" metrics.MeanSquaredError

    // set up a prediction engine
    let engine = context.Model.CreatePredictionEngine model

    // check if Mark likes 'GoldenEye'
    printfn "Does Mark like GoldenEye?"
    let p = engine.Predict { UserID = 999.0f; MovieID = 10.0f; Label = 0.0f }
    printfn "  Score: %f" p.Score

    // load all movie titles
    let movieData = context.Data.LoadFromTextFile<MovieTitle>(titleDataPath, hasHeader = true, separatorChar = ',', allowQuoting = true)
    let movies = context.Data.CreateEnumerable(movieData, reuseRowObject = false)

    // find Mark's top 5 movies
    let marksMovies = 
        movies |> Seq.map(fun m ->
            let p2 = engine.Predict { UserID = 999.0f; MovieID = m.MovieID; Label = 0.0f }
            (m.Title, p2.Score))
        |> Seq.sortByDescending(fun t -> snd t)

    // print the results
    printfn "What are Mark's top-5 movies?"
    marksMovies |> Seq.take(5) |> Seq.iter(fun t -> printfn "  %f %s" (snd t) (fst t))

    0 // return value
