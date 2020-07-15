# Assignment: Recommend new movies to film fans

In this assignment you're going to build a movie recommendation system that can recommend new movies to film fans.

The first thing you'll need is a data file with thousands of movies rated by many different users. The [MovieLens Project](https://movielens.org) has exactly what you need.

Download the [movie ratings for training](https://github.com/mdfarragher/DSC/blob/master/Recommendation/MovieRecommender/recommendation-ratings-train.csv), [movie ratings for testing](https://github.com/mdfarragher/DSC/blob/master/Recommendation/MovieRecommender/recommendation-ratings-test.csv), and the [movie dictionary](https://github.com/mdfarragher/DSC/blob/master/Recommendation/MovieRecommender/recommendation-movies.csv) and save these files in your project folder. You now have 100,000 movie ratings with 99,980 set aside for training and 20 for testing. 

The training and testing files are in CSV format and look like this:
￼

![Data File](./assets/data.png)

There are only four columns of data:

* The ID of the user
* The ID of the movie
* The movie rating on a scale from 1–5
* The timestamp of the rating

There's also a movie dictionary in CSV format with all the movie IDs and titles:


![Data File](./assets/movies.png)

You are going to build a data science model that reads in each user ID, movie ID, and rating, and then predicts the ratings each user would give for every movie in the dataset.

Once you have a fully trained model, you can easily add a new user with a couple of favorite movies and then ask the model to generate predictions for any of the other movies in the dataset.

And in fact this is exactly how the recommendation systems on Netflix and Amazon work. 

Let's get started. You need to build a new application from scratch by opening a terminal and creating a new NET Core console project:

```bash
$ dotnet new console --language F# --output MovieRecommender
$ cd MovieRecommender
```

Now install the following packages

```bash
$ dotnet add package Microsoft.ML
$ dotnet add package Microsoft.ML.Recommender
```

Now you're ready to add some types. You will need one type to hold a movie rating, and one to hold your model’s predictions.

Edit the Program.fs file with Visual Studio Code and replace its contents with the following code:

```fsharp
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

// the rest of the code goes here...
```

The **MovieRating** type holds one single movie rating. Note how each field is tagged with a **LoadColumn** attribute that tell the CSV data loading code which column to import data from.

You're also declaring a **MovieRatingPrediction** type which will hold a single movie rating prediction.

Note the **CLIMutable** attribute that tells F# that we want a 'C#-style' class implementation with a default constructor and setter functions for every property. Without this attribute the compiler would generate an F#-style immutable class with read-only properties and no default constructor. The ML.NET library cannot handle immutable classes.  

Before we continue, we need to set up a third type that will hold our movie dictionary:

```fsharp
/// The MovieTitle class holds a single movie title.
[<CLIMutable>]
type MovieTitle = {
    [<LoadColumn(0)>] MovieID : float32
    [<LoadColumn(1)>] Title : string
    [<LoadColumn(2)>] Genres: string
}

// the rest of the code goes here
```

This **MovieTitle** type contains a movie ID value and its corresponding title and genres. We will use this type later in our code to map movie IDs to their corresponding titles.

Now you need to load the dataset in memory:

```fsharp
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

    // the rest of the code goes here...

    0 // return value
```

This code calls the **LoadFromTextFile** function twice to load the training and testing CSV data into memory. The field annotations we set up earlier tell the function how to store the loaded data in the **MovieRating** class.

Now you're ready to start building the machine learning model:

```fsharp
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

// the rest of the code goes here...
```

Machine learning models in ML.NET are built with pipelines, which are sequences of data-loading, transformation, and learning components.

This pipeline has the following components:

* **MapValueToKey** which reads the UserID column and builds a dictionary of unique ID values. It then produces an output column called UserIDEncoded containing an encoding for each ID. This step converts the IDs to numbers that the model can work with.
* Another **MapValueToKey** which reads the MovieID column, encodes it, and stores the encodings in output column called MovieIDEncoded.
* A **MatrixFactorization** component that performs matrix factorization on the encoded ID columns and the ratings. This step calculates the movie rating predictions for every user and movie.

With the pipeline fully assembled, you train the model by piping the training data into the **Fit** function.

You now have a fully- trained model. So now you need to load the validation data, predict the rating for each user and movie, and calculate the accuracy metrics of the model:

```fsharp
// calculate predictions and compare them to the ground truth
let metrics = testData |> model.Transform |> context.Regression.Evaluate

// show model metrics
printfn "Model metrics:"
printfn "  RMSE: %f" metrics.RootMeanSquaredError
printfn "  MAE:  %f" metrics.MeanAbsoluteError
printfn "  MSE:  %f" metrics.MeanSquaredError

// the rest of the code goes here...
```

This code pipes the test data into the **Transform** function to make predictions for every user and movie in the test dataset. It then pipes these predictions into the **Evaluate** function to compare them to the actual ratings.

The **Evaluate** function calculates the following three metrics:

* **RootMeanSquaredError**: this is the root mean square error or RMSE value. It’s the go-to metric in the field of machine learning to evaluate models and rate their accuracy. RMSE represents the length of a vector in n-dimensional space, made up of the error in each individual prediction.
* **MeanAbsoluteError**: this is the mean absolute prediction error, expressed as a rating.
* **MeanSquaredError**: this is the mean square prediction error, or MSE value. Note that RMSE and MSE are related: RMSE is just the square root of MSE.

To wrap up, let’s use the model to make a prediction about me. Here are 6 movies I like:

* Blade Runner
* True Lies
* Speed
* Twelve Monkeys
* Things to do in Denver when you're dead
* Cloud Atlas

And 6 more movies I really didn't like at all:

* Ace Ventura: when nature calls
* Naked Gun 33 1/3
* Highlander II
* Throw momma from the train
* Jingle all the way
* Dude, where's my car?

You'll find my ratings at the very end of the training file. I added myself as user 999. 

So based on this list, do you think I would enjoy the James Bond movie ‘GoldenEye’?

Let's write some code to find out:

```fsharp
// set up a prediction engine
let engine = context.Model.CreatePredictionEngine model

// check if Mark likes 'GoldenEye'
printfn "Does Mark like GoldenEye?"
let p = engine.Predict { UserID = 999.0f; MovieID = 10.0f; Label = 0.0f }
printfn "  Score: %f" p.Score

// the rest of the code goes here...
```

This code uses the **CreatePredictionEngine** method to set up a prediction engine, and then calls **Predict** to create a prediction for user 999 (me) and movie 10 (GoldenEye). 

Let’s do one more thing and ask the model to predict my top-5 favorite movies. 

We can ask the model to predict my favorite movies, but it will just produce movie ID values. So now's the time to load that movie dictionary that will help us convert movie IDs to their corresponding titles:

```fsharp
// load all movie titles
let movieData = context.Data.LoadFromTextFile<MovieTitle>(titleDataPath, hasHeader = true, separatorChar = ',', allowQuoting = true)
let movies = context.Data.CreateEnumerable(movieData, reuseRowObject = false)

// the rest of the code goes here...
```

This code calls **LoadFromTextFile** to load the movie dictionary in memory, and then calls **CreateEnumerable** to create an enumeration of **MovieTitle** instances. 

We can now find my favorite movies like this:

```fsharp
// find Mark's top 5 movies
let marksMovies = 
    movies |> Seq.map(fun m ->
        let p2 = engine.Predict { UserID = 999.0f; MovieID = m.MovieID; Label = 0.0f }
        (m.Title, p2.Score))
    |> Seq.sortByDescending(fun t -> snd t)

// print the results
printfn "What are Mark's top-5 movies?"
marksMovies |> Seq.take(5) |> Seq.iter(fun t -> printfn "  %f %s" (snd t) (fst t))
```

The code pipes the movie dictionary into **Seq.map** to create an enumeration of tuples. The first tuple element is the movie title and the second element is the rating the model thinks I would give to that movie.

The code then pipes the enumeration of tuples into Seq.**sortByDescending** to sort the list by rating. This will put my favorite movies at the top of the list.

Finally, the code pipes the rated movie list into **Seq.take** to grab the top-5, and then prints out the title and correspnding rating. 

That's it, your code is done. Go to your terminal and run the app:

```bash
$ dotnet run
```

Which training and validation metrics did you get? What are your RMSE and MAE values? Now look at how the data has been partitioned into training and validaton sets. Do you think this a good result? What could you improve?

What rating did the model predict I would give to the movie GoldenEye? And what are my 5 favorite movies according to the model? 

Share your results in our group and then ask me if the predictions are correct ;)
