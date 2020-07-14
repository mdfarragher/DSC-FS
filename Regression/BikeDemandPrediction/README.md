# Assignment: Predict bike sharing demand in Washington DC

In this assignment you're going to build an app that can predict bike sharing demand in Washington DC.

A bike-sharing system is a service in which bicycles are made available to individuals on a short term. Users borrow a bike from a dock and return it at another dock belonging to the same system. Docks are bike racks that lock the bike, and only release it by computer control.

You’ve probably seen docks around town, they look like this:

![Bike sharing rack](./assets/bikesharing.jpeg)

Bike sharing companies try to even out supply by manually distributing bikes across town, but they need to know how many bikes will be in demand at any given time in the city.

So let’s give them a hand with a machine learning model!

You are going to train a forest of regression decision trees on a dataset of bike sharing demand. Then you’ll use the fully-trained model to make a prediction for a given date and time.

The first thing you will need is a data file with lots of bike sharing demand numbers. We are going to use the [UCI Bike Sharing Dataset](http://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) from [Capital Bikeshare](https://www.capitalbikeshare.com/) in Metro DC. This dataset has 17,380 bike sharing records that span a 2-year period.

[Download the dataset](https://github.com/mdfarragher/DSC/blob/master/Regression/BikeDemandPrediction/bikedemand.csv) and save it in your project folder as **bikedmand.csv**.

The file looks like this:

![Data File](./assets/data.png)

It’s a comma-separated file with 17 columns:

* Instant: the record index
* Date: the date of the observation
* Season: the season (1 = springer, 2 = summer, 3 = fall, 4 = winter)
* Year: the year of the observation (0 = 2011, 1 = 2012)
* Month: the month of the observation ( 1 to 12)
* Hour: the hour of the observation (0 to 23)
* Holiday: if the date is a holiday or not
* Weekday: the day of the week of the observation
* WorkingDay: if the date is a working day
* Weather: the weather during the observation (1 = clear, 2 = mist, 3 = light snow/rain, 4 = heavy rain)
* Temperature : the normalized temperature in Celsius
* ATemperature: the normalized feeling temperature in Celsius
* Humidity: the normalized humidity
* Windspeed: the normalized wind speed
* Casual: the number of casual bike users at the time
* Registered: the number of registered bike users at the time
* Count: the total number of rental bikes in operation at the time

You can ignore the record index, the date, and the number of casual and registered bikes, and use everything else as input features. The final column **Count** is the label you're trying to predict.

Let's get started. You need to build a new application from scratch by opening a terminal and creating a new NET Core console project:

```bash
$ dotnet new console --language F# --output BikeDemand
$ cd BikeDemand
```

Now install the following packages

```bash
$ dotnet add package Microsoft.ML
$ dotnet add package Microsoft.ML.FastTree
```

Now you are ready to add some types. You’ll need one to hold a bike demand record, and one to hold your model predictions.

Edit the Program.fs file with Visual Studio Code and replace its contents with the following code:

```fsharp
open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data

/// The DemandObservation class holds one single bike demand observation record.
[<CLIMutable>]
type DemandObservation = {
    [<LoadColumn(2)>] Season : float32
    [<LoadColumn(3)>] Year : float32
    [<LoadColumn(4)>] Month : float32
    [<LoadColumn(5)>] Hour : float32
    [<LoadColumn(6)>] Holiday : float32
    [<LoadColumn(7)>] Weekday : float32
    [<LoadColumn(8)>] WorkingDay : float32
    [<LoadColumn(9)>] Weather : float32
    [<LoadColumn(10)>] Temperature : float32
    [<LoadColumn(11)>] NormalizedTemperature : float32
    [<LoadColumn(12)>] Humidity : float32
    [<LoadColumn(13)>] Windspeed : float32
    [<LoadColumn(16)>] [<ColumnName("Label")>] Count : float32
}

/// The DemandPrediction class holds one single bike demand prediction.
[<CLIMutable>]
type DemandPrediction = {
    [<ColumnName("Score")>] PredictedCount : float32;
}

// the rest of the code goes here...
```

The **DemandObservation** type holds one single bike trip. Note how each field is tagged with a **LoadColumn** attribute that tells the CSV data loading code which column to import data from.

You're also declaring a **DemandPrediction** type which will hold a single bike demand prediction.

Note the **CLIMutable** attribute that tells F# that we want a 'C#-style' class implementation with a default constructor and setter functions for every property. Without this attribute the compiler would generate an F#-style immutable class with read-only properties and no default constructor. The ML.NET library cannot handle immutable classes.  

Now you need to load the training data in memory:

```fsharp
// file paths to data files (assumes os = windows!)
let dataPath = sprintf "%s\\bikedemand.csv" Environment.CurrentDirectory

/// The main application entry point.
[<EntryPoint>]
let main argv =

    // create the machine learning context
    let context = new MLContext();

    // load the dataset
    let data = context.Data.LoadFromTextFile<DemandObservation>(dataPath, hasHeader = true, separatorChar = ',')

    // split the dataset into 80% training and 20% testing
    let partitions = context.Data.TrainTestSplit(data, testFraction = 0.2)

    // the rest of the code goes here...

    0 // return value
```

This code uses the method **LoadFromTextFile** to load the data directly into memory. The field annotations we set up earlier tell the method how to store the loaded data in the **DemandObservation** class.

The code then calls **TrainTestSplit** to reserve 80% of the data for training and 20% for testing.

Now let’s build the machine learning pipeline:

```fsharp
// build a training pipeline
let pipeline = 
    EstimatorChain()
    
        // step 1: concatenate all feature columns
        .Append(context.Transforms.Concatenate("Features", "Season", "Year", "Month", "Hour", "Holiday", "Weekday", "WorkingDay", "Weather", "Temperature", "NormalizedTemperature", "Humidity", "Windspeed"))
                                
        // step 2: cache the data to speed up training
        .AppendCacheCheckpoint(context)

        // step 3: use a fast forest learner
        .Append(context.Regression.Trainers.FastForest(numberOfLeaves = 20, numberOfTrees = 100, minimumExampleCountPerLeaf = 10))

// train the model
let model = partitions.TrainSet |> pipeline.Fit

// the rest of the code goes here...
```

Machine learning models in ML.NET are built with pipelines which are sequences of data-loading, transformation, and learning components.

This pipeline has the following components:

* **Concatenate** which combines all input data columns into a single column called Features. This is a required step because ML.NET can only train on a single input column.
* **AppendCacheCheckpoint** which caches all training data at this point. This is an optimization step that speeds up the learning algorithm.
* A final **FastForest** regression learner which will train the model to make accurate predictions using a forest of decision trees.

The **FastForest** learner is a very nice training algorithm that uses gradient boosting to build a forest of weak decision trees.

Gradient boosting builds a stack of weak decision trees. It starts with a single weak tree that tries to predict the bike demand. Then it adds a second tree on top of the first one to correct the error in the first tree. And then it adds a third tree on top of the second one to correct the output of the second tree. And so on.

The result is a fairly strong prediction model that is made up of a stack of weak decision trees that incrementally correct each other's output. 

Note the use of hyperparameters to configure the learner:

* **NumberOfLeaves** is the maximum number of leaf nodes each weak decision tree will have. In this forest each tree will have at most 10 leaf nodes.
* **NumberOfTrees** is the total number of weak decision trees to create in the forest. This forest will hold 100 trees.
* **MinimumExampleCountPerLeaf** is the minimum number of data points at which a leaf node is split. In this model each leaf is split when it has 10 or more qualifying data points.

These hyperparameters are the default for the **FastForest** learner, but you can tweak them if you want. 

With the pipeline fully assembled, you can pipe the trainig data into the **Fit** function to train the model.

You now have a fully- trained model. So next, you'll have to load the test data, predict the bike demand, and calculate the accuracy of your model:

```fsharp
// evaluate the model
let metrics = partitions.TestSet |> model.Transform |> context.Regression.Evaluate

// show evaluation metrics
printfn "Model metrics:"
printfn "  RMSE:%f" metrics.RootMeanSquaredError
printfn "  MSE: %f" metrics.MeanSquaredError
printfn "  MAE: %f" metrics.MeanAbsoluteError

// the rest of the code goes here...
```

This code pipes the test data into the **Transform** function to set up predictions for every single bike demand record in the test partition. The code then pipes these predictions into the **Evaluate** function to compares them to the actual bike demand and automatically calculate these metrics:

* **RootMeanSquaredError**: this is the root mean squared error or RMSE value. It’s the go-to metric in the field of machine learning to evaluate models and rate their accuracy. RMSE represents the length of a vector in n-dimensional space, made up of the error in each individual prediction.
* **MeanSquaredError**: this is the mean squared error, or MSE value. Note that RMSE and MSE are related: RMSE is the square root of MSE.
* **MeanAbsoluteError**: this is the mean absolute prediction error or MAE value, expressed in number of bikes.

To wrap up, let’s use the model to make a prediction.

I want to rent a bike in the fall of 2012, on a Thursday in August at 10am in the morning in clear weather. What will the bike demand be on that day?

Here’s how to make that prediction:

```fsharp
// set up a sample observation
let sample ={
    Season = 3.0f
    Year = 1.0f
    Month = 8.0f
    Hour = 10.0f
    Holiday = 0.0f
    Weekday = 4.0f
    WorkingDay = 1.0f
    Weather = 1.0f
    Temperature = 0.8f
    NormalizedTemperature = 0.7576f
    Humidity = 0.55f
    Windspeed = 0.2239f
    Count = 0.0f // the field to predict
}

// create a prediction engine
let engine = context.Model.CreatePredictionEngine model

// make the prediction
let prediction = sample |> engine.Predict

// show the prediction
printfn "\r"
printfn "Single prediction:"
printfn "  Predicted bike count: %f" prediction.PredictedCount
```

This code sets up a new bike demand observation, and then uses the **CreatePredictionEngine** function to set up a prediction engine and call **Predict** to make a demand prediction. 

What will the model prediction be?

Time to find out. Go to your terminal and run your code:

```bash
$ dotnet run
```

What results do you get? What are your RMSE and MAE values? Is this a good result? 

And what bike demand does your model predict on the day I wanted to take my bike ride? 

Now take a look at the hyperparameters. Try to change the behavior of the fast forest learner and see what happens to the accuracy of your model. Did your model improve or get worse? 

Share your results in our group!
