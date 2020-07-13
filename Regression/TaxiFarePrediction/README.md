# Assignment: Predict taxi fares in New York

In this assignment you're going to build an app that can predict taxi fares in New York.

The first thing you'll need is a data file with transcripts of New York taxi rides. The [NYC Taxi & Limousine Commission](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) provides yearly TLC Trip Record Data files which have exactly what you need.

Download the [Yellow Taxi Trip Records from December 2018](https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-12.csv) and save it as **yellow_tripdata_2018-12.csv**. 

This is a CSV file with 8,173,233 records that looks like this:
￼

![Data File](./assets/data.png)


There are a lot of columns with interesting information in this data file, but you will only train on the following:

* Column 0: The data provider vendor ID
* Column 3: Number of passengers
* Column 4: Trip distance
* Column 5: The rate code (standard, JFK, Newark, …)
* Column 9: Payment type (credit card, cash, …)
* Column 10: Fare amount

You are going to build a machine learning model in F# that will use columns 0, 3, 4, 5, and 9 as input, and use them to predict the taxi fare for every trip. Then you’ll compare the predicted fares with the actual taxi fares in column 10, and evaluate the accuracy of your model.

Let's get started. You need to build a new application from scratch by opening a terminal and creating a new NET Core console project:

```bash
$ dotnet new console --language F# --output PricePrediction
$ cd PricePrediction
```

Now install the following packages

```bash
$ dotnet add package Microsoft.ML
$ dotnet add package Microsoft.ML.FastTree
```

Now you are ready to add some classes. You’ll need one to hold a taxi trip, and one to hold your model predictions.

Edit the Program.fs file with Visual Studio Code and replace its contents with the following code:

```fsharp
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

// the rest of the code goes here...
```

The **TaxiTrip** type holds one single taxi trip. Note how each field is tagged with a **LoadColumn** attribute that tells the CSV data loading code which column to import data from.

You're also declaring a **TaxiTripFarePrediction** type which will hold a single fare prediction.

Note the **CLIMutable** attribute that tells F# that we want a 'C#-style' class implementation with a default constructor and setter functions for every property. Without this attribute the compiler would generate an F#-style immutable class with read-only properties and no default constructor. The ML.NET library cannot handle immutable classes.  

Also note the **mutable** keyword in the definition for **TaxiTripFarePrediction**. By default F# types are immutable and the compiler will prevent us from assigning to any property after the type has been instantiated. The **mutable** keyword tells the compiler to create a mutable type instead and allow property assignments after construction. 

We're loading all data columns as **float32**, except **VendorId**, **RateCode** and **PaymentType**. These columns hold numeric values but you will load them as string fields.

The reason you need to do this is because RateCode is an enumeration with the following values:

* 1 = standard
* 2 = JFK
* 3 = Newark
* 4 = Nassau
* 5 = negotiated
* 6 = group

And PaymentType is defined as follows:

* 1 = Credit card
* 2 = Cash
* 3 = No charge
* 4 = Dispute
* 5 = Unknown
* 6 = Voided trip

These actual numbers don’t mean anything in this context. And we certainly don’t want the machine learning model to start believing that a trip to Newark is three times as important as a standard fare.

So converting these values to strings is a perfect trick to show the model that **VendorId**, **RateCode** and **PaymentType** are just labels, and the underlying numbers don’t mean anything.

Now you need to load the training data in memory:

```fsharp
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

    // the rest of the code goes here...
```

This code calls **LoadFromTextFile** to load the CSV data into memory. Note the **TaxiTrip** type that tells the method which class to use to load the data.

There is only one single data file, so you need to call **TrainTestSplit** to set up a training partition with 80% of the data and a test partition with the remaining 20% of the data.

You often see this 80/20 split in data science, it’s a very common approach to train and test a model.

Now you’re ready to start building the machine learning model:

```fsharp
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

// the rest of the code goes here...
```

Machine learning models in ML.NET are built with pipelines which are sequences of data-loading, transformation, and learning components.

This pipeline has the following components:

* A group of three **OneHotEncodings** to perform one hot encoding on the three columns that contains enumerative data: VendorId, RateCode, and PaymentType. This is a required step because we don't want the machine learning model to treat the enumerative data as numeric values.
* **Concatenate** which combines all input data columns into a single column called Features. This is a required step because ML.NET can only train on a single input column.
* **AppendCacheCheckpoint** which caches all data in memory to speed up the training process.
* A final **FastTree** regression learner which will train the model to make accurate predictions.

The **FastTreeRegressionTrainer** is a very nice training algorithm that uses gradient boosting, a machine learning technique for regression problems.

A gradient boosting algorithm builds up a collection of weak regression models. It starts out with a weak model that tries to predict the taxi fare. Then it adds a second model that attempts to correct the error in the first model. And then it adds a third model, and so on.

The result is a fairly strong prediction model that is actually just an ensemble of weaker prediction models stacked on top of each other.

We will explore Gradient Boosting in detail in a later section.

With the pipeline fully assembled, you can train the model on the training partition by piping the **TrainSet** into the **pipeline.Fit** function.

You now have a fully- trained model. So next, you'll have to grab the validation data, predict the taxi fare for each trip, and calculate the accuracy of your model:

```fsharp
// get regression metrics to score the model
let metrics = partitions.TestSet |> model.Transform |> context.Regression.Evaluate

// show the metrics
printfn "Model metrics:"
printfn "  RMSE:%f" metrics.RootMeanSquaredError
printfn "  MSE: %f" metrics.MeanSquaredError
printfn "  MAE: %f" metrics.MeanAbsoluteError

// the rest of the code goes here...
```

This code pipes the **TestSet** into the **model.Transform** function to generate predictions for every single taxi trip in the test partition. We then pipe these predictions into the **Evaluate** function to compare then to the actual taxi fares and automatically calculates these metrics:

* **RootMeanSquaredError**: this is the root mean squared error or RMSE value. It’s the go-to metric in the field of machine learning to evaluate models and rate their accuracy. RMSE represents the length of a vector in n-dimensional space, made up of the error in each individual prediction.
* **MeanAbsoluteError**: this is the mean absolute prediction error or MAE value, expressed in dollars.
* **MeanSquaredError**: this is the mean squared error, or MSE value. Note that RMSE and MSE are related: RMSE is the square root of MSE.

To wrap up, let’s use the model to make a prediction.

Imagine that I'm going to take a standard taxi trip, I cover a distance of 3.75 miles, I am the only passenger, and I pay by credit card. What would my fare be? 

Here’s how to make that prediction:

```fsharp
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
```

You use the **CreatePredictionEngine** method to set up a prediction engine. This is a type that can make predictions for individual data records. 

Next, you set up a sample with all the details of my taxi trip and pipe it into the **Predict** function to make a single prediction.

The trip should cost anywhere between $13.50 and $18.50, depending on the trip duration (which depends on the time of day). Will the model predict a fare in this range?  

Let's find out. Go to your terminal and run your code:

```bash
$ dotnet run
```

What results do you get? What are your RMSE and MAE values? Is this a good result? 

And how much does your model predict I have to pay for my taxi ride? Is the prediction in the range of accetable values for this trip? 

Now make some changes to my trip. Change the vendor ID, or the distance, or the manner of payment. How does this affect the final fare prediction? And what do you think this means?  

Think about the code in this assignment. How could you improve the accuracy of the model? What's your best RMSE value? 

Share your results in our group!
