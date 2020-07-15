# Assignment: Cluster Iris flowers

In this assignment you are going to build an unsupervised learning app that clusters Iris flowers into discrete groups. 

There are three types of Iris flowers: Versicolor, Setosa, and Virginica. Each flower has two sets of leaves: the inner Petals and the outer Sepals.

Your goal is to build an app that can identify an Iris flower by its sepal and petal size.

![MNIST digits](./assets/flowers.png)

Your challenge is that you're not going to use the dataset labels. Your app has to recognize patterns in the dataset and cluster the flowers into three groups without any help. 

Clustering is an example of **unsupervised learning** where the data science model has to figure out the labels on its own. 

The first thing you will need for your app is a data file with Iris flower petal and sepal sizes. You can use this [CSV file](https://github.com/mdfarragher/DSC/blob/master/Clustering/IrisFlower/iris-data.csv). Save it as **iris-data.csv** in your project folder.

The file looks like this:

![Data file](./assets/data.png)

It’s a CSV file with 5 columns:

* The length of the Sepal in centimeters
* The width of the Sepal in centimeters
* The length of the Petal in centimeters
* The width of the Petal in centimeters
* The type of Iris flower

You are going to build a clustering data science model that reads the data and then guesses the label for each flower in the dataset.

Of course the app won't know the real names of the flowers, so it's just going to number them: 1, 2, and 3.

Let’s get started. You need to build a new application from scratch by opening a terminal and creating a new NET Core console project:

```bash
$ dotnet new console --language F# --output IrisFlowers
$ cd IrisFlowers
```

Now install the ML.NET package:

```bash
$ dotnet add package Microsoft.ML
```

Now you are ready to add some types. You’ll need one to hold a flower and one to hold your model prediction.

Edit the Program.fs file and replace its contents with this:

```fsharp
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
    [<LoadColumn(4)>] Label : string
}

/// A type that holds a single model prediction.
[<CLIMutable>]
type IrisPrediction = {
    PredictedLabel : uint32
    Score : float32[]
}

// the rest of the code goes here....
```

The **IrisData** type holds one single flower. Note how the fields are tagged with the **LoadColumn** attribute that tells ML.NET how to load the data from the data file.

We are loading the label in the 5th column, but we won't be using the label during training because we want the model to figure out the iris flower types on its own.

There's also an **IrisPrediction** type which will hold a prediction for a single flower. The prediction consists of the ID of the cluster that the flower belongs to. Clusters are numbered from 1 upwards. And notice how the score field is an array? Each individual score value represents the distance of the flower to one specific cluster. 

Note the **CLIMutable** attribute that tells F# that we want a 'C#-style' class implementation with a default constructor and setter functions for every property. Without this attribute the compiler would generate an F#-style immutable class with read-only properties and no default constructor. The ML.NET library cannot handle immutable classes.  

Next you'll need to load the data in memory:

```fsharp
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

    // the rest of the code goes here....

    0 // return value
```

This code uses the **LoadFromTextFile** function to load the CSV data directly into memory, and then calls **TrainTestSplit** to split the dataset into an 80% training partition and a 20% test partition.

Now let’s build the data science pipeline:

```fsharp
// set up a learning pipeline
let pipeline = 
    EstimatorChain()

        // step 1: concatenate features into a single column
        .Append(context.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))

        // step 2: use k-means clustering to find the iris types
        .Append(context.Clustering.Trainers.KMeans(numberOfClusters = 3))

// train the model on the training data
let model = partitions.TrainSet |> pipeline.Fit 

// the rest of the code goes here...
```

Machine learning models in ML.NET are built with pipelines, which are sequences of data-loading, transformation, and learning components.

This pipeline has two components:

* **Concatenate** which converts the PixelValue vector into a single column called Features. This is a required step because ML.NET can only train on a single input column.
* A **KMeans** component which performs K-Means Clustering on the data and tries to find all Iris flower types. 

With the pipeline fully assembled, the code trains the model by piping the training set into the **Fit** function.

You now have a fully- trained model. So now it's time to take the test set, predict the type of each flower, and calculate the accuracy metrics of the model:

```fsharp
// get predictions and compare to ground truth
let metrics = partitions.TestSet |> model.Transform |> context.Clustering.Evaluate

// show results
printfn "Nodel results"
printfn "   Average distance:     %f" metrics.AverageDistance
printfn "   Davies Bouldin index: %f" metrics.DaviesBouldinIndex

// the rest of the code goes here....
```

This code pipes the test set into the **Transform** function to set up predictions for every flower in the test set. Then it pipes these predictions into the **Evaluate** function to compare each predictions with the label and automatically calculates two metrics:

* **AverageDistance**: this is the average distance of a flower to the center point of its cluster, averaged over all clusters in the dataset. It is a measure for the 'tightness' of the clusters. Lower values are better and mean more concentrated clusters. 
* **DaviesBouldinIndex**: this metric is the average 'similarity' of each cluster with its most similar cluster. Similarity is defined as the ratio of within-cluster distances to between-cluster distances. So in other words, clusters which are farther apart and more concentrated will result in a better score. Low values indicate better clustering.

So Average Distance measures how concentrated the clusters are in the dataset, and the Davies Bouldin Index measures both concentration and how far apart the clusters are spaced. Both metrics are negative-based with zero being the perfect score.

To wrap up, let’s use the model to make predictions.

You will pick three arbitrary flowers from the test set, run them through the model, and compare the predictions with the labels provided in the data file.

Here’s how to do it:

```fsharp
    // set up a prediction engine
    let engine = context.Model.CreatePredictionEngine model

    // grab 3 flowers from the dataset
    let flowers = context.Data.CreateEnumerable<IrisData>(partitions.TestSet, reuseRowObject = false) |> Array.ofSeq
    let testFlowers = [ flowers.[0]; flowers.[10]; flowers.[20] ]

    // show predictions for the three flowers
    printfn "Predictions for the 3 test flowers:"
    printfn "  Label\t\t\tPredicted\tScores"
    testFlowers |> Seq.iter(fun f -> 
            let p = engine.Predict f
            printf "  %-15s\t%i\t\t" f.Label p.PredictedLabel
            p.Score |> Seq.iter(fun s -> printf "%f\t" s)
            printfn "")
```

This code calls **CreatePredictionEngine** to set up a prediction engine. This is a type that can generate individual predictions from sample data.

Then we call the **CreateEnumerable** function to convert the test partition into an array of **IrisData** instances. Note the **Array.ofSeq** function at the end which converts the enumeration to an array.

Next, we pick three test flowers and pipe them into **Seq.iter**. For each flower, we generate a prediction, print the predicted label (a cluster ID between 1 and 3) and then use a second **Seq.iter** to write the three scores to the console. 

That's it, you're done!

Go to your terminal and run your code:

```bash
$ dotnet run
```

What results do you get? What is your average distance and your davies bouldin index? 

What do you think this says about the quality of the clusters?

What did the 3 flower predictions look like? Does the cluster prediction match the label every time? 

Now change the code and check the predictions for every flower. How often does the model get it wrong? Which Iris types are the most confusing to the model?

Share your results in our group. 