# Assignment: Recognize handwritten digits

In this article, You are going to build an app that recognizes handwritten digits from the famous MNIST machine learning dataset:

![MNIST digits](./assets/mnist.png)

Your app must read these images of handwritten digits and correctly predict which digit is visible in each image.

This may seem like an easy challenge, but look at this:

![Difficult MNIST digits](./assets/mnist_hard.png)

These are a couple of digits from the dataset. Are you able to identify each one? It probably won’t surprise you to hear that the human error rate on this exercise is around 2.5%.

The first thing you will need for your app is a data file with images of handwritten digits. We will not use the original MNIST data because it's stored in a nonstandard binary format.

Instead, we'll use these excellent [CSV files](https://www.kaggle.com/oddrationale/mnist-in-csv/) prepared by Daniel Dato on Kaggle.

Create a Kaggle account if you don't have one yet, then download **mnist_train.csv** and **mnist_test.csv** and save them in your project folder.

There are 60,000 images in the training file and 10,000 in the test file. Each image is monochrome and resized to 28x28 pixels.

The training file looks like this:

![Data file](./assets/datafile.png)

It’s a CSV file with 785 columns:

* The first column contains the label. It tells us which one of the 10 possible digits is visible in the image.
* The next 784 columns are the pixel intensity values (0..255) for each pixel in the image, counting from left to right and top to bottom.

You are going to build a multiclass classification machine learning model that reads in all 785 columns, and then makes a prediction for each digit in the dataset.

Let’s get started. You need to build a new application from scratch by opening a terminal and creating a new NET Core console project:

```bash
$ dotnet new console --language F# --output Mnist
$ cd Mnist
```

Now install the ML.NET package:

```bash
$ dotnet add package Microsoft.ML
```

Now you are ready to add types. You’ll need one to hold a digit, and one to hold your model prediction.

Replace the contents of the Program.fs file with this:

```fsharp
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
```

The **Digit** type holds one single MNIST digit image. Note how the field is tagged with a **VectorType** attribute. This tells ML.NET to combine the 784 individual pixel columns into a single vector value.

There's also a **DigitPrediction** type which will hold a single prediction. And notice how the prediction score is actually an array? The model will generate 10 scores, one for every possible digit value. 

Also note the **CLIMutable** attribute that tells F# that we want a 'C#-style' class implementation with a default constructor and setter functions for every property. Without this attribute the compiler would generate an F#-style immutable class with read-only properties and no default constructor. The ML.NET library cannot handle immutable classes.  

Next you'll need to load the data in memory:

```fsharp
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

    // the rest of the code goes here....

    0 // return value
```

This code uses the **LoadFromTextFile** function to load the CSV data directly into memory. We call this function twice to load the training and testing datasets separately.

Now let’s build the machine learning pipeline:

```fsharp
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

// the rest of the code goes here....
```

Machine learning models in ML.NET are built with pipelines, which are sequences of data-loading, transformation, and learning components.

This pipeline has the following components:

* **MapValueToKey** which reads the **Number** column and builds a dictionary of unique values. It then produces an output column called **Label** which contains the dictionary key for each number value. We need this step because we can only train a multiclass classifier on keys. 
* **Concatenate** which converts the PixelValue vector into a single column called Features. This is a required step because ML.NET can only train on a single input column.
* **AppendCacheCheckpoint** which caches all training data at this point. This is an optimization step that speeds up the learning algorithm which comes next.
* A **SdcaMaximumEntropy** classification learner which will train the model to make accurate predictions.
* A final **MapKeyToValue** step which converts the keys in the **Label** column back to the original number values. We need this step to show the numbers when making predictions. 

With the pipeline fully assembled, we can train the model by piping the training data into the **Fit** function.

You now have a fully- trained model. So now it's time to take the test set, predict the number for each digit image, and calculate the accuracy metrics of the model:

```fsharp
// get predictions and compare them to the ground truth
let metrics = testData |> model.Transform |> context.MulticlassClassification.Evaluate

// show evaluation metrics
printfn "Evaluation metrics"
printfn "  MicroAccuracy:    %f" metrics.MicroAccuracy
printfn "  MacroAccuracy:    %f" metrics.MacroAccuracy
printfn "  LogLoss:          %f" metrics.LogLoss
printfn "  LogLossReduction: %f" metrics.LogLossReduction

// the rest of the code goes here....
```

This code pipes the test data into the **Transform** function to set up predictions for every single image in the test set. Then it pipes these predictions into the **Evaluate** function to compare these predictions to the actual labels and automatically calculate four metrics:

* **MicroAccuracy**: this is the average accuracy (=the number of correct predictions divided by the total number of predictions) for every digit in the dataset.
* **MacroAccuracy**: this is calculated by first calculating the average accuracy for each unique prediction value, and then taking the averages of those averages.
* **LogLoss**: this is a metric that expresses the size of the error in the predictions the model is making. A logloss of zero means every prediction is correct, and the loss value rises as the model makes more and more mistakes.
* **LogLossReduction**: this metric is also called the Reduction in Information Gain (RIG). It expresses the probability that the model’s predictions are better than random chance.

We can compare the micro- and macro accuracy to discover if the dataset is biased. In an unbiased set each unique label value will appear roughly the same number of times, and the micro- and macro accuracy values will be close together.

If the values are far apart, this suggests that there is some kind of bias in the data that we need to deal with. 

To wrap up, let’s use the model to make a prediction.

You will pick five arbitrary digits from the test set, run them through the model, and make a prediction for each one.

Here’s how to do it:

```fsharp
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
```

This code calls the **CreateEnumerable** function to convert the test dataview to an array of **Digit** instances. Then it picks five random digits for testing.

We then call the **CreatePredictionEngine** function to set up a prediction engine. 

The code then calls **Seq.iter** to print column headings for each of the 10 possible digit values. We then pipe the 5 test digits into another **Seq.iter**, make a prediction for each test digit, and then use a third **Seq.iter** to display the 10 prediction scores.

This will produce a table with 5 rows of test digits, and 10 columns of prediction scores. The column with the highest score represents the prediction for that particular test digit. 

That's it, you're done!

Go to your terminal and run your code:

```bash
$ dotnet run
```

What results do you get? What are your micro- and macro accuracy values? Which logloss and logloss reduction did you get?

Do you think the dataset is biased? 

What can you say about the accuracy? Is this a good model? How far away are you from the human accuracy rate? Is this a superhuman or subhuman AI? 

What did the 5 digit predictions look like? Do you understand why the model gets confused sometimes? 

Think about the code in this assignment. How could you improve the accuracy of the model even further?

Share your results in our group!
