# Assignment: Predict who survived the Titanic disaster

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

![Sinking Titanic](./assets/titanic.jpeg)

In this assignment you're going to build an app that can predict which Titanic passengers survived the disaster. You will use a decision tree classifier to make your predictions.

The first thing you will need for your app is the passenger manifest of the Titanic's last voyage. You will use the famous [Kaggle Titanic Dataset](https://github.com/sbaidachni/MLNETTitanic/tree/master/MLNetTitanic) which has data for a subset of 891 passengers.

Download the [test_data](https://github.com/mdfarragher/DSC/blob/master/BinaryClassification/TitanicPrediction/test_data.csv) and [train_data](https://github.com/mdfarragher/DSC/blob/master/BinaryClassification/TitanicPrediction/train_data.csv) files and save them to your project folder.

The training data file looks like this:

![Training data](./assets/data.jpg)

It’s a CSV file with 12 columns of information:

* The passenger identifier
* The label column containing ‘1’ if the passenger survived and ‘0’ if the passenger perished
* The class of travel (1–3)
* The name of the passenger
* The gender of the passenger (‘male’ or ‘female’)
* The age of the passenger, or ‘0’ if the age is unknown
* The number of siblings and/or spouses aboard
* The number of parents and/or children aboard
* The ticket number
* The fare paid
* The cabin number
* The port in which the passenger embarked

The second column is the label: 0 means the passenger perished, and 1 means the passenger survived. All other columns are input features from the passenger manifest.

You're gooing to build a binary classification model that reads in all columns and then predicts for each passenger if he or she survived.

Let’s get started. Here’s how to set up a new console project in NET Core:

```bash
$ dotnet new console --language F# --output TitanicPrediction
$ cd TitanicPrediction
```

Next, you need to install the correct NuGet packages:

```
$ dotnet add package Microsoft.ML
$ dotnet add package Microsoft.ML.FastTree
```

Now you are ready to add some classes. You’ll need one to hold passenger data, and one to hold your model predictions.

Replace the contents of the Program.fs file with this:

```fsharp
open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms

/// The Passenger class represents one passenger on the Titanic.
[<CLIMutable>]
type Passenger = {
    [<LoadColumn(1)>] Label : bool
    [<LoadColumn(2)>] Pclass : float32
    [<LoadColumn(4)>] Sex : string
    [<LoadColumn(5)>] RawAge : string // not a float!
    [<LoadColumn(6)>] SibSp : float32
    [<LoadColumn(7)>] Parch : float32
    [<LoadColumn(8)>] Ticket : string
    [<LoadColumn(9)>] Fare : float32
    [<LoadColumn(10)>] Cabin : string
    [<LoadColumn(11)>] Embarked : string
}

/// The PassengerPrediction class represents one model prediction. 
[<CLIMutable>]
type PassengerPrediction = {
    [<ColumnName("PredictedLabel")>] Prediction : bool
    Probability : float32
    Score : float32
}

// the rest of the code goes here...
```

The **Passenger** type holds one single passenger record. There's also a **PassengerPrediction** type which will hold a single passenger prediction. There's a boolean **Prediction**, a **Probability** value, and the **Score** the model will assign to the prediction.

Note the **CLIMutable** attribute that tells F# that we want a 'C#-style' class implementation with a default constructor and setter functions for every property. Without this attribute the compiler would generate an F#-style immutable class with read-only properties and no default constructor. The ML.NET library cannot handle immutable classes.  

Now look at the age column in the data file. It's a number, but for some passengers in the manifest the age is not known and the column is empty.

ML.NET can automatically load and process missing numeric values, but only if they are present in the CSV file as a '?'.

The Titanic datafile uses an empty string to denote missing values, so we'll have to perform a feature conversion

Notice how the age is loaded as s string into a Passenger class field called **RawAge**. 

We will process the missing values later in our app. To prepare for this, we'll need an additional helper type:

```fsharp
/// The ToAge class is a helper class for a column transformation.
[<CLIMutable>]
type ToAge = {
    mutable Age : string
}

// the rest of the code goes here...
```

The **ToAge** type will contain the converted age values. We will set up this conversion in a minute. 

Note the **mutable** keyword. By default F# types are immutable and the compiler will prevent us from assigning to any property after the type has been instantiated. The **mutable** keyword tells the compiler to create a mutable type instead and allow property assignments after construction. 

Now you're going to load the training data in memory:

```fsharp
/// file path to the train data file (assumes os = windows!)
let trainDataPath = sprintf "%s\\train_data.csv" Environment.CurrentDirectory

/// file path to the test data file (assumes os = windows!)
let testDataPath = sprintf "%s\\test_data.csv" Environment.CurrentDirectory

[<EntryPoint>]
let main argv = 

    // set up a machine learning context
    let context = new MLContext()

    // load the training and testing data in memory
    let trainData = context.Data.LoadFromTextFile<Passenger>(trainDataPath, hasHeader = true, separatorChar = ',', allowQuoting = true)
    let testData = context.Data.LoadFromTextFile<Passenger>(testDataPath, hasHeader = true, separatorChar = ',', allowQuoting = true)

    // the rest of the code goes here...

    0 // return value
```

This code calls the **LoadFromTextFile** function twice to load the training and testing datasets in memory.

ML.NET expects missing data in CSV files to appear as a ‘?’, but unfortunately the Titanic file uses an empty string to indicate an unknown age. So the first thing you need to do is replace all empty age strings occurrences with ‘?’.

Add the following code:

```fsharp
// set up a training pipeline
let pipeline = 
    EstimatorChain()

        // step 1: replace missing ages with '?'
        .Append(
            context.Transforms.CustomMapping(
                Action<Passenger, ToAge>(fun input output -> output.Age <- if String.IsNullOrEmpty(input.RawAge) then "?" else input.RawAge),
                "AgeMapping"))

        // the rest of the code goes here...
```

Machine learning models in ML.NET are built with pipelines, which are sequences of data-loading, transformation, and learning components.

The **CustomMapping** component converts empty age strings to ‘?’ values.

Now ML.NET is happy with the age values. You will now convert the string ages to numeric values and instruct ML.NET to replace any missing values with the mean age over the entire dataset.

Add the following code, and make sure you match the indentation level of the previous **Append** function exactly. Indentation is significant in F# and the wrong indentation level will lead to compiler errors:

```fsharp
// step 2: convert string ages to floats
.Append(context.Transforms.Conversion.ConvertType("Age", outputKind = DataKind.Single))

// step 3: replace missing age values with the mean age
.Append(context.Transforms.ReplaceMissingValues("Age", replacementMode = MissingValueReplacingEstimator.ReplacementMode.Mean))

// the rest of the code goes here...
```

The **ConvertType** component converts the Age column to a single-precision floating point value. And the **ReplaceMissingValues** component replaces any missing values with the mean value of all ages in the entire dataset. 

Now let's process the rest of the data columns. The Sex, Ticket, Cabin, and Embarked columns are enumerations of string values. As you've already learned, you'll need to one-hot encode them:

```fsharp
// step 4: replace string columns with one-hot encoded vectors
.Append(context.Transforms.Categorical.OneHotEncoding("Sex"))
.Append(context.Transforms.Categorical.OneHotEncoding("Ticket"))
.Append(context.Transforms.Categorical.OneHotEncoding("Cabin"))
.Append(context.Transforms.Categorical.OneHotEncoding("Embarked"))

// the rest of the code goes here...
```

The **OneHotEncoding** components take an input column, one-hot encode all values, and produce a new column with the same name holding the one-hot vectors. 

Now let's wrap up the pipeline:

```fsharp
        // step 5: concatenate everything into a single feature column 
        .Append(context.Transforms.Concatenate("Features", "Age", "Pclass", "SibSp", "Parch", "Sex", "Embarked"))

        // step 6: use a fasttree trainer
        .Append(context.BinaryClassification.Trainers.FastTree())

// the rest of the code goes here (indented back 2 levels!)...
```

The **Concatenate** component concatenates all remaining feature columns into a single column for training. This is required because ML.NET can only train on a single input column.

And the **FastTreeBinaryClassificationTrainer** is the algorithm that's going to train the model. You're going to build a decision tree classifier that uses the Fast Tree algorithm to train on the data and configure the tree.

Note the indentation level of the 'the rest of the code...' comment. Make sure that when you add the remaining code you indent this code back by two levels to match the indentation level of the **main** function.

Now all you need to do now is train the model on the entire dataset, compare the predictions with the labels, and compute a bunch of metrics that describe how accurate the model is:

```fsharp
// train the model
let model = trainData |> pipeline.Fit

// make predictions and compare with ground truth
let metrics = testData |> model.Transform |> context.BinaryClassification.Evaluate

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

// the rest of the code goes here...
```

This code pipes the training data into the **Fit** function to train the model on the entire dataset.

We then pipe the test data into the **Transform** function to set up a prediction for each passenger, and pipe these predictions into the **Evaluate** function to compare them to the label and automatically calculate evaluation metrics.

We then display the following metrics:

* **Accuracy**: this is the number of correct predictions divided by the total number of predictions.
* **AreaUnderRocCurve**: a metric that indicates how accurate the model is: 0 = the model is wrong all the time, 0.5 = the model produces random output, 1 = the model is correct all the time. An AUC of 0.8 or higher is considered good.
* **AreaUnderPrecisionRecallCurve**: an alternate AUC metric that performs better for heavily imbalanced datasets with many more negative results than positive.
* **F1Score**: this is a metric that strikes a balance between Precision and Recall. It’s useful for imbalanced datasets with many more negative results than positive.
* **LogLoss**: this is a metric that expresses the size of the error in the predictions the model is making. A logloss of zero means every prediction is correct, and the loss value rises as the model makes more and more mistakes.
* **LogLossReduction**: this metric is also called the Reduction in Information Gain (RIG). It expresses the probability that the model’s predictions are better than random chance.
* **PositivePrecision**: also called ‘Precision’, this is the fraction of positive predictions that are correct. This is a good metric to use when the cost of a false positive prediction is high.
* **PositiveRecall**: also called ‘Recall’, this is the fraction of positive predictions out of all positive cases. This is a good metric to use when the cost of a false negative is high.
* **NegativePrecision**: this is the fraction of negative predictions that are correct.
* **NegativeRecall**: this is the fraction of negative predictions out of all negative cases.

To wrap up, let's have some fun and pretend that I’m going to take a trip on the Titanic too. I will embark in Southampton and pay $70 for a first-class cabin. I travel on my own without parents, children, or my spouse. 

What are my odds of surviving?

Add the following code:

```fsharp
// set up a prediction engine
let engine = context.Model.CreatePredictionEngine model

// create a sample record
let passenger = {
    Pclass = 1.0f
    Sex = "male"
    RawAge = "48"
    SibSp = 0.0f
    Parch = 0.0f
    Ticket = "B"
    Fare = 70.0f
    Cabin = "123"
    Embarked = "S"
    Label = false // unused!
}

// make the prediction
let prediction = engine.Predict passenger

// report the results
printfn "Model prediction:"
printfn "  Prediction:  %s" (if prediction.Prediction then "survived" else "perished")
printfn "  Probability: %f" prediction.Probability
```

This code uses the **CreatePredictionEngine** method to create a prediction engine. With the prediction engine set up, you can simply call **Predict** to make a single prediction.

The code sets up a new passenger record with my information and then calls **Predict** to make a prediction about my survival chances. 

So would I have survived the Titanic disaster?

Time to find out. Go to your terminal and run your code:

```bash
$ dotnet run
```

What results do you get? What is your accuracy, precision, recall, AUC, AUCPRC, and F1 value?

Is this dataset balanced? Which metrics should you use to evaluate your model? And what do the values say about the accuracy of your model? 

And what about me? Did I survive the disaster?

Do you think a decision tree is a good choice to predict Titanic survivors? Which other algorithms could you use instead? Do they give a better result?

Share your results in our group!
