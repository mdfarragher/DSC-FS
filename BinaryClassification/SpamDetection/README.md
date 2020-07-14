# Assignment: Detect spam SMS messages

In this assignment you're going to build an app that can automatically detect spam SMS messages.

The first thing you'll need is a file with lots of SMS messages, correctly labelled as being spam or not spam. You will use a dataset compiled by Caroline Tagg in her [2009 PhD thesis](http://etheses.bham.ac.uk/253/1/Tagg09PhD.pdf). This dataset has 5574 messages.

Download the [list of messages](https://github.com/mdfarragher/DSC/blob/master/BinaryClassification/SpamDetection/spam.tsv) and save it as **spam.tsv**.

The data file looks like this:

![Spam message list](./assets/data.png)

It’s a TSV file with only 2 columns of information:

* Label: ‘spam’ for a spam message and ‘ham’ for a normal message.
* Message: the full text of the SMS message.

You will build a binary classification model that reads in all messages and then makes a prediction for each message if it is spam or ham.

Let’s get started. You need to build a new application from scratch by opening a terminal and creating a new NET Core console project:

```bash
$ dotnet new console --language F# --output SpamDetection
$ cd SpamDetection
```

Now install the following ML.NET packages:

```bash
$ dotnet add package Microsoft.ML
```

Now you are ready to add some classes. You’ll need need one to hold a labelled message, and one to hold the model predictions.

Replace the contents of the Program.fs file with this:

```fsharp
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

// the rest of the code goes here....
```

The **SpamInput** class holds one single message. Note how each field is tagged with a **LoadColumn** attribute that tells the data loading code which column to import data from.

There's also a **SpamPrediction** class which will hold a single spam prediction. There's a boolean **IsSpam**, a **Probability** value, and the **Score** the model will assign to the prediction.

Note the **CLIMutable** attribute that tells F# that we want a 'C#-style' class implementation with a default constructor and setter functions for every property. Without this attribute the compiler would generate an F#-style immutable class with read-only properties and no default constructor. The ML.NET library cannot handle immutable classes.  

Now look at the first column in the data file. Our label is a string with the value 'spam' meaning it's a spam message, and 'ham' meaning it's a normal message. 

But you're building a Binary Classifier which needs to be trained on boolean labels.

So you'll have to somehow convert the 'raw' text labels (stored in the **Verdict** field) to a boolean value. 

To set that up, you'll need a helper type:

```fsharp
/// This class describes what output columns we want to produce.
[<CLIMutable>]
type ToLabel ={
    mutable Label : bool
}

// the rest of the code goes here....
```

Note how the **ToLabel** type contains a **Label** field with the converted boolean label value. We will set up this conversion in a minute.

Also note the **mutable** keyword. By default F# types are immutable and the compiler will prevent us from assigning to any property after the type has been instantiated. The **mutable** keyword tells the compiler to create a mutable type instead and allow property assignments after construction. 

We need one more helper function before we can load the dataset. Add the following code:

```fsharp
/// Helper function to cast the ML pipeline to an estimator
let castToEstimator (x : IEstimator<_>) = 
    match x with 
    | :? IEstimator<ITransformer> as y -> y
    | _ -> failwith "Cannot cast pipeline to IEstimator<ITransformer>"

// the rest of the code goes here
```

The **castToEstimator** function takes an **IEstimator<>** argument and uses pattern matching to cast the value to an **IEstimator<ITransformer>** type. You'll see in a minute why we need this helper function. 

Now you're ready to load the training data in memory:

```fsharp
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


    // the rest of the code goes here....
```

This code uses the **LoadFromTextFile** function to load the TSV data directly into memory. The field annotations in the **SpamInput** type tell the function how to store the loaded data.

The **TrainTestSplit** function then splits the data into a training partition with 80% of the data and a test partition with 20% of the data.

Now you’re ready to start building the machine learning model:

```fsharp
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

// the rest of the code goes here....
```
Machine learning models in ML.NET are built with pipelines, which are sequences of data-loading, transformation, and learning components.

This pipeline has the following components:

* A **CustomMapping** that transforms the text label to a boolean value. We define 'spam' values as spam and anything else as normal messages.
* **FeaturizeText** which calculates a numerical value for each message. This is a required step because machine learning models cannot handle text data directly.
* A **SdcaLogisticRegression** classification learner which will train the model to make accurate predictions.

The FeaturizeText component is a very nice solution for handling text input data. The component performs a number of transformations on the text to prepare it for model training:

* Normalize the text (=remove punctuation, diacritics, switching to lowercase etc.)
* Tokenize each word.
* Remove all stopwords
* Extract Ngrams and skip-grams
* TF-IDF rescaling
* Bag of words conversion

The result is that each message is converted to a vector of numeric values that can easily be processed by the model.

Before you start training, you're going to perform a quick check to see if the dataset has enough data to reliably train a binary classification model.

We have 5574 messages which makes this a very small dataset. We'd prefer to have between 10k-100k records for reliable training. For small datasets like this one, we'll have to perform **K-Fold Cross Validation** to make sure we have enough data to work with. 

Let's set that up right now:

```fsharp
// test the full data set by performing k-fold cross validation
printfn "Performing cross validation:"
let cvResults = context.BinaryClassification.CrossValidate(data = data, estimator = castToEstimator pipeline, numberOfFolds = 5)

// report the results
cvResults |> Seq.iter(fun f -> printfn "  Fold: %i, AUC: %f" f.Fold f.Metrics.AreaUnderRocCurve)

// the rest of the code goes here....
```

This code calls the **CrossValidate** method to perform K-Fold Cross Validation on the training partition using 5 folds. Note how we call **castToEstimator** to cast the pipeline to a **IEstimator<ITransformer>** type. 

We need to do this because the **EstimatorChain** function we use every time to build the machine learning pipeline produces a type that cannot be read directly by **CrossValidate**. And the F# compiler is unable to perform the type cast for us automatically, so we need the helper function to perform the cast explicitly.   

Next, the code reports the individual AUC for each fold. For a well-balanced dataset we expect to see roughly identical AUC values for each fold. Any outliers are hints that the dataset may be unbalanced and too small to train on.

Now let's train the model and get some validation metrics:

```fsharp
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

// the rest of the code goes here
```

This code trains the model by piping the training data into the **Fit** function. Then it pipes the test data into the **Transform** function to make a prediction for every message in the validation partition. 

The code pipes these predictions into the **Evaluate** function to compare these predictions to the ground truth and calculate the following metrics:

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

When filtering spam, you definitely want to avoid false positives because you don’t want to be sending important emails into the junk folder.

You also want to avoid false negatives but they are not as bad as a false positive. Having some spam slipping through the filter is not the end of the world.

To wrap up, You’re going to create a couple of messages and ask the model to make a prediction:

```fsharp
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
```

This code calls the **CreatePredictionEngine** function to create a prediction engine. With the prediction engine set up, you can simply call **Predict** to make a single prediction.

The code creates four new test messages and calls **List.iter** to make spam predictions for each message. What’s the result going to be?

Time to find out. Go to your terminal and run your code:

```bash
$ dotnet run
```

What results do you get? What are your five AUC values from K-Fold Cross Validation and the average AUC over all folds? Are there any outliers? Are the five values grouped close together? 

What can you conclude from your cross-validation results? Do we have enough data to make reliable spam predictions? 

Based on the results of cross-validation, would you say this dataset is well-balanced? And what does this say about the metrics you should use to evaluate your model? 

Which metrics did you pick to evaluate the model? And what do the values say about the accuracy of your model? 

And what about the four test messages? Dit the model accurately predict which ones are spam?

Think about the code in this assignment. How could you improve the accuracy of the model even more? What are your best AUC values after optimization? 

Share your results in our group!
