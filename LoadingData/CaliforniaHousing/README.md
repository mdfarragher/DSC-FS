# Assignment: Load California housing data

In this assignment you're going to build an app that can load a dataset with the prices of houses in California. The data is not ready for training yet and needs a bit of processing. 

The first thing you'll need is a data file with house prices. The data from the 1990 California cencus has exactly what we need. 

Download the [California 1990 housing census](https://github.com/mdfarragher/DSC/blob/master/LoadingData/CaliforniaHousing/california_housing.csv) and save it as **california_housing.csv**. 

This is a CSV file with 17,000 records that looks like this:
￼
![Data File](./assets/data.png)

The file contains information on 17k housing blocks all over the state of California:

* Column 1: The longitude of the housing block
* Column 2: The latitude of the housing block
* Column 3: The median age of all the houses in the block
* Column 4: The total number of rooms in all houses in the block
* Column 5: The total number of bedrooms in all houses in the block
* Column 6: The total number of people living in all houses in the block
* Column 7: The total number of households in all houses in the block
* Column 8: The median income of all people living in all houses in the block
* Column 9: The median house value for all houses in the block

We can use this data to train an app to predict the value of any house in and outside the state of California. 

Unfortunately we cannot train on this dataset directly. The data needs to be processed first to make it suitable for training. This is what you will do in this assignment. 

Let's get started. 

In these assignments you will not be using the code in Github. Instead, you'll be building all the applications 100% from scratch. So please make sure to create a new folder somewhere to hold all of your assignments.

Now please open a console window. You are going to create a new subfolder for this assignment and set up a blank console application:

```bash
$ dotnet new console --language F# --output LoadingData
$ cd LoadingData
```

Also make sure to copy the dataset file(s) into this folder because the code you're going to type next will expect them here.  

Now install the following packages

```bash
$ dotnet add package Microsoft.ML
$ dotnet add package FSharp.Plotly
```

**Microsoft.ML** is the Microsoft machine learning package. We will use to build all our applications in this course. And **FSharp.Plotly** is an advanced scientific plotting library.

Now you are ready to add types. You’ll need one type to hold all the information for a single housing block.

Edit the Program.fs file with Visual Studio Code and add the following code:

```fsharp
open System
open Microsoft.ML
open Microsoft.ML.Data
open FSharp.Plotly

/// The HouseBlockData class holds one single housing block data record.
[<CLIMutable>]
type HouseBlockData = {
    [<LoadColumn(0)>] Longitude : float32
    [<LoadColumn(1)>] Latitude : float32
    [<LoadColumn(2)>] HousingMedianAge : float32
    [<LoadColumn(3)>] TotalRooms : float32
    [<LoadColumn(4)>] TotalBedrooms : float32
    [<LoadColumn(5)>] Population : float32
    [<LoadColumn(6)>] Households : float32
    [<LoadColumn(7)>] MedianIncome : float32
    [<LoadColumn(8)>] MedianHouseValue : float32
}
```

The **HouseBlockData** class holds all the data for one single housing block. Note that we're loading each column as a 32-bit floating point number, and that every field is tagged with a **LoadColumn** attribute that will tell the CSV data loading code which column to import data from.

We also need the **CLIMutable** attribute to tell F# that we want a 'C#-style' class implementation with a default constructor and setters functions for every property. Without this attribute the compiler would generate an F#-style immutable class with read-only properties and no default constructor. The ML.NET library cannot handle immutable classes.  

Next you need to load the data in memory:

```fsharp
/// file paths to data files (assumes os = windows!)
let dataPath = sprintf "%s\\california_housing.csv" Environment.CurrentDirectory

[<EntryPoint>]
let main argv =

    // create the machine learning context
    let context = new MLContext()

    // load the dataset
    let data = context.Data.LoadFromTextFile<HouseBlockData>(dataPath, hasHeader = true, separatorChar = ',')

    // the rest of the code goes here...

    0 // return value
```

This code sets up the **main** function which is the main entry point of the application. The code calls the **LoadFromTextFile** method to load the CSV data in memory. Note the **HouseBlockData** type argument that tells the method which class to use to load the data.

Also note that **dataPath** uses a Windows path separator to access the data file. Change this accordingly if you're using OS/X or Linux. 

So now we have the data in memory. Let's plot the median house value as a function of median income and see what happens. 

Add the following code:

```fsharp
// get an array of housing data
let houses = context.Data.CreateEnumerable<HouseBlockData>(data, reuseRowObject = false)

// plot median house value by median income
Chart.Point(houses |> Seq.map(fun h -> (h.MedianIncome, h.MedianHouseValue))) 
    |> Chart.withX_AxisStyle "Median income"
    |> Chart.withY_AxisStyle "Median house value"
    |> Chart.Show

// the rest of the code goes here
```

The housing data is stored in memory as a data view, but we want to work with the **HouseBlockData** records directly. So we call **CreateEnumerable** to convert the data view to an enumeration of **HouseDataBlock** instances.

The **Chart.Point** method then sets up a scatterplot. We pipe the **houses** enumeration into the **Seq.map** function and project a tuple for every housing block. The tuples contain the median income and median house value for every block, and **Chart.Point** will use these as X- and Y coordinates.

The **Chart.withX_AxisStyle** and **Chart.withY_AxisStyle** functions set the chart axis titles, and **Chart.Show** renders the chart on screen. Your app will open a web browser and display the chart there. 

This is a good moment to save your work ;) 

We're now ready to run the app. Open a Powershell terminal and make sure you're in the project folder. Then type the following: 

```bash
$ dotnet build
```

This will build the project and populate the bin folder. 

Then type the following:

```bash
$ dotnet run
```

Your app will run and open the chart in a new browser window. It should look like this:

![Median house value by median income](./assets/plot.png)

As the median income level increases, the median house value also increases. There's still a big spread in the house values, but a vague 'cigar' shape is visible which suggests a linear relationship between these two variables.

But look at the horizontal line at 500,000. What's that all about? 

This is what **clipping** looks like. The creator of this dataset has clipped all housing blocks with a median house value above $500,000 back down to $500,000. We see this appear in the graph as a horizontal line that disrupts the linear 'cigar' shape. 

Let's start by using **data scrubbing** to get rid of these clipped records. Add the following code:

```fsharp
// keep only records with a median house value < 500,000
let data = context.Data.FilterRowsByColumn(data, "MedianHouseValue", upperBound = 499999.0)

// the rest of the code goes here...
```

The **FilterRowsByColumn** method will keep only those records with a median house value of 500,000 or less, and remove all other records from the dataset.  

Move your plotting code BELOW this code fragment and run your app again. 

Did this fix the problem? Is the clipping line gone?

Now let's take a closer look at the CSV file. Notice how all the columns are numbers in the range of 0..3000, but the median house value is in a range of 0..500,000. 

Remember when we talked about training data science models that we discussed having all data in a similar range?

So let's fix that now by using **data scaling**. We're going to divide the median house value by 1,000 to bring it down to a range more in line with the other data columns. 

Start by adding the following type:

```fsharp
/// The ToMedianHouseValue class is used in a column data conversion.
[<CLIMutable>]
type ToMedianHouseValue = {
    mutable NormalizedMedianHouseValue : float32
}
```

And then add the following code at the bottom of your **main** function:

```fsharp
// build a data loading pipeline
let pipeline = 
    EstimatorChain()

        // step 1: divide the median house value by 1000
        .Append(
            context.Transforms.CustomMapping(
                Action<HouseBlockData, ToMedianHouseValue>(fun input output -> output.NormalizedMedianHouseValue <- input.MedianHouseValue / 1000.0f),
                "MedianHouseValue"))

// the rest of the code goes here...
```

Machine learning models in ML.NET are built with pipelines which are sequences of data-loading, transformation, and learning components.

This pipeline has only one component:

* **CustomMapping** which takes the median house values, divides them by 1,000 and stores them in a new column called **NormalizedMedianHouseValue**. Note that we need the new **ToMedianHouseValue** type to access this new column in code.

Also note the **mutable** keyword in the type definition for **ToMedianHouseValue**. By default F# types are immutable and the compiler will prevent us from assigning to any property after the type has been instantiated. The **mutable** keyword tells the compiler to create a mutable type instead and allow property assignments after construction. 

If we had left out the keyword, the **output.NormalizedMedianHouseValue = ...** line would fail.

Now let's see if the conversion worked. Add the following code at the bottom of the **main** function:

```fsharp
// get a 10-record preview of the transformed data
let model = data |> pipeline.Fit
let preview = (data |> model.Transform).Preview(maxRows = 10)

// show the preview
preview.ColumnView |> Seq.iter(fun c ->
    printf "%-30s|" c.Column.Name
    preview.RowView |> Seq.iter(fun r -> printf "%10O|" r.Values.[c.Column.Index].Value)
    printfn "")

// the rest of the code goes here...
```

The **pipeline.Fit** method sets up the pipeline, creates a data science model and stores it in the **model** variable. The **model.Transform** method then runs the dataset through the pipeline and creates predictions for every housing block. And finally the **Preview** method extracts a 10-row preview from the collection of predictions.

Next, we use **Seq.iter** to enumerate every column in the preview. We print the column name and then use a second **Seq.iter** to show all the preview values in this column.

This will print a transposed view of the preview data with the columns stacked vertically and the rows stacked horizontally. Flipping the preview makes it easier to read, despite the very long column names. 

Now run your code. 

Find the MedianHouseValue and NormalizedMedianHouseValue columns in the output. Do they contain the correct values? Does the normalized column contain the oroginal house values divided by 1,000? 

Now let's fix the latitude and longitude. We're reading them in directly, but remember that we discussed how **Geo data should always be binned, one-hot encoded, and crossed?** 

Let's do that now. Add the following types at the top of the file:

```fsharp
/// The ToLocation class is used in a column data conversion.
[<CLIMutable>]
type FromLocation = {
    EncodedLongitude : float32[]
    EncodedLatitude : float32[]
}

/// The ToLocation class is used in a column data conversion.
[<CLIMutable>]
type ToLocation = {
    mutable Location : float32[]
}
```

Note the **mutable** keyword again, which indicates that we're going to modify the **Location** property of the **ToLocation** type after construction. 

We will use these types in the next code snippet.

Now scroll down to the bottom of the **main** function and add the following code just before the final line that retuns a zero return value:

```fsharp
// step 2: bin, encode, and cross the longitude and latitude
let pipeline2 = 
    pipeline
        .Append(context.Transforms.NormalizeBinning("BinnedLongitude", "Longitude", maximumBinCount = 10))

        // step 3: bin the latitude
        .Append(context.Transforms.NormalizeBinning("BinnedLatitude", "Latitude", maximumBinCount = 10))

        // step 4: one-hot encode the longitude
        .Append(context.Transforms.Categorical.OneHotEncoding("EncodedLongitude", "BinnedLongitude"))

        // step 5: one-hot encode the latitude
        .Append(context.Transforms.Categorical.OneHotEncoding("EncodedLatitude", "BinnedLatitude"))

        // step 6: cross the longitude and latitude vectors
        .Append(
            context.Transforms.CustomMapping(
                Action<FromLocation, ToLocation>(fun input output -> 
                    output.Location <- [|   for x in input.EncodedLongitude do
                                                for y in input.EncodedLatitude do
                                                    x * y |] ),
                "Location"))

// the rest of the code goes here...
```

Note how we're extending the data loading pipeline with extra components. The new components are:

* Two **NormalizeBinning** components that bin the longitude and latitude values into 10 bins

* Two **OneHotEncoding** components that one-hot encode the longitude and latitude bins

* One **CustomMapping** component that multiples (crosses) the longitude and latitude vectors to create a feature cross: a 100-element vector with all zeroes except for a single '1' value.

Note how the custom mapping uses two nested for-loops inside the **[| ... |]** array brackets. This sets up an inline enumerator that multiples the two longitude and latitude vectors and produces a 1-dimensional array with 100 elements. 

Let's see if this worked. Add the following code to the bottom of the **main** function:

```fsharp
// get a 10-record preview of the transformed data
let model = data |> pipeline2.Fit
let preview = (data |> model.Transform).Preview(maxRows = 10)

// show the preview
preview.ColumnView |> Seq.iter(fun c ->
    printf "%-30s|" c.Column.Name
    preview.RowView |> Seq.iter(fun r -> printf "%10O|" r.Values.[c.Column.Index].Value)
    printfn "")

// the rest of the code goes here...
```

This is the same code you used previously to create predictions, get a preview, and display the preview on the console. But now you're using **pipeline2** instead.

Now run your app. 

What does the data look like now? Is the list of columns correct? Are all the dropped columns gone? And is the new **Location** column present?

You should see the new **Location** column, but the code can't display its contents properly. 

So let's fix that. Add the following code to display all the individual values in the **Location** vector:

```fsharp
// show the dense vector
preview.RowView |> Seq.iter(fun r ->
    let vector = r.Values.[r.Values.Length-1].Value :?> VBuffer<float32>
    vector.DenseValues() |> Seq.iter(fun v -> printf "%i" (int v))
    printfn "")
```

We use **Seq.iter** to enumerate every row in the preview. And note the **:?>** operator which casts the value to a **VBuffer** of floats. With this casted value we can access the **DenseValues** property which is a float array of all the elements in the vector. So we pipe that property into a second **Seq.iter** to print the values.

Now run your app. What do you see? Did it work? Are there 100 digits in the **Location** column? And is there only a single '1' digit in each row? 

Post your results in our group. 