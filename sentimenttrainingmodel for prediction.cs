#Sentiment training and prediction 

#Sample Model 

var dataPath = "sentiment.csv";
var mlContext = new MLContext();
var loader = mlContext.Data.CreateTextLoader(new[]
    {
        new TextLoader.Column("SentimentText", DataKind.String, 1),
        new TextLoader.Column("Label", DataKind.Boolean, 0),
    },
    hasHeader: true,
    separatorChar: ',');
var data = loader.Load(dataPath);
var learningPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
        .Append(mlContext.BinaryClassification.Trainers.FastTree());
var model = learningPipeline.Fit(data);

 # Prediction

var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
var prediction = predictionEngine.Predict(new SentimentData
{
    SentimentText = "I am shy "
});
Console.WriteLine("prediction: " + prediction.Prediction);