using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace FFNN
{
    /// <summary>
    /// Contains the separate problems as methods specified inside the project's rubric.
    /// </summary>
    public class RubricProblems
    {
        // Global variables.
        DataFileOperations theDtFileOps;
        Backpropagation theBackpropagation;

        // Constructor.
        public RubricProblems(DataFileOperations aDtFileOps)
        {
            theDtFileOps = aDtFileOps != null ? aDtFileOps : new DataFileOperations();
            theBackpropagation = new Backpropagation();
        }

        public void ProblemOne()
        {
            // Use randomly selected data point .txt files to set Train and Test Table.
            int theNumberOfOutputs = 10;
            theDtFileOps.TrainTable = theDtFileOps.ConvertTextToDataTable(786, "TrainTable.txt");
            theDtFileOps.TestTable = theDtFileOps.ConvertTextToDataTable(786, "TestTable.txt");

            // Indicate import status.
            Console.WriteLine("Files imported successfully.");

            // Initialize and train network.
            Network cNetwork = theBackpropagation.InitializeNetwork(theDtFileOps.TrainTable.Columns.Count - 1, 100, theNumberOfOutputs);
            theBackpropagation.InitializeMatrix();
            theBackpropagation.TrainNetwork(cNetwork, theDtFileOps.TrainTable, 0.01, 200, theNumberOfOutputs);

            Console.WriteLine();
            theBackpropagation.DisplayMatrix(theBackpropagation.ActualExpectedTrainMatrix);
            theBackpropagation.InitializeMatrix();

            // Test predictions using trained network.
            foreach (DataRow row in theDtFileOps.TestTable.Rows)
            {
                List<double> theRowAsList = theBackpropagation.ConvertDataRowToListDouble(row);
                double aPrediction = theBackpropagation.Predict(cNetwork, theRowAsList);
                Console.WriteLine("Actual={0}, Predicted={1}", theRowAsList[theRowAsList.Count - 1], aPrediction);
                theBackpropagation.ActualExpectedTestMatrix[Convert.ToInt32(aPrediction), Convert.ToInt32(theRowAsList[theRowAsList.Count - 1])]++;
            }

            // Print Confusion matrix for Train and Test sets.
            Console.WriteLine();
            theBackpropagation.DisplayMatrix(theBackpropagation.ActualExpectedTestMatrix);

            // Print EpochError values to file for plotting in other environment.
            theDtFileOps.ConvertListToTextFile(theBackpropagation.EpochErrorList, "EpochError.txt");

            // Create data table and provide designated number of columns.
            DataTable hiddenNeurons = new DataTable();
            for (int col = 0; col < 784; col++)
                hiddenNeurons.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));

            // Add all weights for each neuron in the final network hidden layer to a data table.
            foreach (var neuron in theBackpropagation.OtherFinalNetworkHiddenLayer.Neurons)
            {
                DataRow row = hiddenNeurons.NewRow();
                row.ItemArray = neuron.NeuronWeights.Cast<object>().Take(784).ToArray();
                hiddenNeurons.Rows.Add(row);
            }

            theDtFileOps.ConvertDataTableToTextFile(hiddenNeurons, "TheFinalNetworkHiddenWeights.txt");
        }

        public void ProblemTwo()
        {
            // Designated variable values from rubric.
            int numberOfOutputs = 784;
            int numberOfHidden = 100;

            // Import datasets.
            theDtFileOps.TrainTable = theDtFileOps.ConvertTextToDataTable(786, "TrainTable.txt");
            theDtFileOps.TestTable = theDtFileOps.ConvertTextToDataTable(786, "TestTable.txt");

            // Indicate import status.
            Console.WriteLine("Files imported successfully.");

            // Initialize and train network.
            Network aNetwork = theBackpropagation.InitializeNetwork(theDtFileOps.TrainTable.Columns.Count - 1, numberOfHidden, numberOfOutputs);
            theBackpropagation.TrainNetworkWithJ2Error(aNetwork, theDtFileOps.TrainTable, .001, 20, numberOfOutputs);

            double[] sumLossArray = new double[10];
            var sumLoss = 0.0;

            int count = 0;

            // Create data table and provide designated number of columns.
            DataTable expectedImageMatrices = new DataTable();
            for (int col = 0; col < numberOfOutputs; col++)
                expectedImageMatrices.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));

            // Test operations.
            foreach (DataRow row in theDtFileOps.TestTable.Rows)
            {
                List<double> theRowAsList = theBackpropagation.ConvertDataRowToListDouble(row);
                List<double> aPrediction = theBackpropagation.PredictForJ2(aNetwork, theRowAsList);

                int theActualValue = Convert.ToInt32(row.ItemArray[785]);
                var aRow = row.ItemArray.AsEnumerable().Take(784);
                List<double> expected = aRow.Select(x => Convert.ToDouble(x)).ToList();

                if (count <= 4)
                {
                    object[] objArray = aPrediction.Cast<object>().ToArray();
                    DataRow temp = expectedImageMatrices.NewRow();
                    temp.ItemArray = objArray;
                    expectedImageMatrices.Rows.Add(temp);
                }

                List<double> theDifferenceList = new List<double>();
                for (int i = 0; i < expected.Count; i++)
                {
                    theDifferenceList.Add(Math.Pow(expected[i] - aPrediction[i], 2));
                }

                sumLoss += theBackpropagation.J2LossFunctionCalculation(theDifferenceList);
                sumLossArray[theActualValue] += theBackpropagation.J2LossFunctionCalculation(theDifferenceList);
            }

            Console.WriteLine("Sum Loss: {0}", sumLoss);
            foreach (var item in sumLossArray)
            {
                Console.Write(item + " ");
            }

            List<double> sumLossTestTrainValue = new List<double>
            {
                theBackpropagation.TrainSumLossFinal,
                sumLoss
            };

            // Print outputs for plotting/graphing in Matlab environment.
            theDtFileOps.ConvertListToTextFile(sumLossTestTrainValue, "theSumLoss.txt");
            theDtFileOps.ConvertListToTextFile(theBackpropagation.TrainSumLossFinalArray.ToList(), "trainSumLossArray.txt");
            theDtFileOps.ConvertListToTextFile(sumLossArray.ToList(), "testSumLossArray.txt");
            theDtFileOps.ConvertListToTextFile(theBackpropagation.CompleteSumLossesOfTrain, "sumLossesForEpochs.txt");
            theDtFileOps.ConvertDataTableToTextFile(expectedImageMatrices, "imageMatrices.txt");

            // Add all weights for each neuron in the final network hidden layer to a list of double.
            var NeuronWeights = new List<double>();
            foreach (var neuron in theBackpropagation.FinalNetworkHiddenLayer.Neurons)
            {
                foreach (var item in neuron.NeuronWeights)
                {
                    NeuronWeights.Add(item);
                }
            }

            theDtFileOps.ConvertListToTextFile(NeuronWeights, "FinalNetworkHiddenWeights.txt");
        }

        public void ProblemThree()
        {
            // Designated variable values from rubric.
            int numberOfOutputs = 784;
            int numberOfHidden = 100;

            // Import datasets.
            theDtFileOps.TrainTable = theDtFileOps.ConvertTextToDataTable(786, "TrainTable.txt");
            theDtFileOps.TestTable = theDtFileOps.ConvertTextToDataTable(786, "TestTable.txt");

            // Indicate import status.
            Console.WriteLine("Files imported successfully.");

            // Initialize and train network.
            Network aNetwork = theBackpropagation.InitializeNetwork(theDtFileOps.TrainTable.Columns.Count - 1, numberOfHidden, numberOfOutputs);
            theBackpropagation.TrainNetworkWithJ2Error(aNetwork, theDtFileOps.TrainTable, .001, 15, numberOfOutputs);

            double[] sumLossArray = new double[10];
            var sumLoss = 0.0;

            // Create data table and provide designated number of columns.
            DataTable expectedImageMatrices = new DataTable();
            for (int col = 0; col < numberOfOutputs; col++)
                expectedImageMatrices.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));

            // Test operations.
            foreach (DataRow row in theDtFileOps.TestTable.Rows)
            {
                List<double> theRowAsList = theBackpropagation.ConvertDataRowToListDouble(row);
                List<double> aPrediction = theBackpropagation.PredictForJ2(aNetwork, theRowAsList);

                int theActualValue = Convert.ToInt32(row.ItemArray[785]);
                var aRow = row.ItemArray.AsEnumerable().Take(784);
                List<double> expected = aRow.Select(x => Convert.ToDouble(x)).ToList();

                List<double> theDifferenceList = new List<double>();
                for (int i = 0; i < expected.Count; i++)
                {
                    theDifferenceList.Add(Math.Pow(expected[i] - aPrediction[i], 2));
                }

                sumLoss += theBackpropagation.J2LossFunctionCalculation(theDifferenceList);
                sumLossArray[theActualValue] += theBackpropagation.J2LossFunctionCalculation(theDifferenceList);
            }

            // Create data table and provide designated number of columns.
            DataTable trainHiddenNeurons = new DataTable();
            for (int col = 0; col < numberOfOutputs; col++)
                trainHiddenNeurons.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));

            // Add all weights for each neuron in the final network hidden layer to a data table.
            foreach (var neuron in theBackpropagation.FinalNetworkHiddenLayer.Neurons)
            {
                DataRow row = trainHiddenNeurons.NewRow();
                row.ItemArray = neuron.NeuronWeights.Cast<object>().Take(784).ToArray();
                trainHiddenNeurons.Rows.Add(row);
            }

            theDtFileOps.ConvertDataTableToTextFile(trainHiddenNeurons, "TheTrainFinalNetworkHiddenWeights.txt");

            Network bNetwork = theBackpropagation.InitializeNetwork(theDtFileOps.TrainTable.Columns.Count - 1, numberOfHidden, 10);
            bNetwork.Layers[0] = theBackpropagation.FinalNetworkHiddenLayer;
            theBackpropagation.InitializeMatrix();
            theBackpropagation.TrainNetworkForHW4P1(bNetwork, theDtFileOps.TrainTable, .0005, 100, 10);

            Console.WriteLine();
            theBackpropagation.DisplayMatrix(theBackpropagation.ActualExpectedTrainMatrix);
            theBackpropagation.InitializeMatrix();

            // Test predictions using trained network.
            foreach (DataRow row in theDtFileOps.TestTable.Rows)
            {
                List<double> theRowAsList = theBackpropagation.ConvertDataRowToListDouble(row);
                double aPrediction = theBackpropagation.Predict(bNetwork, theRowAsList);

                List<double> aPredictionList = theBackpropagation.PredictForJ2(aNetwork, theRowAsList);
                object[] objArray = aPredictionList.Cast<object>().ToArray();
                DataRow temp = expectedImageMatrices.NewRow();
                temp.ItemArray = objArray;
                expectedImageMatrices.Rows.Add(temp);

                Console.WriteLine("Actual={0}, Predicted={1}", theRowAsList[theRowAsList.Count - 1], aPrediction);
                theBackpropagation.ActualExpectedTestMatrix[Convert.ToInt32(aPrediction), Convert.ToInt32(theRowAsList[theRowAsList.Count - 1])]++;
            }

            // Print Confusion matrix for Train and Test sets.
            Console.WriteLine();
            theBackpropagation.DisplayMatrix(theBackpropagation.ActualExpectedTestMatrix);

            // Create data table and provide designated number of columns.
            DataTable hiddenNeurons = new DataTable();
            for (int col = 0; col < numberOfOutputs; col++)
                hiddenNeurons.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));

            // Add all weights for each neuron in the final network hidden layer to a data table.
            foreach (var neuron in theBackpropagation.OtherFinalNetworkHiddenLayer.Neurons)
            {
                DataRow row = hiddenNeurons.NewRow();
                row.ItemArray = neuron.NeuronWeights.Cast<object>().Take(784).ToArray();
                hiddenNeurons.Rows.Add(row);
            }

            theDtFileOps.ConvertDataTableToTextFile(hiddenNeurons, "TheFinalNetworkHiddenWeights.txt");

            // Print EpochError values to file for plotting in other environment.
            theDtFileOps.ConvertListToTextFile(theBackpropagation.EpochErrorList, "EpochError.txt");
            theDtFileOps.ConvertDataTableToTextFile(expectedImageMatrices, "imageMatrices.txt");
        }

        public void ProblemFour()
        {
            // Designated variable values from rubric.
            int numberOfOutputs = 784;
            int numberOfHidden = 100;

            // Import datasets.
            theDtFileOps.TrainTable = theDtFileOps.ConvertTextToDataTable(786, "TrainTable.txt");
            theDtFileOps.TestTable = theDtFileOps.ConvertTextToDataTable(786, "TestTable.txt");

            // Indicate import status.
            Console.WriteLine("Files imported successfully.");

            // Initialize and train network.
            Network aNetwork = theBackpropagation.InitializeNetwork(theDtFileOps.TrainTable.Columns.Count - 1, numberOfHidden, numberOfOutputs);
            theBackpropagation.TrainNetworkWithJ2Error(aNetwork, theDtFileOps.TrainTable, .001, 15, numberOfOutputs);

            double[] sumLossArray = new double[10];
            var sumLoss = 0.0;

            // Create data table and provide designated number of columns.
            DataTable expectedImageMatrices = new DataTable();
            for (int col = 0; col < numberOfOutputs; col++)
                expectedImageMatrices.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));

            // Test operations.
            foreach (DataRow row in theDtFileOps.TestTable.Rows)
            {
                List<double> theRowAsList = theBackpropagation.ConvertDataRowToListDouble(row);
                List<double> aPrediction = theBackpropagation.PredictForJ2(aNetwork, theRowAsList);

                int theActualValue = Convert.ToInt32(row.ItemArray[785]);
                var aRow = row.ItemArray.AsEnumerable().Take(784);
                List<double> expected = aRow.Select(x => Convert.ToDouble(x)).ToList();

                List<double> theDifferenceList = new List<double>();
                for (int i = 0; i < expected.Count; i++)
                {
                    theDifferenceList.Add(Math.Pow(expected[i] - aPrediction[i], 2));
                }

                sumLoss += theBackpropagation.J2LossFunctionCalculation(theDifferenceList);
                sumLossArray[theActualValue] += theBackpropagation.J2LossFunctionCalculation(theDifferenceList);
            }

            Network bNetwork = theBackpropagation.InitializeNetwork(theDtFileOps.TrainTable.Columns.Count - 1, numberOfHidden, 10);
            bNetwork.Layers[0] = theBackpropagation.FinalNetworkHiddenLayer;
            theBackpropagation.InitializeMatrix();
            theBackpropagation.TrainNetworkForHW4P2(bNetwork, theDtFileOps.TrainTable, .0005, 200, 10);

            Console.WriteLine();
            theBackpropagation.DisplayMatrix(theBackpropagation.ActualExpectedTrainMatrix);
            theBackpropagation.InitializeMatrix();

            // Test predictions using trained network.
            foreach (DataRow row in theDtFileOps.TestTable.Rows)
            {
                List<double> theRowAsList = theBackpropagation.ConvertDataRowToListDouble(row);
                double aPrediction = theBackpropagation.Predict(bNetwork, theRowAsList);

                List<double> aPredictionList = theBackpropagation.PredictForJ2(aNetwork, theRowAsList);
                object[] objArray = aPredictionList.Cast<object>().ToArray();
                DataRow temp = expectedImageMatrices.NewRow();
                temp.ItemArray = objArray;
                expectedImageMatrices.Rows.Add(temp);

                Console.WriteLine("Actual={0}, Predicted={1}", theRowAsList[theRowAsList.Count - 1], aPrediction);
                theBackpropagation.ActualExpectedTestMatrix[Convert.ToInt32(aPrediction), Convert.ToInt32(theRowAsList[theRowAsList.Count - 1])]++;
            }

            // Create data table and provide designated number of columns.
            DataTable hiddenNeurons = new DataTable();
            for (int col = 0; col < numberOfOutputs; col++)
                hiddenNeurons.Columns.Add(new DataColumn("Column" + (col + 1).ToString()));

            // Add all weights for each neuron in the final network hidden layer to a data table.
            foreach (var neuron in theBackpropagation.OtherFinalNetworkHiddenLayer.Neurons)
            {
                DataRow row = hiddenNeurons.NewRow();
                row.ItemArray = neuron.NeuronWeights.Cast<object>().Take(784).ToArray();
                hiddenNeurons.Rows.Add(row);
            }

            theDtFileOps.ConvertDataTableToTextFile(hiddenNeurons, "TheFinalNetworkHiddenWeights.txt");

            // Print Confusion matrix for Train and Test sets.
            Console.WriteLine();
            theBackpropagation.DisplayMatrix(theBackpropagation.ActualExpectedTestMatrix);

            // Print EpochError values to file for plotting in other environment.
            theDtFileOps.ConvertListToTextFile(theBackpropagation.EpochErrorList, "EpochError.txt");
            theDtFileOps.ConvertDataTableToTextFile(expectedImageMatrices, "imageMatrices.txt");
        }
    }
}
