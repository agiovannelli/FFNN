using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace FFNN
{
    /// <summary>
    /// Class responsible for all backpropagation methods.
    /// </summary>
    public class Backpropagation
    {
        #region HW3 Problem 1 Properties

        // For use during train.
        public List<double> EpochErrorList { get; set; }

        public int[,] ActualExpectedTrainMatrix { get; set; }

        public int[,] ActualExpectedTestMatrix { get; set; }

        #endregion

        #region HW3 Problem 2 Properties

        public double TrainSumLossFinal { get; set; }

        public double[] TrainSumLossFinalArray { get; set; }

        public List<double> CompleteSumLossesOfTrain { get; set; }

        #endregion

        #region HW4 Problem 1 Properties

        public Layer FinalNetworkHiddenLayer { get; set; }

        public Layer OtherFinalNetworkHiddenLayer { get; set; }

        #endregion

        #region Public Methods

        // Initialize a network data type.
        public Network InitializeNetwork(int theNumberOfInputs, int theNumberOfHiddenLayers, int theNumberOfOutputs)
        {
            // Initialize network.
            Network aNetwork = new Network(new List<Layer>());

            // Add hidden layer.
            aNetwork.Layers.Add(CreateLayer(theNumberOfInputs, theNumberOfHiddenLayers));

            // Add output layer.
            aNetwork.Layers.Add(CreateLayer(theNumberOfHiddenLayers, theNumberOfOutputs));

            return aNetwork;
        }

        // Calculates the neuron activation for a given input.
        public double CalculateActivation(List<double> theWeights, List<double> theInputs)
        {
            // Initialize the activation value to the bias weight (last weight w/o result).
            double aActivation = theWeights[theWeights.Count - 2];

            // Neuron activation is calculated in for-loop via weighted sum of inputs.
            for (int idx = 0; idx < theWeights.Count - 2; idx++)
                aActivation += (theWeights[idx] * theInputs[idx]);

            return aActivation;
        }

        // Calculate the transfer neuron activation.
        public double CalculateTransfer(double theActivation)
        {
            return Math.Tanh(theActivation);
            //return (1 / 1 + Math.Pow(Math.E, -theActivation));
        }

        // Forward propagate the input to a network output.
        public List<double> ForwardPropagate(Network theNetwork, List<double> theInputRow)
        {
            // Initialize inputs.
            List<double> aInputs = theInputRow;

            // Iterate through each layer.
            foreach (var layer in theNetwork.Layers)
            {
                // Create new inputs list.
                List<double> aNewInputs = new List<double>();

                // Collect outputs for a layers neurons in aNewInputs, which is used as inputs for next layer.
                foreach (var neuron in layer.Neurons)
                {
                    double aActivation = CalculateActivation(neuron.NeuronWeights, aInputs);
                    neuron.NeuronOutput = CalculateTransfer(aActivation);
                    aNewInputs.Add(neuron.NeuronOutput);
                }

                aInputs = aNewInputs;
            }

            return aInputs;
        }

        // Calculate the derivative of a neuron output.
        public double CalculateTransferDerivative(double theOutputValue)
        {
            double squareTanh = Math.Pow(Math.Tanh(theOutputValue), 2);
            return 1 - squareTanh;
            //return (theOutputValue * (1 - theOutputValue));
        }

        // Backpropagate error and store in neuron delta.
        public void BackwardPropagateError(Network theNetwork, List<double> theExpectedOutput)
        {
            for (int idx = theNetwork.Layers.Count-1; idx >= 0; idx--)
            {
                Layer aTempLayer = theNetwork.Layers[idx];
                List<double> aErrorsList = new List<double>();

                if (idx != theNetwork.Layers.Count-1)
                {
                    for (int j = 0; j < aTempLayer.Neurons.Count; j++)
                    {
                        double aError = 0.0;

                        foreach (var neuron in theNetwork.Layers[idx + 1].Neurons)
                        {
                            aError += (neuron.NeuronWeights[j] * neuron.NeuronDelta);
                        }

                        aErrorsList.Add(aError);
                    }
                }
                else
                {
                    for (int j = 0; j < aTempLayer.Neurons.Count; j++)
                    {
                        Neuron aNeuron = aTempLayer.Neurons[j];
                        aErrorsList.Add(theExpectedOutput[j] - aNeuron.NeuronOutput);
                    }
                }

                for (int j = 0; j < aTempLayer.Neurons.Count; j++)
                {
                    Neuron aNeuron = aTempLayer.Neurons[j];
                    aNeuron.NeuronDelta = aErrorsList[j] * CalculateTransferDerivative(aNeuron.NeuronOutput);
                }
            }
        }

        // Update network weights with error.
        public void UpdateWeights(Network theNetwork, List<double> theRow, double theLearnRate)
        {
            double momentum = 0.0;
            for (int idx = 0; idx < theNetwork.Layers.Count; idx++)
            {
                List<double> inputs = theRow.Take(theRow.Count).ToList();

                if (idx != 0)
                {
                    inputs = new List<double>();
                    foreach (var neuron in theNetwork.Layers[idx - 1].Neurons)
                    {
                        inputs.Add(neuron.NeuronOutput);
                    }
                }

                foreach (var neuron in theNetwork.Layers[idx].Neurons)
                {
                    for (int j = 0; j < inputs.Count-2; j++)
                    {
                        // Float value is a momentum value.
                        neuron.NeuronWeights[j] += theLearnRate * neuron.NeuronDelta * inputs[j] + momentum;
                    }

                    // Bias weight.
                    neuron.NeuronWeights[neuron.NeuronWeights.Count - 2] += theLearnRate * neuron.NeuronDelta + momentum;
                }
            }
        }

        // Train the network for a given number of epochs and learn rate with a provided data set.
        public void TrainNetwork(Network theNetwork, DataTable theTrainDT, double theLearnRate, int theNumOfEpochs, int theNumberOutputs)
        {
            Random rand = new Random();
            List<double> ErrorList = new List<double>();
            for (int idx = 0; idx < theNumOfEpochs; idx++)
            {
                var numberCorrect = 0.0;
                //var shuffleTable = theTrainDT.AsEnumerable().OrderBy(r => rand.Next()).Take(3000).CopyToDataTable();

                foreach (DataRow row in theTrainDT.Rows)
                {
                    int theGuess = 0;
                    List<double> theRowAsList = ConvertDataRowToListDouble(row);
                    var outputs = ForwardPropagate(theNetwork, theRowAsList);

                    List<double> expected = new List<double>();
                    for (int j = 0; j < theNumberOutputs; j++)
                    {
                        expected.Add(0);
                    }

                    var lastRowValue = Convert.ToInt32(row[row.ItemArray.Length - 1]);
                    expected[lastRowValue] = 1;

                    theGuess = outputs.IndexOf(outputs.Max());
                    if (theGuess == lastRowValue)
                    {
                        numberCorrect++;
                    }

                    ActualExpectedTrainMatrix[theGuess, lastRowValue]++;

                    BackwardPropagateError(theNetwork, expected);
                    UpdateWeights(theNetwork, theRowAsList, theLearnRate);
                }
                double Ratio = (Convert.ToDouble(numberCorrect) / Convert.ToDouble(theTrainDT.Rows.Count));
                ErrorList.Add(1 - Ratio);
                Console.WriteLine("Epoch={0}, lrate={1}, HitRate={2}", idx, theLearnRate, Ratio);

                // Give performance of final network for plotting later.
                if (idx == theNumOfEpochs - 1)
                {
                    OtherFinalNetworkHiddenLayer = theNetwork.Layers[0];
                }
            }

            EpochErrorList = ErrorList;
        }

        // Make a prediction for a given network.
        public double Predict(Network theNetwork, List<double> theInputRow)
        {
            List<double> aOutputsList = ForwardPropagate(theNetwork, theInputRow);
            return ReturnMaxIdx(aOutputsList);
        }

        // "Predict function" for J2. Really just calls ForwardPropagate... but this is more understandable.
        public List<double> PredictForJ2(Network theNetwork, List<double> theInputRow)
        {
            return ForwardPropagate(theNetwork, theInputRow);
        }

        // Convert a given data row object to a list of double.
        public List<double> ConvertDataRowToListDouble(DataRow theRow)
        {
            List<double> theList = new List<double>();

            foreach (var item in theRow.ItemArray)
            {
                theList.Add(Convert.ToDouble(item));
            }

            return theList;
        }

        // Creates layer with designated number of neurons with designated number of weights in each neuron.
        public Layer CreateLayer(int numberOfNeurons, int numberOfWeights)
        {
            // Initialize the layer.
            Layer aLayer = new Layer(new List<Neuron>());

            // Initialize Random variable to remove time-dependent issues.
            Random theRandomDouble = new Random();

            for (int idx1 = 0; idx1 < numberOfWeights; idx1++)
            {
                List<double> theDoubleList = new List<double>();

                for (int idx2 = 0; idx2 < numberOfNeurons; idx2++)
                {
                    //theDoubleList.Add((theRandomDouble.NextDouble() * 2.0 - 1.0));
                    // was .1.
                    theDoubleList.Add((theRandomDouble.NextDouble() * 0.1));
                }

                aLayer.Neurons.Add(new Neuron(theDoubleList));
            }

            // Return result.
            return aLayer;
        }

        // Return the max index value for a List<double> collection.
        public double ReturnMaxIdx(List<double> intList)
        {
            double MaxIDX = -1;
            double Max = -1;

            for (int i = 0; i < intList.Count; i++)
            {
                if (i == 0)
                {
                    Max = intList[0];
                    MaxIDX = 0;
                }
                else
                {
                    if (intList[i] > Max)
                    {
                        Max = intList[i];
                        MaxIDX = i;
                    }
                }
            }

            return MaxIDX;
        }

        public void InitializeMatrix()
        {
            int[,] ZeroMatrix =
            {
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0}
            };

            ActualExpectedTrainMatrix = ZeroMatrix;
            ActualExpectedTestMatrix = ZeroMatrix;
        }

        public void DisplayMatrix(int[,] theMatrix)
        {
            int rowLength = theMatrix.GetLength(0);
            int colLength = theMatrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", theMatrix[i, j]));
                }
                Console.Write(Environment.NewLine + Environment.NewLine);
            }
            Console.ReadLine();
        }

        // Train network using J2 error, rather than backward propagate error.
        public void TrainNetworkWithJ2Error(Network theNetwork, DataTable theTrainDT, double theLearnRate, int theNumOfEpochs, int theNumberOutputs)
        {
            Random rand = new Random();
            CompleteSumLossesOfTrain = new List<double>();
            for (int idx = 0; idx < theNumOfEpochs; idx++)
            {
                double[] sumLossArray = new double[10];
                double sumLoss = 0.0;

                var shuffleTable = theTrainDT.AsEnumerable().OrderBy(r => rand.Next()).Take(3000).CopyToDataTable();

                foreach (DataRow row in shuffleTable.Rows)
                {
                    List<double> actualRow = ConvertDataRowToListDouble(row);
                    List<double> predictedRow = ForwardPropagate(theNetwork, actualRow);

                    int theActualValue = Convert.ToInt32(row.ItemArray[785]);
                    List<double> theResultList = new List<double>();

                    for (int i = 0; i < theNumberOutputs; i++)
                    {
                        theResultList.Add(predictedRow[i] - actualRow[i]);
                    }

                    sumLoss += J2LossFunctionCalculation(theResultList);
                    sumLossArray[theActualValue] += J2LossFunctionCalculation(theResultList);

                    BackwardPropagateError(theNetwork, actualRow);
                    UpdateWeights(theNetwork, actualRow, theLearnRate);
                }

                CompleteSumLossesOfTrain.Add(sumLoss);

                Console.WriteLine("Epoch={0}, lrate={1}, Loss={2}", idx, theLearnRate, sumLoss);

                // Give performance of final network for plotting later.
                if (idx == theNumOfEpochs - 1)
                {
                    TrainSumLossFinal = sumLoss;
                    TrainSumLossFinalArray = sumLossArray;
                    FinalNetworkHiddenLayer = theNetwork.Layers[0];
                }

                sumLoss = 0.0;
            }
        }

        // J2 error quantifier for use in Auto-Encoding application. End result multiplied by 1/2 since it is J2.
        public double J2LossFunctionCalculation(List<double> theErrorItemsList)
        {
            double theReturnValue = 0.0;

            foreach (double item in theErrorItemsList)
            {
                theReturnValue += Math.Pow(item, 2);
            }

            return (theReturnValue * 0.5);
        }

        // HW4 Problem 3.
        public void UpdateWeightsForHiddentToOutput(Network theNetwork, List<double> theRow, double theLearnRate)
        {
            double momentum = 0.0;
            for (int idx = 1; idx < theNetwork.Layers.Count; idx++)
            {
                List<double> inputs = theRow.Take(theRow.Count).ToList();

                if (idx != 0)
                {
                    inputs = new List<double>();
                    foreach (var neuron in theNetwork.Layers[idx - 1].Neurons)
                    {
                        inputs.Add(neuron.NeuronOutput);
                    }
                }

                foreach (var neuron in theNetwork.Layers[idx].Neurons)
                {
                    for (int j = 0; j < inputs.Count - 2; j++)
                    {
                        // Float value is a momentum value.
                        neuron.NeuronWeights[j] += theLearnRate * neuron.NeuronDelta * inputs[j] + momentum;
                    }

                    // Bias weight.
                    neuron.NeuronWeights[neuron.NeuronWeights.Count - 2] += theLearnRate * neuron.NeuronDelta + momentum;
                }
            }
        }

        // HW4 Problem 2.
        public void UpdateWeightsAnotherOne(Network theNetwork, List<double> theRow, double theLearnRate)
        {
            double momentum = 0.0;
            for (int idx = 0; idx < theNetwork.Layers.Count; idx++)
            {
                List<double> inputs = theRow.Take(theRow.Count).ToList();

                if (idx != 0)
                {
                    inputs = new List<double>();
                    foreach (var neuron in theNetwork.Layers[idx - 1].Neurons)
                    {
                        inputs.Add(neuron.NeuronOutput);
                    }
                }

                foreach (var neuron in theNetwork.Layers[idx].Neurons)
                {
                    for (int j = 0; j < inputs.Count - 2; j++)
                        if (idx == 0)
                            neuron.NeuronWeights[j] += .00025 * neuron.NeuronDelta * inputs[j] + momentum;
                        else
                            neuron.NeuronWeights[j] += theLearnRate * neuron.NeuronDelta * inputs[j] + momentum;

                    // Bias weight.
                    neuron.NeuronWeights[neuron.NeuronWeights.Count - 2] += theLearnRate * neuron.NeuronDelta + momentum;
                }
            }
        }

        // Train the network for a given number of epochs and learn rate with a provided data set.
        public void TrainNetworkForHW4P1(Network theNetwork, DataTable theTrainDT, double theLearnRate, int theNumOfEpochs, int theNumberOutputs)
        {
            Random rand = new Random();
            List<double> ErrorList = new List<double>();
            for (int idx = 0; idx < theNumOfEpochs; idx++)
            {
                var numberCorrect = 0.0;
                //var shuffleTable = theTrainDT.AsEnumerable().OrderBy(r => rand.Next()).Take(3000).CopyToDataTable();

                foreach (DataRow row in theTrainDT.Rows)
                {
                    int theGuess = 0;
                    List<double> theRowAsList = ConvertDataRowToListDouble(row);
                    var outputs = ForwardPropagate(theNetwork, theRowAsList);

                    List<double> expected = new List<double>();
                    for (int j = 0; j < theNumberOutputs; j++)
                    {
                        expected.Add(0);
                    }

                    var lastRowValue = Convert.ToInt32(row[row.ItemArray.Length - 1]);
                    expected[lastRowValue] = 1;

                    theGuess = outputs.IndexOf(outputs.Max());
                    if (theGuess == lastRowValue)
                    {
                        numberCorrect++;
                    }

                    ActualExpectedTrainMatrix[theGuess, lastRowValue]++;

                    BackwardPropagateError(theNetwork, expected);
                    UpdateWeightsForHiddentToOutput(theNetwork, theRowAsList, theLearnRate);
                }
                double Ratio = (Convert.ToDouble(numberCorrect) / Convert.ToDouble(theTrainDT.Rows.Count));
                ErrorList.Add(1 - Ratio);
                Console.WriteLine("Epoch={0}, lrate={1}, HitRate={2}", idx, theLearnRate, Ratio);

                // Give performance of final network for plotting later.
                if (idx == theNumOfEpochs - 1)
                {
                    OtherFinalNetworkHiddenLayer = theNetwork.Layers[0];
                }
            }

            EpochErrorList = ErrorList;
        }

        // Train the network for a given number of epochs and learn rate with a provided data set.
        public void TrainNetworkForHW4P2(Network theNetwork, DataTable theTrainDT, double theLearnRate, int theNumOfEpochs, int theNumberOutputs)
        {
            Random rand = new Random();
            List<double> ErrorList = new List<double>();
            for (int idx = 0; idx < theNumOfEpochs; idx++)
            {
                var numberCorrect = 0.0;
                //var shuffleTable = theTrainDT.AsEnumerable().OrderBy(r => rand.Next()).Take(3000).CopyToDataTable();

                foreach (DataRow row in theTrainDT.Rows)
                {
                    int theGuess = 0;
                    List<double> theRowAsList = ConvertDataRowToListDouble(row);
                    var outputs = ForwardPropagate(theNetwork, theRowAsList);

                    List<double> expected = new List<double>();
                    for (int j = 0; j < theNumberOutputs; j++)
                    {
                        expected.Add(0);
                    }

                    var lastRowValue = Convert.ToInt32(row[row.ItemArray.Length - 1]);
                    expected[lastRowValue] = 1;

                    theGuess = outputs.IndexOf(outputs.Max());
                    if (theGuess == lastRowValue)
                    {
                        numberCorrect++;
                    }

                    ActualExpectedTrainMatrix[theGuess, lastRowValue]++;

                    BackwardPropagateError(theNetwork, expected);
                    UpdateWeightsAnotherOne(theNetwork, theRowAsList, theLearnRate);
                }
                double Ratio = (Convert.ToDouble(numberCorrect) / Convert.ToDouble(theTrainDT.Rows.Count));
                ErrorList.Add(1 - Ratio);
                Console.WriteLine("Epoch={0}, lrate={1}, HitRate={2}", idx, theLearnRate, Ratio);

                // Give performance of final network for plotting later.
                if (idx == theNumOfEpochs - 1)
                {
                    OtherFinalNetworkHiddenLayer = theNetwork.Layers[0];
                }
            }

            EpochErrorList = ErrorList;
        }

        #endregion
    }
}
