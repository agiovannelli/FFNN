using System;
using System.Collections.Generic;

namespace FFNN
{
    /// <summary>
    /// Custom storage class of type Neuron.
    /// </summary>
    public class Neuron
    {
        // Neuron weight list property.
        public List<double> NeuronWeights { get; set; }

        // Neuron output list property.
        public double NeuronOutput { get; set; }

        // Neuron delta value property.
        public double NeuronDelta { get; set; }

        // Neuron constructor.
        public Neuron(List<double> theNeuronWeights, double theNeuronOutput = 0, double theNeuronDelta = 0)
        {
            // Set properties to passed values. Throw argument null exception when applicable.
            NeuronWeights = theNeuronWeights ?? throw new ArgumentNullException();
            NeuronOutput = theNeuronOutput;
            NeuronDelta = theNeuronDelta;
        }

        // Prints formatted neuron values to console.
        public void ShowNeuron()
        {
            Console.Write("Weights: ");

            foreach (var weight in NeuronWeights)
            {
                Console.Write(weight + " ");
            }

            Console.Write("Output: " + NeuronOutput.ToString());

            Console.Write(" Delta: " + NeuronDelta.ToString());
            Console.WriteLine();
        }
    }
}
