using System;
using System.Collections.Generic;

namespace FFNN
{
    /// <summary>
    /// Custom storage class of type Layer.
    /// </summary>
    public class Layer
    {
        // Layer neurons list property.
        public List<Neuron> Neurons { get; set; }

        // Layer constructor.
        public Layer(List<Neuron> theNeuronsList)
        {
            // Set property to passed value. Throw argument null exception when applicable.
            Neurons = theNeuronsList ?? throw new ArgumentNullException();
        }

        // Prints formatted layer values to console.
        public void ShowLayer()
        {
            foreach (var neuron in Neurons)
            {
                neuron.ShowNeuron();
            }
        }
    }
}
