using System;
using System.Collections.Generic;

namespace FFNN
{
    /// <summary>
    /// Custom storage class of type network.
    /// </summary>
    public class Network
    {
        // Network layer list property.
        public List<Layer> Layers { get; set; }

        // Network constructor.
        public Network(List<Layer> theLayers)
        {
            // Set property to passed value. Throw argument null exception when applicable.
            Layers = theLayers ?? throw new ArgumentNullException();
        }

        // Prints formatted network values to console.
        public void ShowNetwork()
        {
            foreach (var layer in Layers)
            {
                layer.ShowLayer();
            }
        }
    }
}
