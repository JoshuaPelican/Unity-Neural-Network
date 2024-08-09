[System.Serializable]
public class NeuralNetwork
{
    public Layer[] layers;

    public NeuralNetwork(params int[] layerSizes)
    {
        layers = new Layer[layerSizes.Length - 1];
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
        }

    }

    double[] CalculateOuputs(double[] inputs)
    {
        //Calculates activations for each layer, ovveriding the previous
        foreach (Layer layer in layers)
        {
            inputs = layer.CalculateOutputs(inputs);
        }

        //Returns the output layer's activations
        return inputs;
    }

    public int Classify(double[] inputs)
    {
        //Calculates the activations of the network and returns the index of the highest activation
        double[] outputs = CalculateOuputs(inputs);
        return IndexOfMaxValue(outputs);
    }

    int IndexOfMaxValue(double[] values)
    {
        double maxValue = double.MinValue;
        int index = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] > maxValue)
            {
                maxValue = values[i];
                index = i;
            }
        }

        return index;
    }

    //Calculates the overall cost of the network based on a single data point
    double Cost(DataPoint dataPoint)
    {
        double[] outputs = CalculateOuputs(dataPoint.inputs);
        Layer outputLayer = layers[layers.Length - 1];
        double cost = 0;

        for (int nodeOut = 0; nodeOut < outputs.Length; nodeOut++)
        {
            cost += outputLayer.NodeCost(outputs[nodeOut], dataPoint.expectedOutputs[nodeOut]);
        }

        return cost;
    }

    public double Cost(DataPoint[] data)
    {
        double totalCost = 0;

        foreach (DataPoint dataPoint in data)
        {
            totalCost += Cost(dataPoint);
        }

        return totalCost / data.Length;
    }

    public void Learn(DataPoint[] trainingData, double learnRate)
    {
        foreach (DataPoint dataPoint in trainingData)
        {
            UpdateAllGradients(dataPoint);
        }

        ApplyAllGradients(learnRate / trainingData.Length);

        ClearAllGradients();
    }

    void ApplyAllGradients(double learnRate)
    {
        foreach (Layer layer in layers)
        {
            layer.ApplyGradients(learnRate);
        }
    }

    void UpdateAllGradients(DataPoint dataPoint)
    {
        CalculateOuputs(dataPoint.inputs);

        Layer outputLayer = layers[layers.Length - 1];
        double[] nodeValues = outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs);
        outputLayer.UpdateGradients(nodeValues);

        for (int hiddenLayerIndex = layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            Layer hiddenLayer = layers[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
            hiddenLayer.UpdateGradients(nodeValues);
        }
    }

    void ClearAllGradients()
    {
        foreach (Layer layer in layers)
        {
            layer.ClearAllGradients();
        }
    }
}
