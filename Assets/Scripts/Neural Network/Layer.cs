using static System.Math;

[System.Serializable]
public class Layer
{
    public int numNodesIn, numNodesOut;
    public double[] costGradientW;
    public double[] costGradientB;
    public double[] weights;
    public double[] biases;

    double[] activations;
    double[] weightedInputs;
    double[] inputs;

    // Create the layer
    public Layer(int numNodesIn, int numNodesOut)
    {
        costGradientW = new double[numNodesIn * numNodesOut];
        weights = new double[numNodesIn * numNodesOut];

        costGradientB = new double[numNodesOut];
        biases = new double[numNodesOut];

        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        RandomizeWeights();
    }

    // Calculate the output activations of the layer
    public double[] CalculateOutputs(double[] inputs)
    {
        this.inputs = inputs;
        activations = new double[numNodesOut];
        weightedInputs = new double[numNodesOut];

        //For each node in the layer, calculate the weighted input
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            //By default add in the bias for the node
            double weightedInput = biases[nodeOut];

            //For each node going into this node, get the weighted input and add it to the total weighted input
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weightedInput += inputs[nodeIn] * weights[(numNodesIn * nodeOut) + nodeIn];
            }

            //Calculate the activation of this node from the total weighted input + bias
            weightedInputs[nodeOut] = weightedInput;
            activations[nodeOut] = Activation(weightedInput);
        }

        return activations;
    }

    //Squishes the total weighted input into a 0-1 range
    double Activation(double weightedInput)
    {
        return 1 / (1 + Exp(-weightedInput));
    }

    double ActivationDerivative(double weightedInput)
    {
        double activation = Activation(weightedInput);
        return activation * (1 - activation);
    }

    //Calculates how different the activation of the node is from the expected "perfect" activation
    public double NodeCost(double outputActivation, double expectedOutput)
    {
        double error = outputActivation - expectedOutput;
        return error * error;
    }

    double NodeCostDerivative(double outputActivation, double expectedOutput)
    {
        return 2 * (outputActivation - expectedOutput);
    }

    public void ApplyGradients(double learnRate)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            biases[nodeOut] -= costGradientB[nodeOut] * learnRate;
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weights[numNodesIn * nodeOut + nodeIn] -= costGradientW[(numNodesIn * nodeOut) + nodeIn] * learnRate;
            }
        }
    }

    public void UpdateGradients(double[] nodeValues)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                double derivativeCostWrtWeight = inputs[nodeIn] * nodeValues[nodeOut];

                costGradientW[(numNodesIn * nodeOut) + nodeIn] += derivativeCostWrtWeight;
            }

            double derivativeCostWrtBias = 1 * nodeValues[nodeOut];
            costGradientB[nodeOut] += derivativeCostWrtBias;
        }
    }

    public void RandomizeWeights()
    {
        System.Random rng = new System.Random();

        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
        {
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {

                double randomValue = rng.NextDouble() * 2 - 1;

                weights[(numNodesIn * nodeOut) + nodeIn] = randomValue / Sqrt(numNodesIn);
            }
        }
    }

    public double[] CalculateOutputLayerNodeValues(double[] expectedOutputs)
    {
        double[] nodeValues = new double[expectedOutputs.Length];

        for (int i = 0; i < nodeValues.Length; i++)
        {
            double costDerivative = NodeCostDerivative(activations[i], expectedOutputs[i]);
            double activationDerivative = ActivationDerivative(weightedInputs[i]);
            nodeValues[i] = activationDerivative * costDerivative;
        }

        return nodeValues;
    }

    public double[] CalculateHiddenLayerNodeValues(Layer oldLayer, double[] oldNodeValues)
    {
        double[] newNodeValues = new double[numNodesOut];

        for (int newNodeIndex = 0; newNodeIndex < newNodeValues.Length; newNodeIndex++)
        {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.Length; oldNodeIndex++)
            {
                double weightedInputDerivative = oldLayer.weights[(oldLayer.numNodesIn * oldNodeIndex) + newNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            newNodeValue *= ActivationDerivative(weightedInputs[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;
    }


    public void ClearAllGradients()
    {
        costGradientW = new double[numNodesIn * numNodesOut];
        costGradientB = new double[numNodesOut];
    }
}
