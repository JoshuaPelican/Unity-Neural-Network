[System.Serializable]
public struct DataPoint
{
	public double[] inputs;
	public double[] expectedOutputs;
	public int ExpectedHighestIndex => IndexOfMaxValue(expectedOutputs);

	public DataPoint(double[] inputs, double[] expectedOutputs)
	{
		this.inputs = inputs;
		this.expectedOutputs = expectedOutputs;
	}

	public DataPoint(double[] inputs, int expectedHighestIndex)
    {
		this.inputs = inputs;
		expectedOutputs = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		expectedOutputs[expectedHighestIndex] = 1;
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
}
