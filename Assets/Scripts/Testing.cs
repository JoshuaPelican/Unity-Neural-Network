using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO;

public class Testing : MonoBehaviour
{
    NeuralNetwork network;
    [SerializeField] RawImage DisplayImage;
    [SerializeField] TextMeshProUGUI ExpectedText;
    [SerializeField] TextMeshProUGUI PredictedText;
    [SerializeField] TextMeshProUGUI CostText;
    [Space]
    [SerializeField] int[] Layers;
    [SerializeField] float SecondsPerTrain = 1.25f;
    [SerializeField] int BatchSize = 10;
    [SerializeField] double LearnRate = 0.45;
    [SerializeField] int MiniDataSize = 600;
    float c;
    bool isTraining = false;

    private DataPoint[] TrainingData;
    private DataPoint[] MiniTrainingData;
    private DataPoint[] TestingData;

    private void Start()
    {
        network = new NeuralNetwork(Layers);

        TrainingData = MNISTProcessor.ReadTrainingData();
        TestingData = MNISTProcessor.ReadTestData();

        MiniTrainingData = new DataPoint[MiniDataSize];

        for (int i = 0; i < MiniDataSize; i++)
        {
            int start = 0; // Random.Range(0, TrainingData.Length - MiniDataSize);
            MiniTrainingData[i] = TrainingData[start + i];
        }
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.A))
        {
            CalulcateAccuracy();
        }

        if (Input.GetKeyDown(KeyCode.C))
        {
            ClassifyInputData(GrabRandomSample(TestingData, 1)[0]);
        }

        if (Input.GetKeyDown(KeyCode.T))
        {
            isTraining = !isTraining;
        }

        if (Input.GetKeyDown(KeyCode.S))
        {
            SaveNetwork(network, "network.json");
        }

        if (Input.GetKeyDown(KeyCode.L))
        {
            network = LoadNetwork("network.json");
        }


        if (!isTraining)
            return;

        c += Time.deltaTime;
        if (c >= SecondsPerTrain)
        {
            c = 0;
            LearnMini();
        }
    }

    void ClassifyInputData(DataPoint data)
    {
        int output = network.Classify(data.inputs);

        ExpectedText.SetText(data.ExpectedHighestIndex.ToString());
        DisplayImage.texture = MNISTProcessor.ConvertToTexture(data);
        PredictedText.SetText(output.ToString());
    }

    DataPoint[] GrabRandomSample(DataPoint[] data, int sampleSize)
    {
        DataPoint[] sample = new DataPoint[sampleSize];

        for (int i = 0; i < sampleSize; i++)
        {
            sample[i] = data[Random.Range(0, data.Length)];
        }

        return sample;
    }

    void Train()
    {
        network.Learn(GrabRandomSample(TrainingData, BatchSize), LearnRate);
        CostText.text = string.Format("Cost: {0:f5}", network.Cost(TrainingData));
    }

    void LearnMini()
    {
        network.Learn(GrabRandomSample(MiniTrainingData, BatchSize), LearnRate);
        CostText.text = string.Format("Cost: {0:f5}", network.Cost(MiniTrainingData));
    }

    public void SaveNetwork(NeuralNetwork network, string filePath)
    {
        string networkData = JsonUtility.ToJson(network, false);
        File.WriteAllText(Application.dataPath + "/" + filePath, networkData);
        Debug.Log("Network Saved!");
    }

    public NeuralNetwork LoadNetwork(string filePath)
    {
        string networkData = File.ReadAllText(Application.dataPath + "/" + filePath);
        NeuralNetwork network = JsonUtility.FromJson<NeuralNetwork>(networkData);
        Debug.Log("Network Loaded!");
        CostText.text = string.Format("Cost: {0:f5}", network.Cost(MiniTrainingData));
        return network;

    }

    void CalulcateAccuracy()
    {
        float totalCorrect = 0;

        foreach (DataPoint data in TestingData)
        {
            if (network.Classify(data.inputs) != data.ExpectedHighestIndex)
                continue;

            totalCorrect++;
        }

        Debug.Log($"Accuracy: {totalCorrect / TestingData.Length}");
    }
}
