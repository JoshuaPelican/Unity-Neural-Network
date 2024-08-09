using System;
using System.IO;
using System.Linq;
using UnityEngine;

public static class MNISTProcessor
{
    private const string TrainImages = "Assets/MNIST/train-images.idx3-ubyte";
    private const string TrainLabels = "Assets/MNIST/train-labels.idx1-ubyte";
    private const string TestImages = "Assets/MNIST/t10k-images.idx3-ubyte";
    private const string TestLabels = "Assets/MNIST/t10k-labels.idx1-ubyte";

    public static DataPoint[] ReadTrainingData()
    {
        return Read(TrainImages, TrainLabels);
    }

    public static DataPoint[] ReadTestData()
    {
        return Read(TestImages, TestLabels);
    }

    private static DataPoint[] Read(string imagesPath, string labelsPath)
    {
        BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
        BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

        int magicNumber = images.ReadBigInt32();
        int numberOfImages = images.ReadBigInt32();
        int width = images.ReadBigInt32();
        int height = images.ReadBigInt32();

        int magicLabel = labels.ReadBigInt32();
        int numberOfLabels = labels.ReadBigInt32();

        DataPoint[] data = new DataPoint[numberOfImages];

        for (int i = 0; i < numberOfImages; i++)
        {
            byte[] bytes = images.ReadBytes(width * height);
            double[] inputs = new double[bytes.Length];

            for (int j = 0; j < bytes.Length; j++)
            {
                inputs[j] = bytes[j] / 255d;
            }

            data[i] = new DataPoint(inputs, labels.ReadByte());
        }
        return data;
    }

    public static int ReadBigInt32(this BinaryReader br)
    {
        var bytes = br.ReadBytes(sizeof(Int32));
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    public static void ForEach<T>(this T[] source, Action<int> action)
    {
        for (int i = 0; i < source.GetLength(0); i++)
        {
            action(i);
        }
    }

    public static Texture2D ConvertToTexture(DataPoint data)
    {
        Texture2D texture = new Texture2D(28, 28);
        texture.filterMode = FilterMode.Point;
        texture.name = data.ExpectedHighestIndex.ToString();

        for (int x = 0; x < 28; x++)
        {
            for (int y = 0; y < 28; y++)
            {
                float val = (float)data.inputs[(x * 28) + y];
                texture.SetPixel(x, y, new Color(val, val, val));
            }
        }

        texture.Apply();

        return texture;
    }
}