using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;
using System.Linq;

public class ObjectDetection : MonoBehaviour
{
    public NNModel modelAsset;               // Assign your ONNX model
    public GameObject boundingBoxPrefab;     // Prefab for bounding boxes
    public Transform boundingBoxesContainer; // Parent container for bounding boxes
    public Material webcamMaterial;          // Material to display webcam feed

    private Model runtimeModel;
    private Worker worker;
    private WebCamTexture webcamTexture;

    void Start()
    {
        // Load the ONNX model and create a worker
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.Compute, runtimeModel);

        // Start the webcam feed
        webcamTexture = new WebCamTexture();
        webcamMaterial.mainTexture = webcamTexture;
        webcamTexture.Play();
    }

    void Update()
    {
        if (webcamTexture.didUpdateThisFrame)
        {
            // Preprocess webcam frame for the model
            Tensor inputTensor = PreprocessWebcamInput(webcamTexture);

            // Perform inference
            Tensor output = worker.Execute(inputTensor).PeekOutput();

            // Parse and display results
            ProcessModelOutput(output);

            // Release tensor memory
            inputTensor.Dispose();
        }
    }

    Tensor PreprocessWebcamInput(WebCamTexture webcam)
    {
        int inputWidth = 320;  // Adjust based on model requirements
        int inputHeight = 320;

        // Resize and normalize webcam frame
        Tensor inputTensor = new Tensor(1, inputHeight, inputWidth, 3);
        Color32[] pixels = webcam.GetPixels32();
        for (int y = 0; y < inputHeight; y++)
        {
            for (int x = 0; x < inputWidth; x++)
            {
                int pixelIndex = y * webcam.width + x;
                Color32 pixel = pixels[pixelIndex];

                inputTensor[0, y, x, 0] = pixel.r / 255.0f; // Normalize Red channel
                inputTensor[0, y, x, 1] = pixel.g / 255.0f; // Normalize Green channel
                inputTensor[0, y, x, 2] = pixel.b / 255.0f; // Normalize Blue channel
            }
        }
        return inputTensor;
    }

    void ProcessModelOutput(Tensor output)
    {
        // Clear previous bounding boxes
        foreach (Transform child in boundingBoxesContainer)
        {
            Destroy(child.gameObject);
        }

        // Parse bounding boxes from model output
        int detectionCount = 10; // Adjust based on your model output format
        for (int i = 0; i < detectionCount; i++)
        {
            float x = output[i * 6 + 0]; // Normalized X
            float y = output[i * 6 + 1]; // Normalized Y
            float width = output[i * 6 + 2];
            float height = output[i * 6 + 3];
            float confidence = output[i * 6 + 4];
            int classIndex = (int)output[i * 6 + 5];

            if (confidence > 0.5f)
            {
                Rect boundingBox = ScaleBoundingBox(x, y, width, height, webcamTexture.width, webcamTexture.height, 320, 320);
                DisplayBoundingBox(boundingBox, classIndex, confidence);
            }
        }
    }

    Rect ScaleBoundingBox(float x, float y, float width, float height, int imageWidth, int imageHeight, int inputWidth, int inputHeight)
    {
        float scaledX = x * imageWidth / inputWidth;
        float scaledY = y * imageHeight / inputHeight;
        float scaledWidth = width * imageWidth / inputWidth;
        float scaledHeight = height * imageHeight / inputHeight;

        return new Rect(scaledX, scaledY, scaledWidth, scaledHeight);
    }

    void DisplayBoundingBox(Rect boundingBox, int classIndex, float confidence)
    {
        GameObject box = Instantiate(boundingBoxPrefab, boundingBoxesContainer);
        RectTransform rectTransform = box.GetComponent<RectTransform>();
        rectTransform.anchoredPosition = new Vector2(boundingBox.x, boundingBox.y);
        rectTransform.sizeDelta = new Vector2(boundingBox.width, boundingBox.height);

        Text labelText = box.GetComponentInChildren<Text>();
        labelText.text = $"Class: {classIndex} ({confidence:P1})";
    }

    void OnDestroy()
    {
        worker.Dispose();
    }
}
