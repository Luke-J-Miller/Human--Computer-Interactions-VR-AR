using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;

public class ObjectDetection : MonoBehaviour
{
    public NNModel modelAsset;               // Assign your ONNX model
    public Material webcamMaterial;          // Material for displaying webcam feed
    public GameObject boundingBoxPrefab;     // Prefab for bounding boxes
    public Transform boundingBoxesContainer; // Parent container for bounding boxes

    private Model runtimeModel;              // Sentis runtime model
    private IWorker worker;                  // Sentis worker for inference
    private WebCamTexture webcamTexture;     // Webcam feed texture

    void Start()
    {
        // Load the ONNX model and create a Sentis worker
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.Compute, runtimeModel);

        // Initialize the webcam feed
        webcamTexture = new WebCamTexture();
        webcamMaterial.mainTexture = webcamTexture;
        webcamTexture.Play();
    }

    void Update()
    {
        // Ensure the webcam feed has new frames
        if (webcamTexture.didUpdateThisFrame)
        {
            // Preprocess the webcam frame
            Tensor inputTensor = PreprocessWebcamInput(webcamTexture);

            // Run inference
            worker.Execute(inputTensor);

            // Retrieve the model's output tensor
            Tensor output = worker.PeekOutput();

            // Parse and display the output
            DisplayDetections(output);

            // Dispose of input tensor to free memory
            inputTensor.Dispose();
        }
    }

    Tensor PreprocessWebcamInput(WebCamTexture webcam)
    {
        // Resize and normalize webcam input to match the model's expected input size
        int inputWidth = 320;  // Replace with your model's input width
        int inputHeight = 320; // Replace with your model's input height

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

    void DisplayDetections(Tensor output)
    {
        // Clear existing bounding boxes
        foreach (Transform child in boundingBoxesContainer)
        {
            Destroy(child.gameObject);
        }

        // Example: Assuming model output is structured as [x, y, width, height, confidence, classIndex]
        int detectionCount = 10; // Adjust based on your model's output format
        for (int i = 0; i < detectionCount; i++)
        {
            float x = output[i * 6 + 0]; // Normalized X-coordinate
            float y = output[i * 6 + 1]; // Normalized Y-coordinate
            float width = output[i * 6 + 2]; // Normalized width
            float height = output[i * 6 + 3]; // Normalized height
            float confidence = output[i * 6 + 4]; // Confidence score
            int classIndex = (int)output[i * 6 + 5]; // Class index

            if (confidence > 0.5f) // Filter by confidence threshold
            {
                Rect boundingBox = ScaleBoundingBox(x, y, width, height, webcamTexture.width, webcamTexture.height);
                CreateBoundingBox(boundingBox, classIndex, confidence);
            }
        }
    }

    Rect ScaleBoundingBox(float x, float y, float width, float height, int imageWidth, int imageHeight)
    {
        // Scale bounding box from normalized coordinates to image space
        float scaledX = x * imageWidth;
        float scaledY = y * imageHeight;
        float scaledWidth = width * imageWidth;
        float scaledHeight = height * imageHeight;

        return new Rect(scaledX, scaledY, scaledWidth, scaledHeight);
    }

    void CreateBoundingBox(Rect boundingBox, int classIndex, float confidence)
    {
        // Instantiate a bounding box prefab
        GameObject box = Instantiate(boundingBoxPrefab, boundingBoxesContainer);

        // Adjust its position and size
        RectTransform rectTransform = box.GetComponent<RectTransform>();
        rectTransform.anchoredPosition = new Vector2(boundingBox.x, boundingBox.y);
        rectTransform.sizeDelta = new Vector2(boundingBox.width, boundingBox.height);

        // Update the label text
        Text labelText = box.GetComponentInChildren<Text>();
        labelText.text = $"Class: {classIndex}, Confidence: {confidence:P1}";
    }

    void OnDestroy()
    {
        // Dispose of the worker to release resources
        worker.Dispose();
    }
}
