using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

public class ObjectDetection : MonoBehaviour
{
    // Public fields to assign in the Unity Inspector
    public NNModel modelAsset;               // The Barracuda NNModel asset
    public Material webcamMaterial;          // Material to display webcam feed
    public GameObject boundingBoxPrefab;     // Prefab for displaying bounding boxes
    public Transform boundingBoxesContainer; // Parent container for bounding boxes

    // Private fields
    private IWorker worker;                  // Barracuda worker for executing the model
    private WebCamTexture webcamTexture;     // Webcam feed texture

    void Start()
    {
        // Load the model
        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        // Set up the webcam feed
        webcamTexture = new WebCamTexture();
        webcamMaterial.mainTexture = webcamTexture; // Assign the feed to the material
        webcamTexture.Play();
    }

    void Update()
    {
        // Ensure the webcam feed has new frames
        if (webcamTexture.didUpdateThisFrame)
        {
            // Preprocess the webcam feed into a format usable by the model
            Tensor inputTensor = PreprocessWebcamInput(webcamTexture);

            // Run the model on the preprocessed input
            worker.Execute(inputTensor);

            // Retrieve the output tensor and parse results
            Tensor output = worker.PeekOutput();
            ProcessModelOutput(output);

            // Dispose of the input tensor to free memory
            inputTensor.Dispose();
        }
    }

    Tensor PreprocessWebcamInput(WebCamTexture webcam)
    {
        // Create a tensor matching the model input dimensions
        int inputWidth = 320;  // Adjust based on model requirements
        int inputHeight = 320; // Adjust based on model requirements
        Tensor input = new Tensor(1, inputHeight, inputWidth, 3);

        // Resize and normalize webcam pixels
        Color32[] pixels = webcam.GetPixels32();
        for (int y = 0; y < inputHeight; y++)
        {
            for (int x = 0; x < inputWidth; x++)
            {
                int pixelIndex = y * webcam.width + x;
                Color32 pixel = pixels[pixelIndex];

                // Normalize pixel values to [0, 1]
                input[0, y, x, 0] = pixel.r / 255.0f; // Red channel
                input[0, y, x, 1] = pixel.g / 255.0f; // Green channel
                input[0, y, x, 2] = pixel.b / 255.0f; // Blue channel
            }
        }

        return input;
    }

    void ProcessModelOutput(Tensor output)
    {
        // Clear previous bounding boxes
        foreach (Transform child in boundingBoxesContainer)
        {
            Destroy(child.gameObject);
        }

        // Parse the output tensor to extract detection results
        int detectionCount = (int)output[0]; // Placeholder: adjust based on model format
        for (int i = 0; i < detectionCount; i++)
        {
            // Extract bounding box data and confidence scores
            float x = output[i * 6 + 1]; // Placeholder: model-specific
            float y = output[i * 6 + 2]; // Placeholder: model-specific
            float width = output[i * 6 + 3]; // Placeholder: model-specific
            float height = output[i * 6 + 4]; // Placeholder: model-specific
            float confidence = output[i * 6 + 5]; // Placeholder: confidence score

            // Display bounding boxes for confident detections
            if (confidence > 0.5f) // Confidence threshold
            {
                DisplayBoundingBox(new Rect(x, y, width, height), "Object", confidence);
            }
        }
    }

    void DisplayBoundingBox(Rect boundingBox, string label, float confidence)
    {
        // Instantiate a bounding box prefab
        GameObject box = Instantiate(boundingBoxPrefab, boundingBoxesContainer);

        // Adjust the position and size of the bounding box
        RectTransform rectTransform = box.GetComponent<RectTransform>();
        rectTransform.anchoredPosition = new Vector2(boundingBox.x, boundingBox.y);
        rectTransform.sizeDelta = new Vector2(boundingBox.width, boundingBox.height);

        // Set the label text
        Text labelText = box.GetComponentInChildren<Text>();
        labelText.text = $"{label} ({confidence:P1})";
    }

    void OnDestroy()
    {
        // Dispose of the worker to release resources
        worker.Dispose();
    }
}
