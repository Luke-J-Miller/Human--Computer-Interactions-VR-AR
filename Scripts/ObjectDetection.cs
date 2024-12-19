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

    // Hardcoded labels
    private string[] labels = new string[]
    {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

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

                input[0, y, x, 0] = pixel.r / 255.0f; // Normalize Red channel
                input[0, y, x, 1] = pixel.g / 255.0f; // Normalize Green channel
                input[0, y, x, 2] = pixel.b / 255.0f; // Normalize Blue channel
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

    // Model input dimensions
    int inputWidth = 320;  // Adjust based on your model
    int inputHeight = 320;

    // Webcam dimensions
    int imageWidth = webcamTexture.width;
    int imageHeight = webcamTexture.height;

    // Iterate through the model's output
    int detectionCount = 10; // Adjust based on your model's output format
    for (int i = 0; i < detectionCount; i++)
    {
        // Extract bounding box data
        float x = output[i * 6 + 0]; // Normalized X
        float y = output[i * 6 + 1]; // Normalized Y
        float width = output[i * 6 + 2]; // Normalized Width
        float height = output[i * 6 + 3]; // Normalized Height
        float confidence = output[i * 6 + 4]; // Confidence
        int classIndex = (int)output[i * 6 + 5]; // Class index

        // Only display confident detections
        if (confidence > 0.5f)
        {
            // Scale bounding box from normalized model output to image space
            Rect boundingBox = ScaleBoundingBox(x, y, width, height, imageWidth, imageHeight, inputWidth, inputHeight);

            // If using a canvas, convert to screen space
            if (boundingBoxesContainer.GetComponent<Canvas>())
            {
                Vector2 canvasPosition = ScreenToCanvasPosition(boundingBox.x, boundingBox.y, imageWidth, imageHeight);
                boundingBox.x = canvasPosition.x;
                boundingBox.y = canvasPosition.y;
            }

            DisplayBoundingBox(boundingBox, classIndex, confidence);
        }
    }
}


    void DisplayBoundingBox(Rect boundingBox, int classIndex, float confidence)
    {
        // Instantiate a bounding box prefab
        GameObject box = Instantiate(boundingBoxPrefab, boundingBoxesContainer);

        // Adjust the position and size of the bounding box
        RectTransform rectTransform = box.GetComponent<RectTransform>();
        rectTransform.anchoredPosition = new Vector2(boundingBox.x, boundingBox.y);
        rectTransform.sizeDelta = new Vector2(boundingBox.width, boundingBox.height);

        // Set the label text
        Text labelText = box.GetComponentInChildren<Text>();
        labelText.text = $"{labels[classIndex]} ({confidence:P1})";
    }

    void OnDestroy()
    {
        // Dispose of the worker to release resources
        worker.Dispose();
    }
}
