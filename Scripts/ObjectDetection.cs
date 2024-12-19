using UnityEngine;
using Unity.Barracuda;

public class ObjectDetection : MonoBehaviour
{
    public NNModel modelAsset; // Drag the NNModel here
    private IWorker worker;

    public WebCamTexture webcamTexture;
    public Material webcamMaterial; // Assign the material created earlier

    void Start()
    {
        // Load the ML model
        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        // Set up webcam feed
        webcamTexture = new WebCamTexture();
        webcamMaterial.mainTexture = webcamTexture;
        webcamTexture.Play();
    }


    void Update()
    {
        // Process the webcam feed
        if (webcamTexture.didUpdateThisFrame)
        {
            // Convert the webcam frame to a format usable by the model
            var input = new Tensor(webcamTexture.height, webcamTexture.width, 3); 
            Color32[] pixels = webcamTexture.GetPixels32();
            for (int i = 0; i < pixels.Length; i++)
            {
                input[0, i % webcamTexture.width, i / webcamTexture.width, 0] = pixels[i].r / 255.0f;
                input[0, i % webcamTexture.width, i / webcamTexture.width, 1] = pixels[i].g / 255.0f;
                input[0, i % webcamTexture.width, i / webcamTexture.width, 2] = pixels[i].b / 255.0f;
            }

            // Run the ML model
            worker.Execute(input);

            // Process the output
            var output = worker.PeekOutput();
            Debug.Log("Output: " + output[0]); // Debugging the first prediction
            input.Dispose();
        }
    }

    void OnDestroy()
    {
        worker.Dispose();
    }
}



