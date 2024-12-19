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
            // Convert the webcam frame to a tensor
            var input = new Tensor(webcamTexture, 3); // Converts webcamTexture directly to a tensor

            // Run the ML model
            worker.Execute(input);

            // Process the output
            var output = worker.PeekOutput();
            Debug.Log("Output shape: " + output.shape);
            float[] outputArray = output.ToReadOnlyArray();
            Debug.Log("First value: " + outputArray[0]); // Example of processing the output

            // Dispose tensors
            input.Dispose();
            output.Dispose();
        }
    }

    void OnDestroy()
    {
        worker.Dispose();
    }
}
