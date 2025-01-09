using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Rendering;

public class XRView : MonoBehaviour
{
    [Header("Assign your Capture Camera here if using Option A")]
    public Camera captureCamera;

    [Header("RenderTexture from the Camera (TargetTexture)")]
    public RenderTexture cameraTexture;

    // We don’t need RawImage or AspectRatioFitter if we’re not displaying it on a Canvas
    // We'll replicate the "GetCamImage" concept but with RenderTexture readback.

    public bool IsReady { get; private set; } = false;
    private Texture2D tempTexture;

    void Start()
    {
        if (cameraTexture == null && captureCamera != null)
        {
            // If not assigned, grab the camera’s targetTexture.
            cameraTexture = captureCamera.targetTexture;
        }

        // Create a Texture2D that matches the RenderTexture dimensions
        tempTexture = new Texture2D(cameraTexture.width, cameraTexture.height, TextureFormat.RGB24, false);
    }

    // This is similar to "GetCamImage" but for RenderTexture
    public void RequestFrame(System.Action<Texture2D> onFrameReady)
    {
        // We do an async GPU readback of the RenderTexture
        AsyncGPUReadback.Request(cameraTexture, 0, TextureFormat.RGB24, (req) =>
        {
            if (req.hasError)
            {
                Debug.LogError("GPU readback error");
                return;
            }
            var data = req.GetData<byte>();
            tempTexture.LoadRawTextureData(data);
            tempTexture.Apply();

            onFrameReady?.Invoke(tempTexture);
        });
    }
}
