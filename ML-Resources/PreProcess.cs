using UnityEngine;
using UnityEngine.Events;

public class Preprocess : MonoBehaviour
{
    /// <summary>
    /// Scales and/or crops the given Texture2D (already on CPU) to a square of size desiredSize. 
    /// Then returns the raw pixel bytes (RGB24) via the callback.
    /// </summary>
    public void ScaleAndCropImage(Texture2D source, int desiredSize, UnityAction<byte[]> callback)
    {
        // 1) Create a temporary RenderTexture to hold the resized image
        RenderTexture rt = RenderTexture.GetTemporary(desiredSize, desiredSize, 0, RenderTextureFormat.ARGB32);

        // 2) Blit the source onto rt. If source is not already a square, 
        //    this will stretch it. If you need strict center-cropping, 
        //    see Unity docs on Graphics.Blit with scale/offset.
        Graphics.Blit(source, rt);

        // 3) Copy the RenderTexture back into a new CPU Texture2D
        RenderTexture previousRT = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D resizedTex = new Texture2D(desiredSize, desiredSize, TextureFormat.RGB24, false);
        resizedTex.ReadPixels(new Rect(0, 0, desiredSize, desiredSize), 0, 0);
        resizedTex.Apply();

        RenderTexture.active = previousRT;
        RenderTexture.ReleaseTemporary(rt);

        // 4) Get the raw pixel data in RGB24 format (0-255)
        byte[] pixelData = resizedTex.GetRawTextureData();

        // 5) Cleanup
        Destroy(resizedTex);

        // 6) Invoke callback with the result
        callback?.Invoke(pixelData);
    }
}
