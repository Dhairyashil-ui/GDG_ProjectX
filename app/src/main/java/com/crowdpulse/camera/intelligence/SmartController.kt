package com.crowdpulse.camera.intelligence

import android.media.Image
import android.util.Log
import com.crowdpulse.camera.utils.ImageUtils
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Controller handling frame intelligence: motion detection dropping, frame resize/compression,
 * and adaptive FPS mechanisms.
 */
class SmartController(private val onFrameReadyToSend: (ByteArray) -> Unit) {

    private val scope = CoroutineScope(Dispatchers.Default)
    
    // Adaptive controls
    var currentFpsTarget = 10
    private var lastFrameTimeMs = 0L
    private var lastSentFrameTimeMs = 0L
    private val isProcessing = AtomicBoolean(false)

    fun processFrame(image: Image) {
        val currentTime = System.currentTimeMillis()
        val intervalMs = 1000 / currentFpsTarget
        
        if (isProcessing.get() || currentTime - lastFrameTimeMs < intervalMs) {
            image.close()
            return // Skip based on adaptive FPS
        }
        lastFrameTimeMs = currentTime
        isProcessing.set(true)

        scope.launch {
            try {
                // Compress directly and send
                val timeSinceLastSend = currentTime - lastSentFrameTimeMs
                lastSentFrameTimeMs = currentTime
                
                // Resize to max 640x480 & Compress to 35%
                val jpegBytes = ImageUtils.yuv420ToJpeg(image, 640, 480, 35)
                
                withContext(Dispatchers.Main) {
                    onFrameReadyToSend(jpegBytes)
                }
            } catch (e: Exception) {
                Log.e("SmartController", "Error processing frame", e)
            } finally {
                image.close() // Important! Close the image buffer so camera can grab more
                isProcessing.set(false)
            }
        }
    }

    /**
     * Automatically adjusts fps target based on simulated network heuristics.
     * In real implementation, this reads WebRTC RTT stats.
     */
    fun updateNetworkState(isNetworkSlow: Boolean) {
        currentFpsTarget = if (isNetworkSlow) {
            5 // Drop FPS if bad bandwidth
        } else {
            15 // Max FPS
        }
    }
}
