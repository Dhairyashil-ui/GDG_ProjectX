package com.crowdpulse.camera.webrtc

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import okhttp3.*
import okio.ByteString

/**
 * Manages the WebSocket connection to the CrowdPulse backend.
 *
 * Connection URL format: ws(s)://<host>/ws/camera/<sessionCode>
 * The [sessionCode] is a 6-char alphanumeric code obtained from the
 * web dashboard's "New Session" button.
 */
class WebRTCClient(private val context: Context, private val onConnectionStateChanged: (Boolean) -> Unit) {

    private val client = OkHttpClient()
    private var webSocket: WebSocket? = null
    var isConnected = false
        private set

    // Default fallback (emulator localhost). Will be overwritten via setSession().
    private var serverHost  = "10.0.2.2:8000"
    private var useSecure   = false
    private var sessionCode = ""        // current 6-char session code

    private val scope = CoroutineScope(Dispatchers.IO)

    // FIX: Track last sent timestamp to throttle frame rate.
    // Android camera produces 30 FPS but the server can only process ~10-12 FPS.
    // Without throttling, the WebSocket send buffer floods causing OkHttp to
    // silently drop frames erratically — producing the "stuck then jump" behavior.
    private var lastFrameSentAt = 0L
    private val TARGET_FPS = 12
    private val FRAME_INTERVAL_MS = 1000L / TARGET_FPS  // ~83ms between frames

    /**
     * Set or update the backend host. No session code required here;
     * call [setSession] to provide the code and connect.
     */
    fun setServerHost(host: String) {
        val isLocal = host.startsWith("10.0.") || host.startsWith("192.168.") || host.startsWith("localhost")
        serverHost = if (isLocal && !host.contains(":")) "$host:8000" else host
        useSecure  = !isLocal
    }

    /**
     * Set the session code and (re)connect to /ws/camera/{code}.
     * Called after the user enters the code validated by the backend.
     */
    fun setSession(host: String, code: String) {
        setServerHost(host)
        sessionCode = code.uppercase()
        // Use cancel() to abruptly kill the socket rather than waiting for a graceful close handshake
        webSocket?.cancel()
        isConnected = false
        // FIX: Reset frame throttle timer on new session so first frame is
        // sent immediately rather than waiting up to FRAME_INTERVAL_MS
        lastFrameSentAt = 0L
        onConnectionStateChanged(false)
        connectWebSocket()
    }

    private fun connectWebSocket() {
        if (sessionCode.isBlank()) {
            Log.w("WebRTCClient", "No session code set — not connecting.")
            return
        }
        val protocol = if (useSecure) "wss" else "ws"
        val url = "$protocol://$serverHost/ws/camera/$sessionCode"
        Log.d("WebRTCClient", "Connecting to: $url")
        val request = Request.Builder().url(url).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d("WebRTCClient", "WebSocket Connected! Session: $sessionCode")
                isConnected = true
                // FIX: Reset throttle on reconnect so first frame goes out immediately
                lastFrameSentAt = 0L
                onConnectionStateChanged(true)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                // Not expecting text messages from server
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
                Log.d("WebRTCClient", "WebSocket Closing: $reason (code $code)")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (isConnected) {
                    isConnected = false
                    onConnectionStateChanged(false)
                    // Don't auto-retry if session was rejected (4404 = invalid/expired)
                    if (code != 4404) handleDisconnection()
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("WebRTCClient", "WebSocket Error", t)
                if (isConnected || webSocket == this@WebRTCClient.webSocket) {
                    isConnected = false
                    onConnectionStateChanged(false)
                    handleDisconnection()
                }
            }
        })
    }

    private fun handleDisconnection() {
        scope.launch {
            Log.e("WebRTCClient", "Connection lost! Retrying in 3 seconds...")
            delay(3000)
            if (!isConnected) {
                Log.d("WebRTCClient", "Attempting reconnect for session: $sessionCode")
                connectWebSocket()
            }
        }
    }

    /**
     * Send a JPEG frame to the backend.
     *
     * FIX: Added timestamp-based throttling to match server processing rate.
     *
     * Problem: Camera produces frames at ~30 FPS. The server processes at ~10-12 FPS.
     * Sending every frame causes OkHttp's internal send buffer to fill up.
     * When full, send() returns false and frames are dropped unpredictably,
     * causing the dashboard to freeze for seconds then jump forward suddenly.
     *
     * Solution: Drop frames on the sender side at a controlled rate (TARGET_FPS)
     * instead of letting the buffer decide randomly which frames to drop.
     * This keeps the pipe clear and ensures each frame that IS sent gets
     * processed and displayed promptly without queuing delay.
     */
    fun sendFrame(jpegBytes: ByteArray) {
        if (!isConnected) return

        val now = System.currentTimeMillis()

        // FIX: Skip this frame if we haven't waited long enough since the last send
        if (now - lastFrameSentAt < FRAME_INTERVAL_MS) return

        lastFrameSentAt = now

        val queueSize = webSocket?.queueSize() ?: 0L
        // 200KB limit (approx 6-8 frames)
        if (queueSize > 200_000L) {
            Log.w("WebRTCClient", "Dropping frame to avoid lag (queue: $queueSize)")
            return
        }

        val byteString = ByteString.of(*jpegBytes)
        val success = webSocket?.send(byteString) ?: false
        if (!success) {
            // Buffer still full even at reduced rate — log and reset timer
            // so we back off slightly before the next attempt
            Log.w("WebRTCClient", "Frame dropped — socket buffer full (consider reducing TARGET_FPS)")
            lastFrameSentAt = now + FRAME_INTERVAL_MS  // add extra back-off
        }
    }

    fun release() {
        webSocket?.close(1000, "App Destroyed")
        scope.cancel()
    }
}