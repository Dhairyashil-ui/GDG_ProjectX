package com.crowdpulse.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.crowdpulse.camera.auth.AuthManager
import com.crowdpulse.camera.battery.OptimizationManager
import com.crowdpulse.camera.camera.CameraDeviceManager
import com.crowdpulse.camera.intelligence.SmartController
import com.crowdpulse.camera.ui.AuthScreen
import com.crowdpulse.camera.ui.MainScreen
import com.crowdpulse.camera.ui.SplashScreen
import com.crowdpulse.camera.ui.theme.CrowdPulseTheme
import com.crowdpulse.camera.webrtc.WebRTCClient
import com.google.android.gms.auth.api.signin.GoogleSignIn
import kotlinx.coroutines.flow.MutableStateFlow

class MainActivity : ComponentActivity() {

    private lateinit var authManager: AuthManager
    private lateinit var webrtcClient: WebRTCClient
    private lateinit var cameraManager: CameraDeviceManager
    private lateinit var optimizationManager: OptimizationManager
    private lateinit var smartController: SmartController

    private val isStreaming   = MutableStateFlow(false)
    private val isConnected   = MutableStateFlow(false)
    private val userEmail     = MutableStateFlow<String?>(null)
    private val sessionCode   = MutableStateFlow("")
    private val serverHost    = MutableStateFlow("")

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions[Manifest.permission.CAMERA] == true) {
            Log.d("MainActivity", "Camera permission granted")
        }
    }

    private val signInLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val task = GoogleSignIn.getSignedInAccountFromIntent(result.data)
        val email = authManager.handleSignInResult(task)
        if (email != null) {
            userEmail.value = email
            getSharedPreferences("AppPrefs", Context.MODE_PRIVATE).edit().putString("user_email", email).apply()
            Log.d("MainActivity", "Logged in as: $email")
        } else {
            // For testing/hackathon purposes — fallback to demo user if GSI is misconfigured
            userEmail.value = "user@demo.com"
            getSharedPreferences("AppPrefs", Context.MODE_PRIVATE).edit().putString("user_email", "user@demo.com").apply()
            Log.d("MainActivity", "Google sign in failed, using demo user")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        requestPermissionLauncher.launch(
            arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        )

        val prefs = getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
        val savedCode = prefs.getString("session_code", "") ?: ""
        val finalCode = if (savedCode.isNotBlank()) savedCode else {
            val allowedChars = ('A'..'Z') + ('0'..'9')
            val generated = (1..6).map { allowedChars.random() }.joinToString("")
            prefs.edit().putString("session_code", generated).apply()
            generated
        }
        val savedHost = prefs.getString("server_host", "10.0.2.2:8000") ?: "10.0.2.2:8000"
        
        sessionCode.value = finalCode
        serverHost.value = savedHost

        authManager         = AuthManager(this)
        webrtcClient        = WebRTCClient(this) { connected ->
            isConnected.value = connected
        }
        if (finalCode.isNotBlank() && savedHost.isNotBlank()) {
            webrtcClient.setSession(savedHost, finalCode)
        }
        
        optimizationManager = OptimizationManager(this)

        smartController = SmartController { jpegBytes ->
            webrtcClient.sendFrame(jpegBytes)
        }

        cameraManager = CameraDeviceManager(this) { image ->
            if (isStreaming.value) {
                smartController.processFrame(image)
            } else {
                image.close()
            }
        }

        // Check if already signed-in
        val savedEmail = prefs.getString("user_email", null)
        if (savedEmail != null) {
            userEmail.value = savedEmail
        } else {
            val account = GoogleSignIn.getLastSignedInAccount(this)
            if (account?.email != null) {
                userEmail.value = account.email
                prefs.edit().putString("user_email", account.email).apply()
            }
        }

        setContent {
            CrowdPulseTheme {
                val navController = rememberNavController()

                val streamState  by isStreaming.collectAsState()
                val connState    by isConnected.collectAsState()
                val emailState   by userEmail.collectAsState()
                val codeState    by sessionCode.collectAsState()
                val hostState    by serverHost.collectAsState()

                NavHost(
                    navController    = navController,
                    startDestination = "splash"
                ) {
                    composable("splash") {
                        SplashScreen(
                            onSplashComplete = {
                                Log.d("DEBUG", "EmailState: $emailState")
                                // If already signed in, skip auth
                                if (emailState != null) {
                                    navController.navigate("home") {
                                        popUpTo("splash") { inclusive = true }
                                    }
                                } else {
                                    navController.navigate("auth") {
                                        popUpTo("splash") { inclusive = true }
                                    }
                                }
                            }
                        )
                    }

                    composable("auth") {
                        AuthScreen(
                            onGoogleSignInClick = {
                                signInLauncher.launch(authManager.getSignInIntent())
                            }
                        )
                        // Navigate to home once email is set
                        LaunchedEffect(emailState) {
                            if (emailState != null) {
                                navController.navigate("home") {
                                    popUpTo("auth") { inclusive = true }
                                }
                            }
                        }
                    }

                    composable("home") {
                        MainScreen(
                            isConnected    = connState,
                            isStreaming    = streamState,
                            onToggleStream = { toggleStreaming() },
                            onLoginClick   = {
                                if (emailState != null) {
                                    authManager.signOut {
                                        userEmail.value = null
                                        getSharedPreferences("AppPrefs", Context.MODE_PRIVATE).edit().remove("user_email").apply()
                                        navController.navigate("auth") {
                                            popUpTo("home") { inclusive = true }
                                        }
                                    }
                                } else {
                                    signInLauncher.launch(authManager.getSignInIntent())
                                }
                            },
                            userEmail      = emailState,
                            onSurfaceAvailable = { surface ->
                                cameraManager.setPreviewSurface(surface)
                            },
                            onFlipCamera = {
                                cameraManager.flipCamera()
                            },
                            onToggleFlash = {
                                cameraManager.toggleFlash()
                            },
                            sessionCode = codeState,
                            serverHost  = hostState,
                            onSessionChanged = { newHost, newCode ->
                                sessionCode.value = newCode
                                serverHost.value  = newHost
                                getSharedPreferences("AppPrefs", Context.MODE_PRIVATE).edit()
                                    .putString("session_code", newCode)
                                    .putString("server_host", newHost)
                                    .apply()
                                webrtcClient.setSession(newHost, newCode)
                            }
                        )
                    }
                }
            }
        }
    }

    private fun toggleStreaming() {
        if (isStreaming.value) {
            stopStreaming()
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
            ) {
                startStreaming()
            }
        }
    }

    private fun startStreaming() {
        isStreaming.value = true
        cameraManager.startCamera()
        optimizationManager.onStreamingStarted()
    }

    private fun stopStreaming() {
        isStreaming.value = false
        cameraManager.stopCamera()
        optimizationManager.onStreamingStopped()
    }

    override fun onStop() {
        super.onStop()
        if (isStreaming.value) stopStreaming()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.destroy()
        webrtcClient.release()
    }
}