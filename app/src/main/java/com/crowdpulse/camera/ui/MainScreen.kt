@file:OptIn(ExperimentalMaterial3Api::class)
package com.crowdpulse.camera.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.crowdpulse.camera.ui.theme.*
import kotlinx.coroutines.delay
import java.util.concurrent.TimeUnit
import kotlin.math.sin

@Composable
fun MainScreen(
    isConnected: Boolean,
    isStreaming: Boolean,
    onToggleStream: () -> Unit,
    onLoginClick: () -> Unit,
    userEmail: String? = null,
    onSurfaceAvailable: (android.view.Surface?) -> Unit = {},
    onFlipCamera: () -> Unit = {},
    onToggleFlash: () -> Unit = {},
    sessionCode: String = "",
    serverHost: String = "",
    onSessionChanged: (host: String, code: String) -> Unit = { _, _ -> }
) {
    // ── Stats counters ─────────────────────────────────────────────────────────
    var frameCount  by remember { mutableIntStateOf(0) }
    var uptimeMs    by remember { mutableLongStateOf(0L) }
    var fps         by remember { mutableIntStateOf(0) }
    var quality     by remember { mutableIntStateOf(0) }

    LaunchedEffect(isStreaming) {
        if (isStreaming) {
            val start = System.currentTimeMillis()
            var lastFrame = frameCount
            var lastTs = start
            while (isStreaming) {
                delay(500)
                uptimeMs = System.currentTimeMillis() - start
                frameCount += (25..32).random()
                val now = System.currentTimeMillis()
                fps = ((frameCount - lastFrame).toFloat() / ((now - lastTs) / 1000f)).toInt().coerceIn(0, 60)
                lastFrame = frameCount
                lastTs = now
                quality = (85..98).random()
            }
        } else {
            frameCount = 0; uptimeMs = 0L; fps = 0; quality = 0
        }
    }

    val uptimeStr = remember(uptimeMs) {
        val h = TimeUnit.MILLISECONDS.toHours(uptimeMs)
        val m = TimeUnit.MILLISECONDS.toMinutes(uptimeMs) % 60
        val s = TimeUnit.MILLISECONDS.toSeconds(uptimeMs) % 60
        if (h > 0) "%02d:%02d:%02d".format(h, m, s) else "%02d:%02d".format(m, s)
    }

    // ── Pulse animation for stream button ─────────────────────────────────────
    val infiniteTransition = rememberInfiniteTransition(label = "home")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f, targetValue = 1.18f,
        animationSpec = infiniteRepeatable(tween(900, easing = EaseInOutSine), RepeatMode.Reverse),
        label = "pulse"
    )
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.4f, targetValue = 0f,
        animationSpec = infiniteRepeatable(tween(900, easing = EaseOutSine), RepeatMode.Restart),
        label = "pAlpha"
    )

    // ── Status glow ────────────────────────────────────────────────────────────
    val statusGlow by infiniteTransition.animateFloat(
        initialValue = 0.6f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1200, easing = EaseInOutSine), RepeatMode.Reverse),
        label = "statusGlow"
    )

    // ── Waveform when streaming ────────────────────────────────────────────────
    val wavePhase by infiniteTransition.animateFloat(
        initialValue = 0f, targetValue = (2 * Math.PI).toFloat(),
        animationSpec = infiniteRepeatable(tween(1500, easing = LinearEasing)),
        label = "wave"
    )

    var showSettingsDialog by remember { mutableStateOf(false) }
    var codeInput by remember { mutableStateOf(sessionCode) }
    var hostInput by remember { mutableStateOf(serverHost) }

    if (showSettingsDialog) {
        AlertDialog(
            onDismissRequest = { showSettingsDialog = false },
            title = { Text("Connect to Session", fontWeight = FontWeight.Bold) },
            text = {
                Column(verticalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = hostInput,
                        onValueChange = { hostInput = it.trim() },
                        label = { Text("Backend Host") },
                        placeholder = { Text("e.g. crowdpulse.onrender.com") },
                        singleLine = true,
                        supportingText = { Text("Your Render/ngrok URL (no https://)."  ) }
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    if (hostInput.isNotBlank()) {
                        onSessionChanged(hostInput, sessionCode)
                        showSettingsDialog = false
                    }
                }) { Text("Connect", color = PrimaryAccent, fontWeight = FontWeight.Bold) }
            },
            dismissButton = {
                TextButton(onClick = { showSettingsDialog = false }) { Text("Cancel", color = TextSubtle) }
            },
            containerColor = SurfaceElevated
        )
    }

    Scaffold(
        containerColor = BackgroundLight,
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        // Status dot
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(
                                    if (isConnected) SuccessGreen.copy(alpha = statusGlow)
                                    else DangerRed.copy(alpha = statusGlow)
                                )
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            "CrowdPulse",
                            style = MaterialTheme.typography.titleLarge,
                            color = TextPrimary,
                            fontWeight = FontWeight.Bold
                        )
                    }
                },
                actions = {
                    // Status badge
                    Surface(
                        shape = CircleShape,
                        color = if (isConnected) SuccessGlow else DangerGlow,
                        modifier = Modifier.padding(end = 8.dp)
                    ) {
                        Text(
                            text = if (isConnected) "LIVE" else "OFFLINE",
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isConnected) SuccessGreen else DangerRed,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                    }

                    // Profile icon
                    Box(
                        modifier = Modifier
                            .padding(end = 16.dp)
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(
                                Brush.radialGradient(listOf(PrimaryAccent.copy(alpha = 0.3f), SecondaryAccent.copy(alpha = 0.2f)))
                            )
                            .border(1.dp, PrimaryAccent.copy(alpha = 0.4f), CircleShape)
                            .clickable { onLoginClick() },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = userEmail?.firstOrNull()?.uppercaseChar()?.toString() ?: "?",
                            style = MaterialTheme.typography.labelLarge,
                            color = PrimaryAccent,
                            fontWeight = FontWeight.Bold
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = BackgroundLight.copy(alpha = 0.95f),
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(12.dp))

            // ── Camera preview card ─────────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(4f / 3f)
                    .clip(RoundedCornerShape(24.dp))
                    .background(SurfaceElevated)
                    .let { mod ->
                        if (isStreaming) mod.border(
                            width = 1.5.dp,
                            brush = Brush.linearGradient(
                                listOf(PrimaryAccent, SecondaryAccent, PrimaryAccent)
                            ),
                            shape = RoundedCornerShape(24.dp)
                        )
                        else mod.border(1.dp, BorderSubtle, RoundedCornerShape(24.dp))
                    }
            ) {
                // Always render AndroidView for TextureView to have surface ready
                androidx.compose.ui.viewinterop.AndroidView(
                    factory = { ctx ->
                        android.view.TextureView(ctx).apply {
                            surfaceTextureListener = object : android.view.TextureView.SurfaceTextureListener {
                                override fun onSurfaceTextureAvailable(st: android.graphics.SurfaceTexture, w: Int, h: Int) {
                                    st.setDefaultBufferSize(1280, 720)
                                    onSurfaceAvailable(android.view.Surface(st))
                                }
                                override fun onSurfaceTextureSizeChanged(st: android.graphics.SurfaceTexture, w: Int, h: Int) {}
                                override fun onSurfaceTextureDestroyed(st: android.graphics.SurfaceTexture): Boolean {
                                    onSurfaceAvailable(null)
                                    return true
                                }
                                override fun onSurfaceTextureUpdated(st: android.graphics.SurfaceTexture) {}
                            }
                        }
                    },
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(RoundedCornerShape(24.dp))
                )

                if (!isStreaming) {
                    // Background gradient for idle state
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                Brush.verticalGradient(
                                    colors = listOf(Color(0xFFF1F3F4), Color(0xFFE8EAED))
                                )
                            )
                    )
                    // Idle state
                    Column(
                        modifier = Modifier.align(Alignment.Center),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Box(
                            modifier = Modifier
                                .size(64.dp)
                                .clip(CircleShape)
                                .background(SurfaceCard),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Filled.CameraAlt,
                                contentDescription = "Camera",
                                tint = TextSubtle,
                                modifier = Modifier.size(30.dp)
                            )
                        }
                        Spacer(Modifier.height(12.dp))
                        Text(
                            "Camera Preview",
                            style = MaterialTheme.typography.titleSmall,
                            color = TextSubtle
                        )
                        Text(
                            "Press start to begin streaming",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSubtle.copy(alpha = 0.6f)
                        )
                    }
                }

                // ── Overlay metrics (top-left) ───────────────────────────────────
                if (isStreaming) {
                    Column(
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(12.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(Color.Black.copy(alpha = 0.5f))
                            .padding(8.dp)
                    ) {
                        MetricLabel("FPS", "$fps")
                        MetricLabel("LIMIT", "10")
                    }

                    // Timer top-right
                    Row(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(12.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(DangerRed.copy(alpha = 0.2f))
                            .border(1.dp, DangerRed.copy(alpha = 0.4f), RoundedCornerShape(8.dp))
                            .padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(6.dp)
                                .clip(CircleShape)
                                .background(DangerRed)
                        )
                        Spacer(Modifier.width(6.dp))
                        Text(
                            text = "REC $uptimeStr",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            Spacer(Modifier.height(20.dp))

            // ── Stats row ──────────────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                StatCard("UPTIME",  uptimeStr,          modifier = Modifier.weight(1f), accentColor = PrimaryAccent)
                StatCard("FRAMES",  "$frameCount",      modifier = Modifier.weight(1f), accentColor = SecondaryAccent)
                StatCard("CODE",    sessionCode,        modifier = Modifier.weight(1f), accentColor = SuccessGreen)
            }

            Spacer(Modifier.height(28.dp))

            // ── Main stream button ─────────────────────────────────────────────
            Box(contentAlignment = Alignment.Center) {
                // Pulse ring (only when streaming)
                if (isStreaming) {
                    Box(
                        modifier = Modifier
                            .size(100.dp)
                            .graphicsLayer { scaleX = pulseScale; scaleY = pulseScale }
                            .alpha(pulseAlpha)
                            .clip(CircleShape)
                            .background(DangerRed.copy(alpha = 0.3f))
                    )
                }

                Box(
                    modifier = Modifier
                        .size(80.dp)
                        .clip(CircleShape)
                        .background(
                            if (isStreaming)
                                Brush.radialGradient(listOf(DangerRed, Color(0xFFB71C1C)))
                            else
                                Brush.radialGradient(listOf(PrimaryAccent, PrimaryDim))
                        )
                        .border(
                            width = 2.dp,
                            color = if (isStreaming) DangerRed.copy(alpha = 0.5f) else PrimaryAccent.copy(alpha = 0.5f),
                            shape = CircleShape
                        )
                        .clickable { onToggleStream() },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (isStreaming) Icons.Filled.Stop else Icons.Filled.PlayArrow,
                        contentDescription = if (isStreaming) "Stop" else "Start",
                        tint = if (isStreaming) Color.White else Color.White,
                        modifier = Modifier.size(36.dp)
                    )
                }
            }

            Spacer(Modifier.height(8.dp))

            Text(
                text = if (isStreaming) "Tap to stop streaming" else "Tap to start streaming",
                style = MaterialTheme.typography.bodySmall,
                color = TextSubtle,
                textAlign = TextAlign.Center
            )

            Spacer(Modifier.height(24.dp))

            // ── Quick actions ──────────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                QuickActionButton(icon = Icons.Outlined.FlipCameraAndroid, label = "Flip", onClick = onFlipCamera)
                QuickActionButton(icon = Icons.Outlined.FlashOn,          label = "Flash", onClick = onToggleFlash)
                QuickActionButton(icon = Icons.Outlined.HighQuality,      label = "Quality")
                QuickActionButton(icon = Icons.Outlined.Settings,         label = "Host", onClick = {
                    hostInput = serverHost
                    showSettingsDialog = true
                })
            }
        }
    }
}

@Composable
private fun MetricLabel(label: String, value: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            text = "$label ",
            style = MaterialTheme.typography.labelSmall,
            color = Color.White.copy(alpha = 0.8f),
            fontSize = 9.sp
        )
        Text(
            text = value,
            style = MaterialTheme.typography.labelSmall,
            color = PrimaryAccent,
            fontWeight = FontWeight.Bold,
            fontSize = 10.sp
        )
    }
}

@Composable
private fun StatCard(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    accentColor: Color = PrimaryAccent
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceElevated)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value.ifEmpty { "--" },
            style = MaterialTheme.typography.headlineSmall,
            color = accentColor,
            fontWeight = FontWeight.Bold
        )
        Spacer(Modifier.height(2.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = TextSubtle,
            letterSpacing = 1.sp
        )
    }
}

@Composable
private fun QuickActionButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit = {}
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            modifier = Modifier
                .size(50.dp)
                .clip(RoundedCornerShape(14.dp))
                .background(SurfaceElevated)
                .border(1.dp, BorderSubtle, RoundedCornerShape(14.dp))
                .clickable { onClick() },
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = icon,
                contentDescription = label,
                tint = TextSecondary,
                modifier = Modifier.size(22.dp)
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(label, style = MaterialTheme.typography.labelSmall, color = TextSubtle)
    }
}
