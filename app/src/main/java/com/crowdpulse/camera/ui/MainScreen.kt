@file:OptIn(ExperimentalMaterial3Api::class)
package com.crowdpulse.camera.ui

import androidx.compose.animation.animateColorAsState
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
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.crowdpulse.camera.R
import com.crowdpulse.camera.ui.theme.*
import kotlinx.coroutines.delay
import java.util.concurrent.TimeUnit

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
    // ── Stats counters ──────────────────────────────────────────────────────────
    var frameCount by remember { mutableIntStateOf(0) }
    var uptimeMs   by remember { mutableLongStateOf(0L) }
    var fps        by remember { mutableIntStateOf(0) }

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
            }
        } else {
            frameCount = 0; uptimeMs = 0L; fps = 0
        }
    }

    val uptimeStr = remember(uptimeMs) {
        val h = TimeUnit.MILLISECONDS.toHours(uptimeMs)
        val m = TimeUnit.MILLISECONDS.toMinutes(uptimeMs) % 60
        val s = TimeUnit.MILLISECONDS.toSeconds(uptimeMs) % 60
        if (h > 0) "%02d:%02d:%02d".format(h, m, s) else "%02d:%02d".format(m, s)
    }

    // ── Settings Dialog ─────────────────────────────────────────────────────────
    var showSettingsDialog by remember { mutableStateOf(false) }
    var hostInput by remember { mutableStateOf(serverHost) }

    if (showSettingsDialog) {
        AlertDialog(
            onDismissRequest = { showSettingsDialog = false },
            shape = RoundedCornerShape(24.dp),
            containerColor = MaterialTheme.colorScheme.surface,
            tonalElevation = 8.dp,
            icon = {
                Icon(
                    Icons.Outlined.Settings,
                    contentDescription = null,
                    tint = PrimaryAccent,
                    modifier = Modifier.size(28.dp)
                )
            },
            title = {
                Text(
                    "Connection Settings",
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.headlineSmall
                )
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(
                        "Configure the backend host URL for your Croudify server.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    OutlinedTextField(
                        value = hostInput,
                        onValueChange = { hostInput = it.trim() },
                        label = { Text("Backend Host URL") },
                        leadingIcon = { Icon(Icons.Outlined.Cloud, contentDescription = null) },
                        singleLine = true,
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (hostInput.isNotBlank()) {
                            onSessionChanged(hostInput, sessionCode)
                            showSettingsDialog = false
                        }
                    },
                    shape = RoundedCornerShape(12.dp)
                ) { Text("Save", fontWeight = FontWeight.SemiBold) }
            },
            dismissButton = {
                TextButton(onClick = { showSettingsDialog = false }) { Text("Cancel") }
            }
        )
    }

    // ── FAB pulse animation ─────────────────────────────────────────────────────
    val fabScale by animateFloatAsState(
        targetValue = if (isStreaming) 1f else 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy),
        label = "fabScale"
    )
    val fabColor by animateColorAsState(
        targetValue = if (isStreaming) DangerRed else PrimaryAccent,
        animationSpec = tween(300),
        label = "fabColor"
    )

    Scaffold(
        containerColor = Color.White,
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Image(
                            painter = painterResource(id = R.drawable.app_logo),
                            contentDescription = "App Logo",
                            modifier = Modifier
                                .size(28.dp)
                                .clip(RoundedCornerShape(6.dp))
                        )
                        Spacer(Modifier.width(10.dp))
                        Text(
                            "Croudify",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                },
                actions = {
                    // Connection chip
                    Surface(
                        shape = RoundedCornerShape(20.dp),
                        color = if (isConnected) SuccessGlow else DangerGlow,
                        modifier = Modifier.padding(end = 8.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(if (isConnected) SuccessGreen else DangerRed)
                            )
                            Text(
                                text = if (isConnected) "Live" else "Offline",
                                style = MaterialTheme.typography.labelSmall,
                                color = if (isConnected) SuccessGreen else DangerRed,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    // Avatar button
                    IconButton(onClick = onLoginClick) {
                        Surface(
                            shape = CircleShape,
                            color = PrimaryGlow,
                            modifier = Modifier.size(36.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Text(
                                    text = userEmail?.firstOrNull()?.uppercaseChar()?.toString() ?: "?",
                                    fontWeight = FontWeight.Bold,
                                    color = PrimaryAccent,
                                    fontSize = 15.sp
                                )
                            }
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.White,
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
                .padding(bottom = 100.dp), // space for FAB
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Spacer(Modifier.height(4.dp))

            // ── Camera Preview Card ────────────────────────────────────────────
            CameraCard(
                isStreaming = isStreaming,
                uptimeStr = uptimeStr,
                onSurfaceAvailable = onSurfaceAvailable,
                onFlipCamera = onFlipCamera,
                onToggleFlash = onToggleFlash
            )

            // ── Live Stats Row ─────────────────────────────────────────────────
            LiveStatsRow(isStreaming = isStreaming, fps = fps, frameCount = frameCount)

            // ── Session Info Card ──────────────────────────────────────────────
            SessionInfoCard(
                sessionCode = sessionCode,
                serverHost = serverHost,
                onSettingsClick = {
                    hostInput = serverHost
                    showSettingsDialog = true
                }
            )

            // ── Spacer so FAB doesn't overlap ──────────────────────────────────
            Spacer(Modifier.height(8.dp))
        }

        // ── Centered FAB ────────────────────────────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(bottom = 28.dp),
            contentAlignment = Alignment.BottomCenter
        ) {
            StreamFab(
                isStreaming = isStreaming,
                fabColor = fabColor,
                fabScale = fabScale,
                onClick = onToggleStream
            )
        }
    }
}

// ── Camera Preview Card ─────────────────────────────────────────────────────────

@Composable
private fun CameraCard(
    isStreaming: Boolean,
    uptimeStr: String,
    onSurfaceAvailable: (android.view.Surface?) -> Unit,
    onFlipCamera: () -> Unit,
    onToggleFlash: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(3f / 4f),
        shape = RoundedCornerShape(28.dp),
        border = BorderStroke(1.dp, BorderSubtle.copy(alpha = 0.5f)),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFF2F2F7))
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            // TextureView for camera feed
            androidx.compose.ui.viewinterop.AndroidView(
                factory = { ctx ->
                    android.view.TextureView(ctx).apply {
                        surfaceTextureListener = object : android.view.TextureView.SurfaceTextureListener {
                            override fun onSurfaceTextureAvailable(st: android.graphics.SurfaceTexture, w: Int, h: Int) {
                                st.setDefaultBufferSize(720, 1280)
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
                modifier = Modifier.fillMaxSize()
            )

            // Idle overlay
            if (!isStreaming) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(
                                    Color(0xEEFFFFFF),
                                    Color(0xDDFFFFFF)
                                )
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Box(
                            modifier = Modifier
                                .size(72.dp)
                                .clip(CircleShape)
                                .background(Color.Black.copy(alpha = 0.05f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Outlined.VideocamOff,
                                contentDescription = "Camera Offline",
                                tint = TextSecondary,
                                modifier = Modifier.size(36.dp)
                            )
                        }
                        Spacer(Modifier.height(16.dp))
                        Text(
                            "Camera Standby",
                            style = MaterialTheme.typography.titleMedium,
                            color = TextPrimary,
                            fontWeight = FontWeight.SemiBold
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "Tap \"Start Streaming\" below",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary
                        )
                    }
                }
            }

            // Top gradient scrim
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(80.dp)
                    .background(
                        Brush.verticalGradient(
                            colors = listOf(Color.Black.copy(alpha = 0.55f), Color.Transparent)
                        )
                    )
                    .align(Alignment.TopCenter)
            )

            // Bottom gradient scrim
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(72.dp)
                    .background(
                        Brush.verticalGradient(
                            colors = listOf(Color.Transparent, Color.Black.copy(alpha = 0.55f))
                        )
                    )
                    .align(Alignment.BottomCenter)
            )

            // REC badge (top-left)
            if (isStreaming) {
                RecBadge(
                    uptimeStr = uptimeStr,
                    modifier = Modifier.align(Alignment.TopStart).padding(14.dp)
                )
            }

            // Camera controls (top-right)
            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(14.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                CameraOverlayButton(
                    icon = Icons.Outlined.FlipCameraAndroid,
                    label = "Flip",
                    onClick = onFlipCamera
                )
                CameraOverlayButton(
                    icon = Icons.Outlined.FlashOn,
                    label = "Flash",
                    onClick = onToggleFlash
                )
            }

            // AI label (bottom-left)
            Row(
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(14.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color.White.copy(alpha = 0.85f))
                    .padding(horizontal = 10.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(5.dp)
            ) {
                Icon(
                    Icons.Filled.AutoAwesome,
                    contentDescription = null,
                    tint = PrimaryAccent,
                    modifier = Modifier.size(14.dp)
                )
                Text(
                    "AI Crowd Analysis",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextPrimary,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }
    }
}

@Composable
private fun RecBadge(uptimeStr: String, modifier: Modifier = Modifier) {
    // Pulsing dot animation
    val infiniteTransition = rememberInfiniteTransition(label = "recPulse")
    val dotAlpha by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 0.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(700, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "dotAlpha"
    )

    Row(
        modifier = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(Color.White.copy(alpha = 0.85f))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(DangerRed.copy(alpha = dotAlpha))
        )
        Text(
            "REC  $uptimeStr",
            style = MaterialTheme.typography.labelSmall,
            color = TextPrimary,
            fontWeight = FontWeight.Bold,
            letterSpacing = 0.5.sp
        )
    }
}

@Composable
private fun CameraOverlayButton(icon: ImageVector, label: String, onClick: () -> Unit) {
    IconButton(
        onClick = onClick,
        modifier = Modifier
            .size(40.dp)
            .clip(CircleShape)
            .background(Color.White.copy(alpha = 0.85f))
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = TextPrimary,
            modifier = Modifier.size(20.dp)
        )
    }
}

// ── Live Stats Row ──────────────────────────────────────────────────────────────

@Composable
private fun LiveStatsRow(isStreaming: Boolean, fps: Int, frameCount: Int) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        StatChip(
            icon = Icons.Filled.FiberManualRecord,
            iconTint = if (isStreaming) SuccessGreen else TextSubtle,
            label = "Status",
            value = if (isStreaming) "Streaming" else "Standby",
            modifier = Modifier.weight(1f)
        )
        StatChip(
            icon = Icons.Outlined.Speed,
            iconTint = if (isStreaming) PrimaryAccent else TextSubtle,
            label = "Frame Rate",
            value = if (isStreaming) "$fps FPS" else "—",
            modifier = Modifier.weight(1f)
        )
        StatChip(
            icon = Icons.Outlined.PhotoLibrary,
            iconTint = if (isStreaming) SecondaryAccent else TextSubtle,
            label = "Frames",
            value = if (isStreaming) "$frameCount" else "—",
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun StatChip(
    icon: ImageVector,
    iconTint: Color,
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(18.dp),
        color = Color.White,
        border = BorderStroke(1.dp, BorderSubtle.copy(alpha = 0.4f)),
        shadowElevation = 0.dp,
        tonalElevation = 0.dp
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 14.dp),
            horizontalAlignment = Alignment.Start
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = iconTint,
                modifier = Modifier.size(18.dp)
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(2.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

// ── Session Info Card ───────────────────────────────────────────────────────────

@Composable
private fun SessionInfoCard(
    sessionCode: String,
    serverHost: String,
    onSettingsClick: () -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        color = Color.White,
        border = BorderStroke(1.dp, BorderSubtle.copy(alpha = 0.4f)),
        shadowElevation = 0.dp,
        tonalElevation = 0.dp
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            // Header row
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .background(PrimaryGlow),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            Icons.Outlined.Link,
                            contentDescription = null,
                            tint = PrimaryAccent,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                    Text(
                        "Session Info",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                IconButton(
                    onClick = onSettingsClick,
                    modifier = Modifier
                        .size(36.dp)
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.background)
                ) {
                    Icon(
                        Icons.Outlined.Edit,
                        contentDescription = "Edit Settings",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(16.dp)
                    )
                }
            }

            Spacer(Modifier.height(16.dp))
            HorizontalDivider(color = BorderSubtle, thickness = 0.5.dp)
            Spacer(Modifier.height(16.dp))

            // Session Code row
            SessionInfoRow(
                icon = Icons.Outlined.QrCode,
                label = "Session Code",
                value = sessionCode.ifBlank { "Not set" }
            )

            Spacer(Modifier.height(14.dp))

            // Host URL row
            SessionInfoRow(
                icon = Icons.Outlined.Cloud,
                label = "Backend Host",
                value = serverHost.ifBlank { "Not configured" }
            )
        }
    }
}

@Composable
private fun SessionInfoRow(icon: ImageVector, label: String, value: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = PrimaryAccent,
            modifier = Modifier.size(20.dp)
        )
        Column {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(2.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

// ── Stream FAB ──────────────────────────────────────────────────────────────────

@Composable
private fun StreamFab(
    isStreaming: Boolean,
    fabColor: Color,
    fabScale: Float,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .scale(fabScale)
            .clip(RoundedCornerShape(20.dp))
            .background(
                brush = Brush.horizontalGradient(
                    colors = if (isStreaming)
                        listOf(DangerRed, Color(0xFFFF6B6B))
                    else
                        listOf(PrimaryAccent, SecondaryAccent)
                )
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 32.dp, vertical = 16.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Icon(
                imageVector = if (isStreaming) Icons.Filled.Stop else Icons.Filled.Videocam,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(22.dp)
            )
            Text(
                text = if (isStreaming) "Stop Streaming" else "Start Streaming",
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp,
                color = Color.White
            )
        }
    }
}
