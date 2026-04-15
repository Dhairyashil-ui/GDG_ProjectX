package com.crowdpulse.camera.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.Image
import androidx.compose.ui.res.painterResource
import com.crowdpulse.camera.R
import com.crowdpulse.camera.ui.theme.*
import kotlinx.coroutines.delay

@Composable
fun SplashScreen(
    onSplashComplete: () -> Unit
) {
    val currentOnSplashComplete by rememberUpdatedState(onSplashComplete)
    LaunchedEffect(Unit) {
        delay(2600)
        currentOnSplashComplete()
    }

    // ── Pulsing rings ─────────────────────────────────────────────────────────
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")

    val ring1Scale by infiniteTransition.animateFloat(
        initialValue = 0.6f, targetValue = 1.4f,
        animationSpec = infiniteRepeatable(tween(2000, easing = EaseOutSine), RepeatMode.Restart),
        label = "ring1"
    )
    val ring1Alpha by infiniteTransition.animateFloat(
        initialValue = 0.6f, targetValue = 0f,
        animationSpec = infiniteRepeatable(tween(2000, easing = EaseOutSine), RepeatMode.Restart),
        label = "ring1a"
    )
    val ring2Scale by infiniteTransition.animateFloat(
        initialValue = 0.6f, targetValue = 1.5f,
        animationSpec = infiniteRepeatable(tween(2000, 350, easing = EaseOutSine), RepeatMode.Restart),
        label = "ring2"
    )
    val ring2Alpha by infiniteTransition.animateFloat(
        initialValue = 0.5f, targetValue = 0f,
        animationSpec = infiniteRepeatable(tween(2000, 350, easing = EaseOutSine), RepeatMode.Restart),
        label = "ring2a"
    )
    val ring3Scale by infiniteTransition.animateFloat(
        initialValue = 0.6f, targetValue = 1.4f,
        animationSpec = infiniteRepeatable(tween(2000, 700, easing = EaseOutSine), RepeatMode.Restart),
        label = "ring3"
    )
    val ring3Alpha by infiniteTransition.animateFloat(
        initialValue = 0.4f, targetValue = 0f,
        animationSpec = infiniteRepeatable(tween(2000, 700, easing = EaseOutSine), RepeatMode.Restart),
        label = "ring3a"
    )

    // ── Shimmer progress ──────────────────────────────────────────────────────
    val shimmerOffset by infiniteTransition.animateFloat(
        initialValue = -1f, targetValue = 2f,
        animationSpec = infiniteRepeatable(tween(1400, easing = EaseInOutSine)),
        label = "shimmer"
    )

    // ── Logo inner rotation ────────────────────────────────────────────────────
    val logoRotation by infiniteTransition.animateFloat(
        initialValue = 0f, targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(8000, easing = LinearEasing)),
        label = "rotate"
    )

    // ── Word reveal alpha (staggered) ─────────────────────────────────────────
    var word1Alpha by remember { mutableStateOf(0f) }
    var word2Alpha by remember { mutableStateOf(0f) }
    var tagAlpha   by remember { mutableStateOf(0f) }
    LaunchedEffect(Unit) {
        delay(500);  word1Alpha = 1f
        delay(150);  word2Alpha = 1f
        delay(300);  tagAlpha   = 1f
    }
    val animWord1 by animateFloatAsState(word1Alpha, tween(500), label = "w1")
    val animWord2 by animateFloatAsState(word2Alpha, tween(500), label = "w2")
    val animTag   by animateFloatAsState(tagAlpha,   tween(600), label = "tag")

    // ── Layout ────────────────────────────────────────────────────────────────
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BackgroundLight),
        contentAlignment = Alignment.Center
    ) {
        // Background ambient glow blobs
        Canvas(modifier = Modifier.fillMaxSize()) {
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(SecondaryGlow, Color.Transparent),
                    center = Offset(size.width * 0.2f, size.height * 0.25f),
                    radius = 400f
                ),
                radius = 400f,
                center = Offset(size.width * 0.2f, size.height * 0.25f)
            )
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(PrimaryGlow, Color.Transparent),
                    center = Offset(size.width * 0.8f, size.height * 0.75f),
                    radius = 350f
                ),
                radius = 350f,
                center = Offset(size.width * 0.8f, size.height * 0.75f)
            )
        }

        // Content
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Logo Image with scale-in animation
            val logoScale by infiniteTransition.animateFloat(
                initialValue = 0.8f, targetValue = 1.05f,
                animationSpec = infiniteRepeatable(tween(2600, easing = EaseOutSine), RepeatMode.Reverse),
                label = "logo_scale"
            )

            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(160.dp)
                    .scale(logoScale)
            ) {
                Image(
                    painter = painterResource(id = R.drawable.app_logo),
                    contentDescription = "Croudify Logo",
                    modifier = Modifier.fillMaxSize().clip(CircleShape)
                )
            }

            Spacer(Modifier.height(36.dp))

            // Brand wordmark
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "Croud",
                    style = MaterialTheme.typography.displayMedium,
                    color = TextPrimary,
                    modifier = Modifier.alpha(animWord1)
                )
                Text(
                    text = "ify",
                    style = MaterialTheme.typography.displayMedium,
                    color = PrimaryAccent,
                    modifier = Modifier.alpha(animWord2)
                )
            }

            Spacer(Modifier.height(12.dp))

            // Tagline
            Text(
                text = "Intelligence at Scale",
                style = MaterialTheme.typography.titleMedium,
                color = TextSecondary,
                letterSpacing = 2.sp,
                modifier = Modifier.alpha(animTag)
            )

            Spacer(Modifier.height(72.dp))

            // Shimmer progress bar
            Box(
                modifier = Modifier
                    .width(180.dp)
                    .height(2.dp)
                    .clip(CircleShape)
                    .background(SurfaceElevated)
                    .alpha(animTag)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            Brush.horizontalGradient(
                                colors = listOf(
                                    Color.Transparent,
                                    PrimaryAccent,
                                    Color.Transparent
                                ),
                                startX = shimmerOffset * 400f,
                                endX   = shimmerOffset * 400f + 200f
                            )
                        )
                )
            }
        }
    }
}

@Composable
private fun PulseRing(scale: Float, alpha: Float, size: Dp, color: Color) {
    Box(
        modifier = Modifier
            .size(size)
            .scale(scale)
            .alpha(alpha)
            .clip(CircleShape)
            .background(color.copy(alpha = 0.12f))
    )
}
