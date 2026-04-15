package com.crowdpulse.camera.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.crowdpulse.camera.R
import com.crowdpulse.camera.ui.theme.*

@Composable
fun AuthScreen(
    onGoogleSignInClick: () -> Unit
) {
    val infiniteTransition = rememberInfiniteTransition(label = "bg")

    // ── Floating orb animations ───────────────────────────────────────────────
    val orb1X by infiniteTransition.animateFloat(
        initialValue = 0.15f, targetValue = 0.25f,
        animationSpec = infiniteRepeatable(tween(6000, easing = EaseInOutSine), RepeatMode.Reverse),
        label = "o1x"
    )
    val orb1Y by infiniteTransition.animateFloat(
        initialValue = 0.1f, targetValue = 0.2f,
        animationSpec = infiniteRepeatable(tween(7000, easing = EaseInOutSine), RepeatMode.Reverse),
        label = "o1y"
    )
    val orb2X by infiniteTransition.animateFloat(
        initialValue = 0.75f, targetValue = 0.85f,
        animationSpec = infiniteRepeatable(tween(8000, easing = EaseInOutSine), RepeatMode.Reverse),
        label = "o2x"
    )
    val orb2Y by infiniteTransition.animateFloat(
        initialValue = 0.6f, targetValue = 0.75f,
        animationSpec = infiniteRepeatable(tween(5000, easing = EaseInOutSine), RepeatMode.Reverse),
        label = "o2y"
    )
    val orb3X by infiniteTransition.animateFloat(
        initialValue = 0.5f, targetValue = 0.6f,
        animationSpec = infiniteRepeatable(tween(9000, easing = EaseInOutSine), RepeatMode.Reverse),
        label = "o3x"
    )
    val orb3Y by infiniteTransition.animateFloat(
        initialValue = 0.8f, targetValue = 0.9f,
        animationSpec = infiniteRepeatable(tween(6500, easing = EaseInOutSine), RepeatMode.Reverse),
        label = "o3y"
    )

    // ── Card entrance ─────────────────────────────────────────────────────────
    var cardVisible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { cardVisible = true }
    val cardAlpha by animateFloatAsState(
        if (cardVisible) 1f else 0f, tween(700, 200), label = "card"
    )
    val cardOffset by animateFloatAsState(
        if (cardVisible) 0f else 60f, tween(700, 200, EaseOutQuint), label = "cardY"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BackgroundLight),
        contentAlignment = Alignment.Center
    ) {
        // ── Animated background canvas ─────────────────────────────────────────
        androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
            // Orb 1 — violet
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(SecondaryAccent.copy(alpha = 0.15f), Color.Transparent),
                    center = Offset(size.width * orb1X, size.height * orb1Y),
                    radius = 380f
                ),
                radius = 380f,
                center = Offset(size.width * orb1X, size.height * orb1Y)
            )
            // Orb 2 — cyan
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(PrimaryAccent.copy(alpha = 0.15f), Color.Transparent),
                    center = Offset(size.width * orb2X, size.height * orb2Y),
                    radius = 420f
                ),
                radius = 420f,
                center = Offset(size.width * orb2X, size.height * orb2Y)
            )
            // Orb 3 — violet subtle bottom
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(SecondaryDim.copy(alpha = 0.1f), Color.Transparent),
                    center = Offset(size.width * orb3X, size.height * orb3Y),
                    radius = 300f
                ),
                radius = 300f,
                center = Offset(size.width * orb3X, size.height * orb3Y)
            )
        }

        // ── Glass card ─────────────────────────────────────────────────────────
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp)
                .offset(y = cardOffset.dp)
                .alpha(cardAlpha)
                .clip(RoundedCornerShape(32.dp))
                .background(SurfaceCard)
                .border(
                    width = 1.dp,
                    brush = Brush.linearGradient(
                        colors = listOf(
                            BorderSubtle,
                            BorderSubtle,
                            Color.Transparent
                        )
                    ),
                    shape = RoundedCornerShape(32.dp)
                )
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Logo mark
            Box(
                modifier = Modifier
                    .size(80.dp),
                contentAlignment = Alignment.Center
            ) {
                Image(
                    painter = painterResource(id = R.drawable.app_logo),
                    contentDescription = "Croudify Logo",
                    modifier = Modifier.fillMaxSize().clip(CircleShape)
                )
            }

            Spacer(Modifier.height(20.dp))

            Text(
                text = "Croudify",
                style = MaterialTheme.typography.headlineMedium,
                color = TextPrimary,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(6.dp))
            Text(
                text = "Intelligence at Scale",
                style = MaterialTheme.typography.bodyMedium,
                color = TextSecondary,
                textAlign = TextAlign.Center,
                letterSpacing = 1.sp
            )

            Spacer(Modifier.height(28.dp))

            // Feature chips
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                FeatureChip("Real-time", modifier = Modifier.weight(1f))
                FeatureChip("Secure",   modifier = Modifier.weight(1f))
                FeatureChip("AI-Powered",modifier = Modifier.weight(1f))
            }

            Spacer(Modifier.height(32.dp))

            // Divider label
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                HorizontalDivider(modifier = Modifier.weight(1f), color = BorderSubtle)
                Text(
                    text = "  sign in to continue  ",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextSubtle
                )
                HorizontalDivider(modifier = Modifier.weight(1f), color = BorderSubtle)
            }

            Spacer(Modifier.height(24.dp))

            // ── Google Sign-In button ──────────────────────────────────────────
            GoogleSignInButton(onClick = onGoogleSignInClick)

            Spacer(Modifier.height(24.dp))

            Text(
                text = "By continuing, you agree to our Terms of Service\nand Privacy Policy",
                style = MaterialTheme.typography.bodySmall,
                color = TextSubtle,
                textAlign = TextAlign.Center,
                lineHeight = 18.sp
            )
        }
    }
}

@Composable
private fun FeatureChip(label: String, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(SurfaceElevated)
            .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
            .padding(vertical = 8.dp, horizontal = 4.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = TextSecondary,
            textAlign = TextAlign.Center,
            maxLines = 1
        )
    }
}

@Composable
private fun GoogleSignInButton(onClick: () -> Unit) {
    var pressed by remember { mutableStateOf(false) }
    val scale by animateFloatAsState(
        if (pressed) 0.96f else 1f, spring(stiffness = Spring.StiffnessMedium), label = "btnScale"
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .graphicsLayer { scaleX = scale; scaleY = scale }
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceElevated)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .clickable {
                pressed = true
                onClick()
            }
            .padding(horizontal = 20.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            // Google G icon
            GoogleGIcon()
            Spacer(Modifier.width(12.dp))
            Text(
                text = "Continue with Google",
                style = MaterialTheme.typography.titleSmall,
                color = TextPrimary,
                fontWeight = FontWeight.SemiBold
            )
        }
    }

    LaunchedEffect(pressed) {
        if (pressed) {
            kotlinx.coroutines.delay(200)
            pressed = false
        }
    }
}

@Composable
private fun GoogleGIcon() {
    androidx.compose.foundation.Canvas(modifier = Modifier.size(22.dp)) {
        val cx = size.width / 2f
        val cy = size.height / 2f
        val r  = size.minDimension / 2f

        // Draw a simplified Google G using arcs
        drawArc(color = Color(0xFF4285F4), startAngle = -30f,  sweepAngle = 120f, useCenter = false, style = Stroke(width = 5f))
        drawArc(color = Color(0xFF34A853), startAngle =  90f,  sweepAngle = 90f,  useCenter = false, style = Stroke(width = 5f))
        drawArc(color = Color(0xFFFBBC05), startAngle = 180f,  sweepAngle = 90f,  useCenter = false, style = Stroke(width = 5f))
        drawArc(color = Color(0xFFEA4335), startAngle = 270f,  sweepAngle = 60f,  useCenter = false, style = Stroke(width = 5f))

        // Horizontal bar of G
        drawLine(
            color = Color(0xFF4285F4),
            start = Offset(cx, cy),
            end   = Offset(cx + r, cy),
            strokeWidth = 5f,
            cap = StrokeCap.Round
        )
    }
}
