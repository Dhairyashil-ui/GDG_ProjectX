// ── DOM refs ────────────────────────────────────────────────────────────────
const videoStream = document.getElementById('videoStream');
const noSignal = document.getElementById('noSignal');
const reconnectingOverlay = document.getElementById('reconnectingOverlay');
const peopleCount = document.getElementById('peopleCount');
const statusBadge = document.getElementById('statusBadge');
const statusIndicator = document.getElementById('statusIndicator');
const statusLabel = document.getElementById('statusLabel');
const timestampDisplay = document.getElementById('timestampDisplay');
const latencyCalc = document.getElementById('latencyCalc');
const densityCard = document.getElementById('densityCard');
const alertBox = document.getElementById('alertBox');
const alertIcon = document.getElementById('alertIcon');
const alertTitle = document.getElementById('alertTitle');
const alertMessage = document.getElementById('alertMessage');
const statusText = document.getElementById('statusText');
const statusDot = document.getElementById('statusDot');
const alertSound = document.getElementById('alertSound');
const sessionInstructions = document.getElementById('sessionInstructions');
const connectedSessionCode = document.getElementById('connectedSessionCode');
const currentSessionCode = document.getElementById('currentSessionCode');

// ── State ───────────────────────────────────────────────────────────────────
let ws;
let lastFrameTime = Date.now();
let checkConnectionInterval;
let redStateStartTime = 0;
let isAlertActive = false;
let activeSessionCode = null;     // current 6-char code
let activeServerHost = null;     // current host (no protocol)

// FIX: Track the pending preload image so we can cancel it if a newer
// frame arrives before the previous one finishes loading. Without this,
// out-of-order onload callbacks could display an older frame over a newer one.
let pendingImage = null;

const STORAGE_KEY = 'crowdpulse_server_host';
const CODE_STORAGE_KEY = 'crowdpulse_session_code';

// ── Helpers ─────────────────────────────────────────────────────────────────
function getServerHost() {
    const raw = document.getElementById('serverUrlInput').value.trim();
    if (raw) return raw.replace(/^wss?:\/\//, '').replace(/^https?:\/\//, '');
    return localStorage.getItem(STORAGE_KEY) || '';
}

function getHttpBase(host) {
    const isLocal = host.startsWith('localhost') || host.startsWith('127.') || host.startsWith('10.0.');
    return isLocal ? `http://${host}` : `https://${host}`;
}

function getWsBase(host) {
    const isLocal = host.startsWith('localhost') || host.startsWith('127.') || host.startsWith('10.0.');
    return isLocal ? `ws://${host}` : `wss://${host}`;
}

// ── UI Connection ─────────────────────────────────────────────────────────
function connectFromUI() {
    const host = getServerHost();
    const code = document.getElementById('sessionCodeInput').value.trim().toUpperCase();

    if (!host) { alert('Please enter the backend server URL first.'); return; }
    if (code.length < 6) { alert('Please enter a valid 6-char session code.'); return; }

    localStorage.setItem(STORAGE_KEY, host);
    localStorage.setItem(CODE_STORAGE_KEY, code);

    // Show session connection success
    sessionInstructions.classList.remove('hidden');
    connectedSessionCode.textContent = code;
    currentSessionCode.textContent = code;

    connectToDashboard(host, code);
}

// ── WebSocket connection (code-scoped) ────────────────────────────────────────
function connectToDashboard(host, code) {
    activeServerHost = host;
    activeSessionCode = code;

    const wsUrl = `${getWsBase(host)}/ws/dashboard/${code}`;

    statusText.textContent = 'Connecting...';
    statusDot.className = 'w-3 h-3 rounded-full bg-amber-400 animate-pulse';

    if (ws) {
        ws.close();
        ws = null;
    }

    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        statusText.textContent = 'Connected';
        statusDot.className = 'w-3 h-3 rounded-full bg-emerald-500 animate-pulse';
        clearInterval(checkConnectionInterval);
        checkConnectionInterval = setInterval(checkFallbackState, 1000);
    };

    ws.onclose = (event) => {
        statusText.textContent = 'Disconnected';
        statusDot.className = 'w-3 h-3 rounded-full bg-red-500';
        clearInterval(checkConnectionInterval);

        // If the backend restarted or the session expired, it sends code 4404
        if (event.code === 4404) {
            console.warn("Session expired or invalid code from server.");
            alert("This session has ended (the server restarted or connection expired). Please click 'New Session' to start a new one.");
            localStorage.removeItem(CODE_STORAGE_KEY);
            activeSessionCode = null;
            return; // Don't auto-retry
        }

        // Auto-retry after 4 seconds with the same code (don't generate a new one)
        setTimeout(() => {
            if (activeSessionCode && activeServerHost) {
                connectToDashboard(activeServerHost, activeSessionCode);
            }
        }, 4000);
    };

    ws.onerror = () => {
        statusText.textContent = 'Error';
        statusDot.className = 'w-3 h-3 rounded-full bg-red-500';
    };

    ws.onmessage = (event) => {
        if (typeof event.data === "string") {
            const data = JSON.parse(event.data);
            
            const serverToBrowserMs = Date.now() - (data.timestamp * 1000);
            console.log(`%c[NETWORK] Received frame JSON metadata. Network Transit Delay: ~${Math.floor(serverToBrowserMs)}ms`, 'color: #3b82f6; font-weight: bold;');
            
            updateDashboard(data);
        } else if (event.data instanceof ArrayBuffer) {
            console.log(`%c[RENDER] ArrayBuffer frame arrived (${event.data.byteLength} bytes). Decoding...`, 'color: #8b5cf6;');
            renderFrame(event.data);
        }
    };
}

// ── Frame rendering ───────────────────────────────────────────────────────────
/**
 * FIX: Replaced direct img.src swap with an off-screen preload + atomic swap.
 *
 * OLD behaviour:
 *   videoStream.src = newBlobUrl
 *   → The <img> tag briefly goes blank while the browser decodes the new JPEG.
 *   → At 10-12 FPS this blank gap is visible as constant flickering / strobing.
 *
 * NEW behaviour:
 *   1. Create an off-screen Image() and set its src to the new blob URL.
 *   2. Only swap videoStream.src AFTER the new image is fully decoded (onload).
 *   3. Revoke the OLD blob URL only after the swap — never before — so there
 *      is no moment where the visible <img> has no valid src.
 *   4. Cancel any in-flight preload if a newer frame arrives first, preventing
 *      out-of-order frames from appearing (fast network bursts can deliver
 *      two frames before the first onload fires).
 */
// Use a counter to prevent painting older frames if they arrive out-of-order
let currentFrameId = 0;

function renderFrame(arrayBuffer) {
    const blob = new Blob([arrayBuffer], { type: "image/jpeg" });
    const newUrl = URL.createObjectURL(blob);
    
    currentFrameId++;
    const thisFrameId = currentFrameId;
    const decodeStart = Date.now();

    const img = new Image();

    img.onload = () => {
        const decodeTime = Date.now() - decodeStart;
        // If a newer frame started loading while we were decoding, discard this one
        if (thisFrameId !== currentFrameId) {
            console.log(`[RENDER] 🚨 Dropped older frame #${thisFrameId} (Superseded by #${currentFrameId}) after ${decodeTime}ms decode.`);
            URL.revokeObjectURL(newUrl);
            return;
        }

        console.log(`%c[RENDER] ✅ Painted frame #${thisFrameId} successfully in ${decodeTime}ms.`, 'color: #10b981;');

        // Revoke the OLD blob URL now that the new frame is ready
        if (window.previousImageUrl) {
            URL.revokeObjectURL(window.previousImageUrl);
        }

        // Atomic swap — browser paints new frame with zero blank gap
        videoStream.src = newUrl;
        window.previousImageUrl = newUrl;

        // Show stream, hide placeholders
        videoStream.classList.remove('hidden');
        noSignal.classList.add('hidden');
        reconnectingOverlay.classList.replace('opacity-100', 'opacity-0');
        reconnectingOverlay.classList.add('pointer-events-none');

        lastFrameTime = Date.now();
    };

    img.onerror = () => {
        URL.revokeObjectURL(newUrl);
    };

    img.src = newUrl;
}

// ── Dashboard update ─────────────────────────────────────────────────────────
function updateDashboard(data) {
    peopleCount.textContent = data.count;

    const d = new Date();
    timestampDisplay.textContent = d.toLocaleTimeString();

    applyStatusColor(data.status);

    if (data.status === "RED") {
        if (redStateStartTime === 0) redStateStartTime = Date.now();
        const duration = (Date.now() - redStateStartTime) / 1000;
        if (duration > 10 && !isAlertActive) triggerAlert();
    } else {
        redStateStartTime = 0;
        if (isAlertActive) clearAlert();
    }
}

function applyStatusColor(status) {
    statusLabel.textContent = status;
    statusBadge.className = "status-badge w-full transition-colors duration-300";
    densityCard.className = "bento-card p-6 border-t-[6px] transition-all duration-500";

    if (status === "GREEN") {
        statusIndicator.className = "w-2.5 h-2.5 rounded-full bg-[#10b981]";
        statusBadge.classList.add('bg-[#ecfdf5]', 'text-[#059669]');
        densityCard.classList.add('border-t-[#10b981]');
    } else if (status === "YELLOW") {
        statusIndicator.className = "w-2.5 h-2.5 rounded-full bg-[#f59e0b] animate-pulse";
        statusBadge.classList.add('bg-[#fffbeb]', 'text-[#d97706]');
        densityCard.classList.add('border-t-[#f59e0b]');
    } else if (status === "RED") {
        statusIndicator.className = "w-2.5 h-2.5 rounded-full bg-[#e11d48] animate-ping";
        statusBadge.classList.add('bg-[#fff1f2]', 'text-[#be123c]');
        densityCard.classList.add('border-t-[#e11d48]', 'pulse-red-soft');
    }
}

function triggerAlert() {
    isAlertActive = true;
    alertBox.className = "bento-card p-6 border border-[#fecdd3] opacity-100 transition-all duration-300 pulse-red-soft bg-[#fff1f2]";
    alertIcon.className = "w-10 h-10 rounded-full bg-[#ffe4e6] flex items-center justify-center text-[#e11d48] animate-bounce";
    alertIcon.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>`;
    alertTitle.textContent = "CRITICAL SERVER ALERT";
    alertTitle.className = "font-bold text-[#be123c] text-sm";
    alertMessage.textContent = "High density threshold exceeded for >10 seconds. Dispatching security.";
    alertMessage.className = "text-xs text-[#e11d48] mt-1 font-bold leading-relaxed";

    try {
        alertSound.play().catch(e => console.log("Sound blocked by browser policy"));
    } catch (e) { }
}

function clearAlert() {
    isAlertActive = false;
    alertBox.className = "bento-card p-6 bg-[#f5f5f7] border border-transparent opacity-60 transition-all duration-300";
    alertIcon.className = "w-10 h-10 rounded-full bg-[#e8e8ed] flex items-center justify-center text-[#86868b]";
    alertIcon.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
    alertTitle.textContent = "Threat Monitoring";
    alertTitle.className = "font-bold text-[#1d1d1f] text-sm";
    alertMessage.textContent = "Continuous density metrics within expected parameters.";
    alertMessage.className = "text-xs text-[#86868b] mt-1 font-medium leading-relaxed";
}

function checkFallbackState() {
    // If the video stream is still hidden, we haven't received the first frame yet.
    // Let the default "Waiting for mobile stream..." UI handle it.
    if (videoStream.classList.contains('hidden')) {
        return;
    }

    const timeSinceLastFrame = Date.now() - lastFrameTime;
    latencyCalc.textContent = `${Math.floor(timeSinceLastFrame / 1000)}s ago`;

    if (timeSinceLastFrame > 35000) {
        reconnectingOverlay.classList.replace('opacity-0', 'opacity-100');
        reconnectingOverlay.classList.remove('pointer-events-none');
    }
}

// ── On load: restore saved session ───────────────────────────────────────────
const savedHost = localStorage.getItem(STORAGE_KEY);
const savedCode = localStorage.getItem(CODE_STORAGE_KEY);

if (savedHost) {
    document.getElementById('serverUrlInput').value = savedHost;
}
if (savedCode) {
    document.getElementById('sessionCodeInput').value = savedCode;
}

if (savedHost && savedCode) {
    connectFromUI();
}