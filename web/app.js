// ── DOM refs ────────────────────────────────────────────────────────────────
const videoStream        = document.getElementById('videoStream');
const noSignal           = document.getElementById('noSignal');
const reconnectingOverlay= document.getElementById('reconnectingOverlay');
const peopleCount        = document.getElementById('peopleCount');
const statusBadge        = document.getElementById('statusBadge');
const statusIndicator    = document.getElementById('statusIndicator');
const statusLabel        = document.getElementById('statusLabel');
const timestampDisplay   = document.getElementById('timestampDisplay');
const latencyCalc        = document.getElementById('latencyCalc');
const densityCard        = document.getElementById('densityCard');
const alertBox           = document.getElementById('alertBox');
const alertIcon          = document.getElementById('alertIcon');
const alertTitle         = document.getElementById('alertTitle');
const alertMessage       = document.getElementById('alertMessage');
const statusText         = document.getElementById('statusText');
const statusDot          = document.getElementById('statusDot');
const alertSound         = document.getElementById('alertSound');
const sessionPanel       = document.getElementById('sessionPanel');
const codeDisplay        = document.getElementById('codeDisplay');
const qrCanvas           = document.getElementById('qrCanvas');
const currentSessionCode = document.getElementById('currentSessionCode');
const newSessionBtn      = document.getElementById('newSessionBtn');
const copyBtn            = document.getElementById('copyBtn');

// ── State ───────────────────────────────────────────────────────────────────
let ws;
let lastFrameTime       = Date.now();
let checkConnectionInterval;
let redStateStartTime   = 0;
let isAlertActive       = false;
let activeSessionCode   = null;     // current 6-char code
let activeServerHost    = null;     // current host (no protocol)

const STORAGE_KEY       = 'crowdpulse_server_host';
const CODE_STORAGE_KEY  = 'crowdpulse_session_code';

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

// ── Session creation ─────────────────────────────────────────────────────────
async function createNewSession() {
    const host = getServerHost();
    if (!host) {
        alert('Please enter the backend server URL first.');
        return;
    }

    // Save & persist the host
    localStorage.setItem(STORAGE_KEY, host);
    activeServerHost = host;
    document.getElementById('serverUrlInput').value = host;

    // Close any existing WS
    if (ws) { ws.close(); ws = null; }

    // Show spinner on button
    newSessionBtn.disabled = true;
    newSessionBtn.innerHTML = `
        <svg class="w-4 h-4 animate-spin-slow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
        </svg>
        Creating...`;

    try {
        const res = await fetch(`${getHttpBase(host)}/session/create`, { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const code = data.code;

        activeSessionCode = code;
        localStorage.setItem(CODE_STORAGE_KEY, code);

        showSessionCode(code);
        connectToDashboard(host, code);

    } catch (err) {
        console.error('Session creation failed', err);
        alert(`Failed to reach server at "${host}".\n\nMake sure the backend is running and the URL is correct.`);
    } finally {
        newSessionBtn.disabled = false;
        newSessionBtn.innerHTML = `
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
            </svg>
            New Session`;
    }
}

// ── Show the 6-char code visually ─────────────────────────────────────────────
function showSessionCode(code) {
    // Render each char as a styled box
    codeDisplay.innerHTML = '';
    for (const ch of code) {
        const span = document.createElement('span');
        span.className = 'code-char animate-slide-down';
        span.textContent = ch;
        codeDisplay.appendChild(span);
    }

    // Update sidebar badge
    currentSessionCode.textContent = code;

    // Generate QR code using qrcodejs (new QRCode(element, opts))
    qrCanvas.innerHTML = '';          // clear any previous QR
    new QRCode(qrCanvas, {
        text: code,
        width: 112,
        height: 112,
        colorDark: '#1e3a8a',
        colorLight: '#eff6ff',
        correctLevel: QRCode.CorrectLevel.M
    });

    // Show panel
    sessionPanel.classList.remove('hidden');
}

// ── Copy code to clipboard ───────────────────────────────────────────────────
function copyCode() {
    if (!activeSessionCode) return;
    navigator.clipboard.writeText(activeSessionCode).then(() => {
        copyBtn.classList.add('copy-btn-success');
        copyBtn.textContent = '✓ Copied!';
        setTimeout(() => {
            copyBtn.classList.remove('copy-btn-success');
            copyBtn.innerHTML = `
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
                </svg>
                Copy Code`;
        }, 2000);
    });
}

// ── WebSocket connection (code-scoped) ────────────────────────────────────────
function connectToDashboard(host, code) {
    activeServerHost = host;
    activeSessionCode = code;

    const wsUrl = `${getWsBase(host)}/ws/dashboard/${code}`;

    statusText.textContent = 'Connecting...';
    statusDot.className = 'w-3 h-3 rounded-full bg-amber-400 animate-pulse';

    ws = new WebSocket(wsUrl);

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
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };
}

// ── Dashboard update ─────────────────────────────────────────────────────────
function updateDashboard(data) {
    lastFrameTime = Date.now();
    reconnectingOverlay.classList.replace('opacity-100', 'opacity-0');
    reconnectingOverlay.classList.add('pointer-events-none');
    
    if (data.image_b64) {
        videoStream.src = "data:image/jpeg;base64," + data.image_b64;
        videoStream.classList.remove('hidden');
        noSignal.classList.add('hidden');
    }

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
    statusBadge.className = "inline-flex w-full justify-center items-center gap-2 px-4 py-3 rounded-xl font-bold transition-colors duration-300";
    densityCard.className = "glass-panel rounded-2xl p-6 transition-all duration-500 border-2";

    if (status === "GREEN") {
        statusIndicator.className = "w-2.5 h-2.5 rounded-full bg-emerald-500";
        statusBadge.classList.add('bg-emerald-100', 'text-emerald-700');
        densityCard.classList.add('border-emerald-200');
    } else if (status === "YELLOW") {
        statusIndicator.className = "w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse";
        statusBadge.classList.add('bg-amber-100', 'text-amber-700');
        densityCard.classList.add('border-amber-200');
    } else if (status === "RED") {
        statusIndicator.className = "w-2.5 h-2.5 rounded-full bg-red-600 animate-ping";
        statusBadge.classList.add('bg-red-100', 'text-red-700');
        densityCard.classList.add('border-red-300', 'shadow-lg', 'shadow-red-500/20');
    }
}

function triggerAlert() {
    isAlertActive = true;
    alertBox.className = "glass-panel rounded-2xl p-6 border-l-4 border-red-500 opacity-100 transition-all duration-300 pulse-red bg-red-50";
    alertIcon.className = "p-2 rounded-lg bg-red-100 text-red-600 animate-bounce";
    alertIcon.innerHTML = `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>`;
    alertTitle.textContent = "CRITICAL SERVER ALERT";
    alertTitle.classList.add('text-red-700');
    alertMessage.textContent = "High density threshold exceeded for >10 seconds. Dispatching security.";
    alertMessage.classList.replace('text-slate-500', 'text-red-600');
    
    try {
        alertSound.play().catch(e => console.log("Sound blocked by browser policy"));
    } catch(e) {}
}

function clearAlert() {
    isAlertActive = false;
    alertBox.className = "glass-panel rounded-2xl p-6 border-l-4 border-slate-300 opacity-50 transition-all duration-300";
    alertIcon.className = "p-2 rounded-lg bg-slate-100 text-slate-400";
    alertIcon.innerHTML = `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
    alertTitle.textContent = "System Monitoring";
    alertTitle.classList.remove('text-red-700');
    alertMessage.textContent = "Density stabilized.";
    alertMessage.classList.replace('text-red-600', 'text-slate-500');
}

function checkFallbackState() {
    // If the video stream is still hidden, we haven't received the first frame yet.
    // Let the default "Waiting for mobile stream..." UI handle it.
    if (videoStream.classList.contains('hidden')) {
        return;
    }

    const timeSinceLastFrame = Date.now() - lastFrameTime;
    latencyCalc.textContent = `${Math.floor(timeSinceLastFrame / 1000)}s ago`;

    if (timeSinceLastFrame > 5000) {
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
if (savedHost && savedCode) {
    // Restore the UI but let the user click "New Session" for a fresh code,
    // or auto-reconnect if the session might still be alive.
    showSessionCode(savedCode);
    connectToDashboard(savedHost, savedCode);
}
