// ==========================================================================
// BMW M TECHNICAL INTELLIGENCE — FRONTEND CLIENT CONTROLLER
// Adhering strictly to DESIGN-bmw-m.md
// ==========================================================================

// DOM Elements
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('userInput');
const chatBox = document.getElementById('chatBox');
const typingIndicator = document.getElementById('typingIndicator');
const sendBtn = document.getElementById('sendBtn');
const chatScrollContainer = document.getElementById('chatScrollContainer');
const historyList = document.getElementById('historyList');
const newChatBtn = document.getElementById('newChatBtn');
const topNewChatBtn = document.getElementById('topNewChatBtn');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const sidebar = document.getElementById('sidebar');
const welcomeHero = document.getElementById('welcomeHero');

// State Management
let sessions = JSON.parse(localStorage.getItem('bmw_m_rag_sessions')) || {};
let currentSessionId = localStorage.getItem('bmw_m_rag_current_session') || null;

// Initialize Session
if (!currentSessionId || !sessions[currentSessionId]) {
    createNewSession();
} else {
    renderSidebar();
    loadSession(currentSessionId);
}

// Generate Session ID
function generateId() {
    return 'bmw_m_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now().toString(36);
}

// Create New Session
function createNewSession() {
    currentSessionId = generateId();
    sessions[currentSessionId] = {
        title: "NEW INQUIRY",
        timestamp: Date.now(),
        messages: []
    };
    saveState();
    renderSidebar();
    loadSession(currentSessionId);
}

// Save to LocalStorage
function saveState() {
    localStorage.setItem('bmw_m_rag_sessions', JSON.stringify(sessions));
    localStorage.setItem('bmw_m_rag_current_session', currentSessionId);
}

// Render Sidebar Sessions
function renderSidebar() {
    historyList.innerHTML = '';
    const sortedSessions = Object.entries(sessions).sort((a, b) => b[1].timestamp - a[1].timestamp);

    sortedSessions.forEach(([id, sessionData]) => {
        const li = document.createElement('li');
        li.textContent = sessionData.title;
        if (id === currentSessionId) li.classList.add('active');

        li.addEventListener('click', () => {
            currentSessionId = id;
            saveState();
            renderSidebar();
            loadSession(currentSessionId);
            if (window.innerWidth <= 768) sidebar.classList.remove('open');
        });

        historyList.appendChild(li);
    });
}

// Load Session
function loadSession(id) {
    chatBox.innerHTML = '';
    const sessionMessages = sessions[id].messages;

    if (sessionMessages.length === 0) {
        if (welcomeHero) welcomeHero.style.display = 'flex';
    } else {
        if (welcomeHero) welcomeHero.style.display = 'none';
        sessionMessages.forEach(msg => {
            renderMessageToUI(msg);
        });
    }
    scrollToBottom();
}

// Render Message
function renderMessageToUI(msgObj) {
    const isUser = msgObj.isUser;
    const text = msgObj.text;
    const citations = msgObj.citations || [];
    const telemetry = msgObj.telemetry || null;
    const usedFallback = msgObj.usedFallback || false;

    const messageEl = document.createElement('div');
    messageEl.className = `message ${isUser ? 'user-message' : 'ai-message'}`;

    const containerEl = document.createElement('div');
    containerEl.className = 'message-container';

    // Header Meta
    const headerEl = document.createElement('div');
    headerEl.className = 'message-meta-header';

    if (isUser) {
        headerEl.innerHTML = `
            <div class="message-meta-left">
                <span>[OPERATOR INQUIRY]</span>
            </div>
            <span class="message-timestamp">${new Date(msgObj.timestamp || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        `;
    } else {
        headerEl.innerHTML = `
            <div class="message-meta-left">
                <span class="mini-m-stripes">
                    <span class="mini-stripe mini-blue-lt"></span>
                    <span class="mini-stripe mini-blue-dk"></span>
                    <span class="mini-stripe mini-red"></span>
                </span>
                <span>BMW M TECHNICAL INTELLIGENCE // S58 ENGINE DOCS</span>
            </div>
            <span class="message-timestamp">DETERMINISTIC RETRIEVAL</span>
        `;
    }
    containerEl.appendChild(headerEl);

    // Message Body Text
    const textEl = document.createElement('div');
    textEl.className = 'message-text';

    if (isUser) {
        textEl.innerHTML = `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
    } else {
        let aiHtml = '';

        // Guardrail Fallback Warning Box if activated
        if (usedFallback) {
            aiHtml += `
                <div class="fallback-warning-box">
                    <span class="fallback-icon">▲</span>
                    <span>GUARDRAIL ACTIVATED: QUERY EXCEEDS S58 MANUAL SCOPE OR FAILED CITATION VERIFICATION</span>
                </div>
            `;
        }

        // Markdown Parsing
        if (window.marked) {
            aiHtml += marked.parse(text);
        } else {
            aiHtml += `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
        }

        // Verified Citations Panel
        if (citations && citations.length > 0) {
            const uniqueId = 'cit_group_' + Math.random().toString(36).substring(2, 9);
            let citHtml = `
                <div class="citations-panel">
                    <div class="citations-header">
                        <span class="citations-label">
                            VERIFIED PROVENANCE (${citations.length} CHUNKS CITED)
                        </span>
                        <span class="provenance-status">● VERIFIED GROUND TRUTH</span>
                    </div>
                    <div class="citations-grid">
            `;

            citations.forEach((c, idx) => {
                const drawerId = `${uniqueId}_drawer_${idx}`;
                citHtml += `
                    <div class="citation-pill-item">
                        <div class="citation-pill" onclick="toggleCitationDrawer(this, '${drawerId}')">
                            <span class="citation-doc-icon">◧</span>
                            <span>${escapeHtml(c.doc)} · ${escapeHtml(c.page_or_chapter)}</span>
                            <span class="citation-arrow">▶</span>
                        </div>
                        <div id="${drawerId}" class="citation-drawer">
                            <div class="citation-meta-row">
                                <div class="citation-meta-item">DOCUMENT: <span>${escapeHtml(c.doc)}</span></div>
                                <div class="citation-meta-item">CHAPTER: <span>${escapeHtml(c.page_or_chapter)}</span></div>
                                <div class="citation-meta-item">CHUNK: <code>${escapeHtml(c.chunk_id)}</code></div>
                            </div>
                            <div class="quote-snippet">"${escapeHtml(c.quote_snippet)}"</div>
                        </div>
                    </div>
                `;
            });

            citHtml += `</div></div>`;
            aiHtml += citHtml;
        }

        // Telemetry Readout HUD
        if (telemetry) {
            const durMs = telemetry.duration_s ? Math.round(telemetry.duration_s * 1000) : (telemetry.ttft_ms || 0);
            const costUsd = telemetry.computed_cost_usd !== undefined ? `$${telemetry.computed_cost_usd.toFixed(6)}` : '$0.000000';
            const totalTokens = (telemetry.prompt_tokens || 0) + (telemetry.completion_tokens || 0);
            const modelName = telemetry.model_name || 'gemini-2.5-flash';

            aiHtml += `
                <div class="telemetry-hud">
                    <div class="hud-group">
                        <span class="hud-metric">LATENCY: <span class="hud-val accent">${durMs}ms</span></span>
                        <span class="hud-metric">COST: <span class="hud-val">${costUsd}</span></span>
                        <span class="hud-metric">TOKENS: <span class="hud-val">${totalTokens} (${telemetry.prompt_tokens || 0} in / ${telemetry.completion_tokens || 0} out)</span></span>
                    </div>
                    <div class="hud-group">
                        <span class="hud-metric">GATE: <span class="hud-val ${usedFallback ? 'accent' : ''}">${usedFallback ? 'REFUSED / FALLBACK' : 'PASSED (≥0.72)'}</span></span>
                        <span class="hud-metric">ENGINE: <span class="hud-val">${modelName}</span></span>
                    </div>
                </div>
            `;
        }

        textEl.innerHTML = aiHtml;
    }

    containerEl.appendChild(textEl);
    messageEl.appendChild(containerEl);
    chatBox.appendChild(messageEl);
}

// Toggle Citation Drawer
function toggleCitationDrawer(pillEl, drawerId) {
    const drawer = document.getElementById(drawerId);
    if (!drawer) return;

    const isVisible = drawer.classList.contains('visible');
    if (isVisible) {
        drawer.classList.remove('visible');
        pillEl.classList.remove('expanded');
    } else {
        drawer.classList.add('visible');
        pillEl.classList.add('expanded');
    }
}

// HTML Escaping
function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, function(m) {
        return {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        }[m];
    });
}

// Scroll to Bottom
function scrollToBottom() {
    chatScrollContainer.scrollTop = chatScrollContainer.scrollHeight;
}

// Auto-expand Textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 160) + 'px';
    if (this.value.trim() === '') {
        this.style.height = '48px';
    }
});

chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// Submit Chat Request
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (sendBtn.disabled) return;

    const question = chatInput.value.trim();
    if (!question) return;

    // Hide welcome hero on first message
    if (welcomeHero) welcomeHero.style.display = 'none';

    // Add user message to UI & state
    const userMsg = { 
        isUser: true, 
        text: question, 
        timestamp: Date.now() 
    };
    renderMessageToUI(userMsg);
    sessions[currentSessionId].messages.push(userMsg);

    // Auto-generate session title from first message
    if (sessions[currentSessionId].messages.length === 1) {
        let title = question.toUpperCase().substring(0, 26);
        if (question.length > 26) title += '...';
        sessions[currentSessionId].title = title;
        renderSidebar();
    }
    sessions[currentSessionId].timestamp = Date.now();
    saveState();

    // Reset input
    chatInput.value = '';
    chatInput.style.height = '48px';
    sendBtn.disabled = true;
    chatInput.disabled = true;

    // Show indicator
    typingIndicator.classList.remove('hidden');
    scrollToBottom();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: question, session_id: currentSessionId })
        });

        if (!response.ok) {
            throw new Error(`API HTTP Error: ${response.status}`);
        }

        const data = await response.json();
        typingIndicator.classList.add('hidden');

        const aiMsg = {
            isUser: false,
            text: data.answer,
            citations: data.citations || [],
            telemetry: data.telemetry || null,
            usedFallback: data.used_fallback || false,
            timestamp: Date.now()
        };

        renderMessageToUI(aiMsg);
        sessions[currentSessionId].messages.push(aiMsg);
        sessions[currentSessionId].timestamp = Date.now();
        saveState();

    } catch (error) {
        console.error('Chat execution failed:', error);
        typingIndicator.classList.add('hidden');
        const errorMsg = {
            isUser: false,
            text: "SYSTEM ERROR: Failed to communicate with BMW M RAG engine. Ensure the FastAPI backend is running on http://127.0.0.1:8000.",
            citations: [],
            telemetry: null,
            usedFallback: true,
            timestamp: Date.now()
        };
        renderMessageToUI(errorMsg);
    } finally {
        sendBtn.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
        scrollToBottom();
    }
});

// Quick Preset and Prompt Card Click Handlers
function setupQuickPrompts() {
    document.querySelectorAll('[data-query]').forEach(el => {
        el.addEventListener('click', () => {
            const query = el.getAttribute('data-query');
            if (query) {
                chatInput.value = query;
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
                chatInput.focus();
                chatForm.dispatchEvent(new Event('submit'));
                if (window.innerWidth <= 768) sidebar.classList.remove('open');
            }
        });
    });
}
setupQuickPrompts();

// New Chat Button Listeners
if (newChatBtn) newChatBtn.addEventListener('click', createNewSession);
if (topNewChatBtn) topNewChatBtn.addEventListener('click', createNewSession);

// Mobile Sidebar Toggle
if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });
}
