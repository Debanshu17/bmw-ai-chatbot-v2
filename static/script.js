// --- BMW M Production RAG Frontend Controller ---

// --- DOM Elements ---
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('userInput');
const chatBox = document.getElementById('chatBox');
const typingIndicator = document.getElementById('typingIndicator');
const sendBtn = document.getElementById('sendBtn');
const mainChatArea = document.querySelector('.chat-container');
const historyList = document.getElementById('historyList');
const newChatBtn = document.getElementById('newChatBtn');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const sidebar = document.getElementById('sidebar');

// --- State Management ---
let sessions = JSON.parse(localStorage.getItem('bmw_rag_sessions')) || {};
let currentSessionId = localStorage.getItem('bmw_rag_current_session') || null;

// Initialize
if (!currentSessionId || !sessions[currentSessionId]) {
    createNewSession();
} else {
    renderSidebar();
    loadSession(currentSessionId);
}

// --- Functions ---
function generateId() {
    return 'bmw_session_' + Math.random().toString(36).substring(2, 12);
}

function createNewSession() {
    currentSessionId = generateId();
    sessions[currentSessionId] = {
        title: "NEW CHAT",
        timestamp: Date.now(),
        messages: []
    };
    saveState();
    renderSidebar();
    loadSession(currentSessionId);
}

function saveState() {
    localStorage.setItem('bmw_rag_sessions', JSON.stringify(sessions));
    localStorage.setItem('bmw_rag_current_session', currentSessionId);
}

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

function loadSession(id) {
    chatBox.innerHTML = '';
    const sessionMessages = sessions[id].messages;

    if (sessionMessages.length === 0) {
        const welcomeHtml = `
            <div class="message ai-message">
                <div class="message-content">
                    <p>BMW M MOTORSPORT ENGINEERING INTELLIGENCE ONLINE. HOW CAN I ASSIST YOUR TRACK SETUP TODAY?</p>
                </div>
            </div>`;
        chatBox.insertAdjacentHTML('beforeend', welcomeHtml);
    } else {
        sessionMessages.forEach(msg => {
            renderMessageToUI(msg);
        });
    }
    scrollToBottom();
}

function renderMessageToUI(msgObj) {
    const isUser = msgObj.isUser;
    const text = msgObj.text;
    const citations = msgObj.citations || [];
    const telemetry = msgObj.telemetry || null;
    const usedFallback = msgObj.usedFallback || false;

    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    msgDiv.classList.add(isUser ? 'user-message' : 'ai-message');

    const content = document.createElement('div');
    content.classList.add('message-content');

    if (isUser) {
        content.innerHTML = `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
    } else {
        let bodyHtml = '';

        if (usedFallback) {
            bodyHtml += `<div class="fallback-pill">⚠️ GUARDRAIL FALLBACK ACTIVATED</div>`;
        }

        if (window.marked) {
            bodyHtml += marked.parse(text);
        } else {
            bodyHtml += `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
        }

        // Render Citations
        if (citations && citations.length > 0) {
            let citHtml = `
                <div class="citations-container">
                    <div class="citations-title">VERIFIED PROVENANCE (${citations.length})</div>
                    <div class="citations-list">`;

            citations.forEach((c, idx) => {
                citHtml += `
                    <div>
                        <div class="citation-pill" onclick="toggleCitationDrawer(this, 'drawer_${idx}')">
                            <span>📄</span> ${escapeHtml(c.doc)} | ${escapeHtml(c.page_or_chapter)}
                        </div>
                        <div id="drawer_${idx}" class="citation-drawer">
                            <div><strong>Source:</strong> ${escapeHtml(c.doc)} (${escapeHtml(c.page_or_chapter)})</div>
                            <div><strong>Chunk ID:</strong> <code>${escapeHtml(c.chunk_id)}</code></div>
                            <div class="quote-text">"${escapeHtml(c.quote_snippet)}"</div>
                        </div>
                    </div>`;
            });

            citHtml += `</div></div>`;
            bodyHtml += citHtml;
        }

        // Render Telemetry HUD
        if (telemetry) {
            const costStr = telemetry.computed_cost_usd !== undefined ? `$${telemetry.computed_cost_usd.toFixed(6)}` : '$0.000000';
            const durMs = telemetry.duration_s ? Math.round(telemetry.duration_s * 1000) : (telemetry.ttft_ms || 0);
            bodyHtml += `
                <div class="telemetry-hud">
                    <span class="hud-metric">LATENCY: <span class="hud-val">${durMs}ms</span></span>
                    <span class="hud-metric">COST: <span class="hud-val">${costStr}</span></span>
                    <span class="hud-metric">TOKENS: <span class="hud-val">${telemetry.prompt_tokens + telemetry.completion_tokens}</span></span>
                    <span class="hud-metric">MODEL: <span class="hud-val">${telemetry.model_name || 'gemini-2.5-flash'}</span></span>
                </div>`;
        }

        content.innerHTML = bodyHtml;
    }

    msgDiv.appendChild(content);
    chatBox.appendChild(msgDiv);
}

function toggleCitationDrawer(btn, drawerId) {
    const drawer = document.getElementById(drawerId);
    if (drawer) {
        drawer.classList.toggle('visible');
    }
}

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

function scrollToBottom() {
    mainChatArea.scrollTop = mainChatArea.scrollHeight;
}

// --- Event Listeners ---
newChatBtn.addEventListener('click', createNewSession);

mobileMenuBtn.addEventListener('click', () => {
    sidebar.classList.toggle('open');
});

chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value.trim() === '') {
        this.style.height = '60px';
    }
});

chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (sendBtn.disabled) return;

    const question = chatInput.value.trim();
    if (!question) return;

    // Add user message to UI & state
    const userMsg = { isUser: true, text: question };
    renderMessageToUI(userMsg);
    sessions[currentSessionId].messages.push(userMsg);

    // Auto-generate title
    if (sessions[currentSessionId].messages.length === 1) {
        let title = question.substring(0, 25);
        if (question.length > 25) title += '...';
        sessions[currentSessionId].title = title;
        renderSidebar();
    }
    sessions[currentSessionId].timestamp = Date.now();
    saveState();

    chatInput.value = '';
    chatInput.style.height = '60px';
    sendBtn.disabled = true;
    chatInput.disabled = true;

    typingIndicator.classList.remove('hidden');
    scrollToBottom();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: question, session_id: currentSessionId })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        const data = await response.json();
        typingIndicator.classList.add('hidden');

        const aiMsg = {
            isUser: false,
            text: data.answer,
            citations: data.citations || [],
            telemetry: data.telemetry || null,
            usedFallback: data.used_fallback || false
        };

        renderMessageToUI(aiMsg);
        sessions[currentSessionId].messages.push(aiMsg);
        sessions[currentSessionId].timestamp = Date.now();
        saveState();

    } catch (error) {
        console.error('Error in chat request:', error);
        typingIndicator.classList.add('hidden');
        const errorMsg = {
            isUser: false,
            text: "SYSTEM ERROR: FAILED TO CONNECT TO RAG ENGINE. ENSURE BACKEND IS ONLINE AT HTTP://127.0.0.1:8000.",
            citations: [],
            telemetry: null,
            usedFallback: true
        };
        renderMessageToUI(errorMsg);
    } finally {
        sendBtn.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
        scrollToBottom();
    }
});
