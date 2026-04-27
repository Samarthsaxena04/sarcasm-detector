const API = "";  // same origin since FastAPI serves the frontend

/**
 * Parse conversation text into structured messages.
 * Expected format: "Name: Message" per line
 */
function parseConversation(text) {
    return text
        .split("\n")
        .map(line => line.trim())
        .filter(line => line.includes(":"))
        .map(line => {
            const colonIndex = line.indexOf(":");
            return {
                sender: line.substring(0, colonIndex).trim(),
                text: line.substring(colonIndex + 1).trim()
            };
        })
        .filter(msg => msg.sender && msg.text);
}

/**
 * Main analysis function — called on button click.
 */
async function analyzeConversation() {
    const input = document.getElementById("conversation-input").value.trim();
    if (!input) return;

    const messages = parseConversation(input);
    if (messages.length === 0) return;

    const btn = document.getElementById("analyze-btn");
    btn.classList.add("loading");
    btn.disabled = true;

    try {
        const res = await fetch(`${API}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ messages })
        });

        const data = await res.json();
        renderMessages(data.results);
    } catch (err) {
        console.error(err);
        alert("Failed to connect to the API. Make sure the server is running.");
    } finally {
        btn.classList.remove("loading");
        btn.disabled = false;
    }
}



/**
 * Render individual message results as chat bubbles.
 */
function renderMessages(results) {
    const container = document.getElementById("chat-container");
    container.innerHTML = "";

    results.forEach((msg, i) => {
        const pct = Math.round(msg.score * 100);

        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${msg.label}`;
        bubble.style.animationDelay = `${i * 0.1}s`;

        bubble.innerHTML = `
            <div class="sender">${escapeHtml(msg.sender)}</div>
            <div class="message">${escapeHtml(msg.text)}</div>
            <div class="meta">
                <span class="tag">${msg.label}</span>
                <div class="confidence-bar-track">
                    <div class="confidence-bar-fill" id="bar-${i}"></div>
                </div>
                <span class="confidence-label">${pct}%</span>
            </div>
        `;

        container.appendChild(bubble);

        // Animate confidence bar
        setTimeout(() => {
            const bar = document.getElementById(`bar-${i}`);
            if (bar) bar.style.width = `${pct}%`;
        }, 100 + i * 100);
    });
}

/**
 * Escape HTML to prevent XSS.
 */
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Clear the conversation input and results.
 */
function clearConversation() {
    document.getElementById("conversation-input").value = "";
    document.getElementById("chat-container").innerHTML = "";
}

// Allow Ctrl+Enter to analyze
document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
        analyzeConversation();
    }
});
