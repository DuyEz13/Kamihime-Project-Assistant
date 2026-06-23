const updateButtons = Array.from(document.querySelectorAll(".update-button"));
const translateButton = document.getElementById("translate-database");
const translateForm = document.getElementById("translate-form");
const translationProvider = document.getElementById("translation-provider");
const actionButtons = [
  ...updateButtons,
  ...Array.from(document.querySelectorAll(".translate-button")),
];
const statusBox = document.getElementById("refresh-status");
const statusMessage = document.getElementById("refresh-message");
const progressPanel = document.getElementById("translation-progress");
const progressBar = document.getElementById("translation-progress-bar");
const progressTrack = progressPanel?.querySelector(".progress-track");
const progressPercent = document.getElementById("translation-percent");
const translationDevice = document.getElementById("translation-device");
const translationModel = document.getElementById("translation-model");
const crawlProgressPanel = document.getElementById("crawl-progress");
const crawlProgressList = document.getElementById("crawl-progress-list");
const crawlElements = (crawlProgressPanel?.dataset.elements || "")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatThread = document.getElementById("chat-thread");
const chatModelProvider = document.getElementById("chat-model-provider");
const newChatSessionButton = document.getElementById("new-chat-session");
const sidebarNewChat = document.getElementById("sidebar-new-chat");
const chatHistoryList = document.getElementById("chat-history-list");
const chatHistoryEmpty = document.getElementById("chat-history-empty");
let polling = false;

function titleCase(value) {
  return String(value || "")
    .replace(/[-_]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function ensureCrawlProgressRow(element) {
  let row = crawlProgressList?.querySelector(`[data-element="${element}"]`);
  if (row || !crawlProgressList) return row;

  const label = titleCase(element);
  row = document.createElement("div");
  row.className = "crawl-progress-item";
  row.dataset.element = element;

  const meta = document.createElement("div");
  meta.className = "progress-meta";
  const name = document.createElement("span");
  name.className = "crawl-element-label";
  name.textContent = label;
  const count = document.createElement("span");
  count.className = "crawl-count";
  count.textContent = "0/0";
  meta.append(name, count);

  const track = document.createElement("div");
  track.className = "progress-track";
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", `${label} crawl progress`);
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", "100");
  track.setAttribute("aria-valuenow", "0");
  const fill = document.createElement("div");
  fill.className = "progress-fill";
  fill.dataset.progressFill = "";
  track.appendChild(fill);

  const current = document.createElement("div");
  current.className = "progress-model crawl-current";
  row.append(meta, track, current);
  crawlProgressList.appendChild(row);
  return row;
}

function renderCrawlProgress(progressByElement) {
  if (!crawlProgressPanel || !crawlProgressList) return;
  const source = progressByElement || {};
  const existing = Array.from(crawlProgressList.querySelectorAll("[data-element]"))
    .map((row) => row.dataset.element)
    .filter(Boolean);
  const keys = crawlElements.length
    ? crawlElements
    : Array.from(new Set([...existing, ...Object.keys(source)]));
  keys.forEach((element) => {
    const item = source[element] || {};
    const processed = Number(item.processed) || 0;
    const total = Number(item.total) || 0;
    const progress = total ? Math.max(0, Math.min(100, Number(item.progress) || 0)) : 0;
    const row = ensureCrawlProgressRow(element);
    if (!row) return;
    const count = row.querySelector(".crawl-count");
    count.textContent = `${processed}/${total}`;
    const track = row.querySelector(".progress-track");
    track.setAttribute("aria-valuenow", String(progress));
    const fill = row.querySelector("[data-progress-fill]");
    fill.style.width = `${progress}%`;
    const current = row.querySelector(".crawl-current");
    current.textContent = item.character || (total ? "" : "Waiting for character list...");
  });
}

function renderStatus(status) {
  if (!statusBox || !actionButtons.length) return false;
  statusBox.dataset.state = status.state;
  if (statusMessage) statusMessage.textContent = status.message;
  const progress = Math.max(0, Math.min(100, Number(status.progress) || 0));
  const translating = status.state === "translating"
    || (status.state === "starting" && status.mode === "translate");
  const crawling = ["starting", "updating"].includes(status.state);
  if (crawlProgressPanel) crawlProgressPanel.hidden = !crawling;
  if (crawling) renderCrawlProgress(status.crawl_progress || {});
  if (progressPanel) progressPanel.hidden = !translating;
  if (progressBar) progressBar.style.width = `${progress}%`;
  if (progressPercent) progressPercent.textContent = `${progress}%`;
  if (progressTrack) progressTrack.setAttribute("aria-valuenow", String(progress));
  if (translationDevice) {
    translationDevice.textContent = status.device || "Detecting device...";
  }
  if (translationModel) translationModel.textContent = status.model || "";
  const active = ["starting", "updating", "translating"].includes(status.state);
  actionButtons.forEach((button) => {
    button.disabled = active;
  });
  if (translationProvider) translationProvider.disabled = active;
  updateButtons.forEach((button) => {
    const original = button.dataset.originalText || button.textContent;
    button.dataset.originalText = original;
    button.textContent = active && button.dataset.updateMode === status.mode
      ? status.state === "translating" ? "Translating..." : "Updating..."
      : original;
  });
  if (translateButton) {
    const original = translateButton.dataset.originalText || translateButton.textContent;
    translateButton.dataset.originalText = original;
    translateButton.textContent = active && status.mode === "translate"
      ? "Translating..."
      : original;
  }
  return active;
}

async function pollStatus() {
  if (polling) return;
  polling = true;
  const timer = setInterval(async () => {
    try {
      const response = await fetch("/api/update/status");
      const status = await response.json();
      if (!renderStatus(status)) {
        clearInterval(timer);
        polling = false;
        if (status.state === "completed") window.location.reload();
      }
    } catch {
      clearInterval(timer);
      polling = false;
      if (statusMessage) statusMessage.textContent = "Could not read refresh status.";
      actionButtons.forEach((button) => { button.disabled = false; });
      if (translationProvider) translationProvider.disabled = false;
    }
  }, 2000);
}

updateButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    actionButtons.forEach((item) => { item.disabled = true; });
    if (translationProvider) translationProvider.disabled = true;
    statusBox.dataset.state = "starting";
    if (statusMessage) statusMessage.textContent = "Starting update...";
    if (crawlProgressPanel) crawlProgressPanel.hidden = false;
    renderCrawlProgress({});
    try {
      const mode = button.dataset.updateMode;
      const response = await fetch(`/api/update/${mode}`, { method: "POST" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Update failed to start");
      renderStatus(body);
      pollStatus();
    } catch (error) {
      statusBox.dataset.state = "failed";
      if (statusMessage) statusMessage.textContent = error.message;
      actionButtons.forEach((item) => { item.disabled = false; });
      if (translationProvider) translationProvider.disabled = false;
    }
  });
});

translateForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!statusBox) return;
  actionButtons.forEach((item) => { item.disabled = true; });
  if (translationProvider) translationProvider.disabled = true;
  statusBox.dataset.state = "starting";
  if (statusMessage) statusMessage.textContent = "Starting translation...";
  if (crawlProgressPanel) crawlProgressPanel.hidden = true;
  if (progressPanel) progressPanel.hidden = false;
  try {
    const provider = translationProvider?.value || "deepl";
    const response = await fetch(`/api/translate/${provider}`, { method: "POST" });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Translation failed to start");
    renderStatus(body);
    pollStatus();
  } catch (error) {
    statusBox.dataset.state = "failed";
    if (statusMessage) statusMessage.textContent = error.message;
    actionButtons.forEach((item) => { item.disabled = false; });
    if (translationProvider) translationProvider.disabled = false;
  }
});

if (["starting", "updating", "translating"].includes(statusBox?.dataset.state)) {
  pollStatus();
}

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
}

sidebarToggle?.addEventListener("click", () => {
  document.body.classList.toggle("sidebar-open");
});

sidebarBackdrop?.addEventListener("click", closeSidebar);

document.querySelectorAll(".element-link").forEach((link) => {
  link.addEventListener("click", closeSidebar);
});

const CHAT_SESSION_KEY = "kamiwiki_chat_session_id";
const CHAT_PROVIDER_KEY = "kamiwiki_chat_provider";
const urlParams = new URLSearchParams(window.location.search);
let chatSessionId = urlParams.get("chat") || window.localStorage.getItem(CHAT_SESSION_KEY) || "";
if (urlParams.get("chat")) {
  window.localStorage.setItem(CHAT_SESSION_KEY, chatSessionId);
}

function removeChatWelcome() {
  chatThread?.querySelector(".chat-welcome")?.remove();
}

function restoreChatWelcome() {
  if (!chatThread || chatThread.querySelector(".chat-welcome")) return;
  chatThread.innerHTML = `
    <div class="chat-welcome">
      <span class="assistant-logo">K</span>
      <h1>How can I help with Kamihime?</h1>
      <p>Ask about characters, elements, skills, release data, or acquisition data.</p>
      <div class="prompt-suggestions">
        <button type="button" data-prompt="Show me notable Fire characters">Notable Fire characters</button>
        <button type="button" data-prompt="Compare Nike and Sol">Compare two characters</button>
        <button type="button" data-prompt="Which Water characters have healing skills?">Find healing skills</button>
      </div>
    </div>
  `;
  bindPromptButtons();
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
}

function renderMarkdown(text) {
  const lines = String(text || "").split(/\r?\n/);
  const blocks = [];
  let paragraph = [];
  let list = null;

  function flushParagraph() {
    if (!paragraph.length) return;
    blocks.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!list) return;
    const tag = list.type === "ol" ? "ol" : "ul";
    blocks.push(`<${tag}>${list.items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${tag}>`);
    list = null;
  }

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    if (ordered || unordered) {
      flushParagraph();
      const type = ordered ? "ol" : "ul";
      if (!list || list.type !== type) {
        flushList();
        list = { type, items: [] };
      }
      list.items.push((ordered || unordered)[1]);
      return;
    }

    flushList();
    paragraph.push(trimmed);
  });

  flushParagraph();
  flushList();
  return blocks.join("");
}

function appendChatMessage(role, text, options = {}) {
  if (!chatThread) return;
  removeChatWelcome();
  const row = document.createElement("div");
  row.className = `chat-message ${role}`;
  if (options.pending) row.dataset.pending = "1";
  if (options.error) row.classList.add("error");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  if (role === "assistant" && !options.pending && !options.error) {
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }
  if (options.sources?.length) {
    const sources = document.createElement("div");
    sources.className = "chat-sources";
    sources.textContent = `Sources: ${options.sources
      .slice(0, 5)
      .map((source) => source.name)
      .join(", ")}`;
    bubble.appendChild(sources);
  }
  row.appendChild(bubble);
  chatThread.appendChild(row);
  chatThread.scrollTop = chatThread.scrollHeight;
  return row;
}

function setActiveHistoryItem() {
  document.querySelectorAll("[data-chat-session]").forEach((row) => {
    row.classList.toggle("active", row.dataset.chatSession === chatSessionId);
  });
}

function clearChatThread() {
  if (!chatThread) return;
  chatThread.innerHTML = "";
  restoreChatWelcome();
}

function startNewChat(event) {
  window.localStorage.removeItem(CHAT_SESSION_KEY);
  chatSessionId = "";
  if (!chatThread) return;
  event?.preventDefault();
  clearChatThread();
  setActiveHistoryItem();
  chatInput?.focus();
  if (window.location.search) {
    window.history.replaceState({}, "", window.location.pathname);
  }
}

function renderChatHistory(sessions) {
  if (!chatHistoryList) return;
  chatHistoryList.innerHTML = "";
  if (chatHistoryEmpty) chatHistoryEmpty.hidden = sessions.length > 0;

  sessions.forEach((session) => {
    const row = document.createElement("div");
    row.className = "history-row";
    row.dataset.chatSession = session.session_id;

    const link = document.createElement("a");
    link.className = "history-link";
    link.href = `/?chat=${encodeURIComponent(session.session_id)}`;
    link.title = session.title;
    const icon = document.createElement("span");
    icon.className = "history-icon";
    icon.textContent = "◷";
    const title = document.createElement("span");
    title.textContent = session.title || "New chat";
    link.append(icon, title);

    link.addEventListener("click", (event) => {
      window.localStorage.setItem(CHAT_SESSION_KEY, session.session_id);
      chatSessionId = session.session_id;
      if (!chatThread) return;
      event.preventDefault();
      loadChatSession(session.session_id);
      setActiveHistoryItem();
      closeSidebar();
      window.history.replaceState({}, "", `/?chat=${encodeURIComponent(session.session_id)}`);
    });

    const deleteButton = document.createElement("button");
    deleteButton.className = "history-delete";
    deleteButton.type = "button";
    deleteButton.title = "Delete chat";
    deleteButton.setAttribute("aria-label", `Delete ${session.title || "chat"}`);
    deleteButton.textContent = "×";
    deleteButton.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const confirmed = window.confirm(`Delete chat "${session.title || "New chat"}"?`);
      if (!confirmed) return;
      try {
        const response = await fetch(`/api/chat/${session.session_id}`, { method: "DELETE" });
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Could not delete chat");
        if (chatSessionId === session.session_id) {
          startNewChat();
        }
        await loadChatHistory();
      } catch (error) {
        window.alert(error.message);
      }
    });

    row.append(link, deleteButton);
    chatHistoryList.appendChild(row);
  });
  setActiveHistoryItem();
}

async function loadChatHistory() {
  if (!chatHistoryList) return;
  try {
    const response = await fetch("/api/chat/sessions");
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Could not load chat history");
    renderChatHistory(body.sessions || []);
  } catch {
    chatHistoryList.innerHTML = '<p class="history-note">Could not load chat history.</p>';
    if (chatHistoryEmpty) chatHistoryEmpty.hidden = true;
  }
}

chatInput?.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 180)}px`;
});

function setChatBusy(busy) {
  if (chatInput) chatInput.disabled = busy;
  chatForm?.querySelector(".send-button")?.toggleAttribute("disabled", busy);
  if (chatModelProvider) chatModelProvider.disabled = busy;
}

async function loadChatModels() {
  if (!chatModelProvider) return;
  const preferred = window.localStorage.getItem(CHAT_PROVIDER_KEY) || chatModelProvider.value;
  try {
    const response = await fetch("/api/chat/models");
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Could not load chat models");
    chatModelProvider.innerHTML = "";
    body.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.provider;
      option.textContent = `${model.label} (${model.model})${model.configured ? "" : " - missing key"}`;
      option.disabled = !model.configured;
      chatModelProvider.appendChild(option);
    });
    const preferredOption = Array.from(chatModelProvider.options).find(
      (option) => option.value === preferred && !option.disabled
    );
    const firstEnabled = Array.from(chatModelProvider.options).find((option) => !option.disabled);
    chatModelProvider.value = preferredOption?.value || firstEnabled?.value || "gpt";
  } catch {
    chatModelProvider.value = preferred;
  }
}

async function loadChatSession(sessionId = chatSessionId) {
  if (!chatThread || !sessionId) return;
  try {
    const response = await fetch(`/api/chat/${sessionId}`);
    const body = await response.json();
    if (!response.ok || !body.messages?.length) return;
    chatThread.innerHTML = "";
    chatSessionId = sessionId;
    window.localStorage.setItem(CHAT_SESSION_KEY, chatSessionId);
    body.messages.forEach((message) => {
      appendChatMessage(message.role, message.content, { sources: message.sources });
    });
    setActiveHistoryItem();
  } catch {
    // Keep the blank welcome screen if stored history cannot be loaded.
  }
}

chatModelProvider?.addEventListener("change", () => {
  window.localStorage.setItem(CHAT_PROVIDER_KEY, chatModelProvider.value);
});

newChatSessionButton?.addEventListener("click", startNewChat);
sidebarNewChat?.addEventListener("click", startNewChat);

chatForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  appendChatMessage("user", message);
  chatInput.value = "";
  chatInput.style.height = "auto";
  const pending = appendChatMessage("assistant", "Thinking...", { pending: true });
  setChatBusy(true);
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: chatSessionId || null,
        provider: chatModelProvider?.value || "gpt",
        message,
      }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Chat request failed");
    chatSessionId = body.session_id;
    window.localStorage.setItem(CHAT_SESSION_KEY, chatSessionId);
    pending?.remove();
    appendChatMessage("assistant", body.answer, { sources: body.sources });
    await loadChatHistory();
  } catch (error) {
    pending?.remove();
    appendChatMessage("assistant", error.message, { error: true });
  } finally {
    setChatBusy(false);
    chatInput.focus();
  }
});

function bindPromptButtons() {
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    if (button.dataset.boundPrompt === "1") return;
    button.dataset.boundPrompt = "1";
    button.addEventListener("click", () => {
      if (!chatInput) return;
      chatInput.value = button.dataset.prompt || "";
      chatInput.focus();
      chatInput.dispatchEvent(new Event("input"));
    });
  });
}

bindPromptButtons();
loadChatModels();
loadChatSession();
loadChatHistory();
