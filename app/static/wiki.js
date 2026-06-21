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

function appendChatMessage(role, text) {
  if (!chatThread) return;
  const row = document.createElement("div");
  row.className = `chat-message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  chatThread.appendChild(row);
  chatThread.scrollTop = chatThread.scrollHeight;
}

chatInput?.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 180)}px`;
});

chatForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  appendChatMessage("user", message);
  chatInput.value = "";
  chatInput.style.height = "auto";
  window.setTimeout(() => {
    appendChatMessage(
      "assistant",
      "The chat model is not connected yet. Use the element list on the left to browse character data."
    );
  }, 250);
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!chatInput) return;
    chatInput.value = button.dataset.prompt || "";
    chatInput.focus();
  });
});
