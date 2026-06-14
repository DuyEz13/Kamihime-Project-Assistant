const updateButtons = Array.from(document.querySelectorAll(".update-button"));
const statusBox = document.getElementById("refresh-status");
const statusMessage = document.getElementById("refresh-message");
const progressPanel = document.getElementById("translation-progress");
const progressBar = document.getElementById("translation-progress-bar");
const progressTrack = progressPanel?.querySelector(".progress-track");
const progressPercent = document.getElementById("translation-percent");
const translationDevice = document.getElementById("translation-device");
const translationModel = document.getElementById("translation-model");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatThread = document.getElementById("chat-thread");
let polling = false;

function renderStatus(status) {
  if (!statusBox || !updateButtons.length) return false;
  statusBox.dataset.state = status.state;
  if (statusMessage) statusMessage.textContent = status.message;
  const progress = Math.max(0, Math.min(100, Number(status.progress) || 0));
  const translating = status.state === "translating";
  if (progressPanel) progressPanel.hidden = !translating;
  if (progressBar) progressBar.style.width = `${progress}%`;
  if (progressPercent) progressPercent.textContent = `${progress}%`;
  if (progressTrack) progressTrack.setAttribute("aria-valuenow", String(progress));
  if (translationDevice) {
    translationDevice.textContent = status.device || "Detecting device...";
  }
  if (translationModel) translationModel.textContent = status.model || "";
  const active = ["starting", "updating", "translating"].includes(status.state);
  updateButtons.forEach((button) => {
    button.disabled = active;
    const original = button.dataset.originalText || button.textContent;
    button.dataset.originalText = original;
    button.textContent = active && button.dataset.updateMode === status.mode
      ? status.state === "translating" ? "Translating..." : "Updating..."
      : original;
  });
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
      updateButtons.forEach((button) => { button.disabled = false; });
    }
  }, 2000);
}

updateButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    updateButtons.forEach((item) => { item.disabled = true; });
    if (statusMessage) statusMessage.textContent = "Starting update...";
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
      updateButtons.forEach((item) => { item.disabled = false; });
    }
  });
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
