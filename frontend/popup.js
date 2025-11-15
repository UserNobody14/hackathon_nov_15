const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const formEl = document.querySelector("#tab-plan-form");
const promptEl = document.querySelector("#prompt");
const limitEl = document.querySelector("#limit");
const tempEl = document.querySelector("#temperature");
const submitButton = document.querySelector("#submit-button");
const speechButton = document.querySelector("#speech-button");

const BOOKMARK_LIMIT = 200;
const HISTORY_LIMIT = 20;
const HISTORY_LOOKBACK_MS = 7 * 24 * 60 * 60 * 1000;

const isHttpUrl = (value) => {
  if (!value || typeof value !== "string") {
    return false;
  }
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch (error) {
    return false;
  }
};

const setStatus = (message, isError = false) => {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#dc2626" : "";
};

const extractErrorMessage = (input) => {
  if (!input) {
    return null;
  }
  if (typeof input === "string") {
    return input;
  }
  if (Array.isArray(input)) {
    const parts = input
      .map((item) => extractErrorMessage(item))
      .filter((value) => typeof value === "string" && value.length > 0);
    if (parts.length > 0) {
      return parts.join("; ");
    }
    return null;
  }
  if (typeof input === "object") {
    if ("detail" in input && input.detail) {
      return extractErrorMessage(input.detail);
    }
    if ("message" in input && input.message) {
      return extractErrorMessage(input.message);
    }
    if ("msg" in input && input.msg) {
      return extractErrorMessage(input.msg);
    }
    try {
      return JSON.stringify(input);
    } catch (error) {
      return null;
    }
  }
  return null;
};

const toggleLoading = (isLoading) => {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Planning…" : "Plan Tabs";
};

const chromePromise = (fn) =>
  new Promise((resolve, reject) => {
    fn((value) => {
      const error = chrome.runtime.lastError;
      if (error) {
        reject(new Error(error.message));
        return;
      }
      resolve(value);
    });
  });

const collectBookmarks = async () => {
  const tree = await chromePromise((resolve) =>
    chrome.bookmarks.getTree(resolve)
  );

  const bookmarks = [];
  const stack = [...tree];

  while (stack.length > 0 && bookmarks.length < BOOKMARK_LIMIT) {
    const node = stack.pop();
    if (!node) {
      continue;
    }
    if (node.children && node.children.length) {
      stack.push(...node.children);
    }
    if (node.url && isHttpUrl(node.url)) {
      bookmarks.push({
        title: node.title || node.url,
        url: node.url,
        tags: null,
        description: null,
      });
    }
  }

  return bookmarks;
};

const collectHistory = async () => {
  const startTime = Date.now() - HISTORY_LOOKBACK_MS;
  const items = await chromePromise((resolve) =>
    chrome.history.search(
      {
        text: "",
        maxResults: HISTORY_LIMIT,
        startTime,
      },
      resolve
    )
  );

  return items
    .filter((entry) => isHttpUrl(entry.url))
    .map((entry) => ({
      title: entry.title || entry.url,
      url: entry.url,
      last_visited: entry.lastVisitTime
        ? new Date(entry.lastVisitTime).toISOString()
        : null,
    }));
};

const collectOpenTabs = async () => {
  const tabs = await chromePromise((resolve) => chrome.tabs.query({}, resolve));
  return tabs
    .filter((tab) => isHttpUrl(tab.url))
    .map((tab) => ({
      title: tab.title || tab.url,
      url: tab.url,
      opened_at: null,
      pinned: tab.pinned ?? null,
    }));
};

const renderResults = (suggestions) => {
  resultsEl.replaceChildren();

  if (!suggestions || suggestions.length === 0) {
    const placeholder = document.createElement("p");
    placeholder.textContent = "No suggestions yet.";
    resultsEl.appendChild(placeholder);
    return;
  }

  suggestions.forEach((item) => {
    const container = document.createElement("article");
    container.className = "suggestion";

    const heading = document.createElement("a");
    heading.href = item.url;
    heading.target = "_blank";
    heading.rel = "noopener noreferrer";
    heading.textContent = item.title;

    const reason = document.createElement("p");
    reason.textContent = item.reason;

    const score = document.createElement("p");
    score.className = "score";
    score.textContent = `Confidence: ${(item.score * 100).toFixed(0)}%`;

    container.append(heading, reason, score);
    resultsEl.appendChild(container);
    chrome.tabs.create({ url: item.url });
  });
};

const parseNumericInput = (inputEl, fallback, min, max) => {
  const value = Number.parseFloat(inputEl.value);
  if (Number.isNaN(value)) {
    return fallback;
  }
  return Math.min(Math.max(value, min), max);
};

const handleSubmit = async (event) => {
  event.preventDefault();

  const prompt = promptEl.value.trim();
  await sendToServer(prompt);
};

const sendToServer = async (prompt) => {
  if (!prompt) {
    setStatus("Please enter a prompt before planning tabs.", true);
    return;
  }

  setStatus("Collecting bookmarks, history, and open tabs…");
  toggleLoading(true);
  renderResults([]);

  try {
    const [bookmarks, history, openTabs] = await Promise.all([
      collectBookmarks(),
      collectHistory(),
      collectOpenTabs(),
    ]);

    if (bookmarks.length === 0) {
      setStatus(
        "No bookmarks found. Add at least one bookmark to use the planner.",
        true
      );
      return;
    }

    const limit = parseNumericInput(limitEl, 5, 1, 10);
    const temperature = parseNumericInput(tempEl, 0.2, 0, 2);

    setStatus("Requesting tab plan from server…");

    const response = await fetch(
      `https://hackathon.sobel.club/tabs?limit=${encodeURIComponent(
        limit
      )}&temperature=${encodeURIComponent(temperature)}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt,
          bookmarks,
          history,
          open_tabs: openTabs,
        }),
      }
    );

    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      const message =
        extractErrorMessage(detail) ??
        `Request failed with status ${response.status} ${response.statusText}`;
      throw new Error(message);
    }

    const payload = await response.json();
    renderResults(payload.tabs);
    setStatus("Tab plan ready. Review the suggestions below.");
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Something went wrong.";
    setStatus(message, true);
  } finally {
    toggleLoading(false);
  }
};

if (speechButton) {
  speechButton.addEventListener("click", () => {
    // chrome.tabs.create({
    //   url: chrome.runtime.getURL("mic.html"),
    // });
    speechButton.textContent = "🎤 Listening...";
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";
    recognition.start();
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      promptEl.value = transcript;
      speechButton.textContent = "🎤 Processing...";
      sendToServer(transcript);
    };
    recognition.onerror = (event) => {
      setStatus(`Speech recognition error: ${event.error}`, true);
    };
    recognition.onend = () => {
      speechButton.textContent = "🎤 Speak";
    };
  });
}

formEl.addEventListener("submit", handleSubmit);

renderResults([]);
