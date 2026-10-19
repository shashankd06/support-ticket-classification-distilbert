"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const input = $("query");
  const form = $("query-form");
  const panel = document.querySelector(".result-panel");
  const submit = $("classify-button");
  const spinner = document.querySelector(".button-spinner");
  const examples = [...document.querySelectorAll(".example")];
  const state = { busy: false, submitted: null, hasResult: false };
  const modelNames = { distilbert: "DistilBERT", baseline: "TF-IDF + logistic regression" };

  function setAPI(ready) {
    const badge = $("api-status");
    badge.classList.toggle("ready", ready);
    badge.classList.toggle("offline", !ready);
    $("api-status-text").textContent = ready ? "API ready" : "API unavailable";
  }

  function onEdit() {
    const length = input.value.length;
    $("character-count").textContent = `${length.toLocaleString("en-US")} / 4,000`;
    $("character-count").classList.toggle("near-limit", length > 3600);
    input.removeAttribute("aria-invalid");
    $("query-error").hidden = true;
    examples.forEach((button) => button.setAttribute("aria-pressed", String(input.value === button.dataset.example)));
    const outdated = state.hasResult && input.value !== state.submitted;
    panel.classList.toggle("outdated", outdated);
    if (state.hasResult) $("result-state").textContent = outdated ? "Previous result · input changed" : "Prediction";
    if (state.busy) $("request-status").textContent = input.value !== state.submitted
      ? "Classifying the submitted query. Your edits will need a new request."
      : "Classifying your query…";
  }

  function showError(message, invalid = false) {
    $("query-error").textContent = message;
    $("query-error").hidden = false;
    if (invalid) { input.setAttribute("aria-invalid", "true"); input.focus(); }
  }

  function showPrediction(result, submitted) {
    const readable = result.label.replace(/_/g, " ").replace(/\?$/, "");
    $("intent-label").textContent = readable.charAt(0).toUpperCase() + readable.slice(1);
    $("canonical-label").textContent = result.label;
    $("model-score").textContent = `${(result.score * 100).toFixed(1)}%`;
    $("model-name").textContent = modelNames[result.model] || result.model;
    $("submitted-query").textContent = submitted;
    const smoke = result.mode === "smoke";
    $("mode-notice").classList.toggle("smoke", smoke);
    $("mode-notice").textContent = smoke
      ? "Smoke-test checkpoint · functionality check, not the fully trained model"
      : result.mode === "full" ? "Full-training checkpoint" : `Checkpoint mode: ${result.mode}`;
    $("truncation-notice").hidden = !result.truncated;
    $("result-empty").hidden = true;
    $("result-content").hidden = false;
    state.hasResult = true;
    onEdit();
  }

  function validResponse(result) {
    return result && typeof result.label === "string" && result.label.length > 0
      && typeof result.score === "number" && Number.isFinite(result.score)
      && result.score >= 0 && result.score <= 1 && typeof result.model === "string"
      && typeof result.mode === "string" && typeof result.truncated === "boolean";
  }

  async function classify(event) {
    event.preventDefault();
    if (state.busy) return;
    const submitted = input.value;
    if (!submitted.trim()) { showError("Write a banking query or choose an example first.", true); return; }
    if (submitted.length > 4000) { showError("Keep your query within 4,000 characters.", true); return; }
    state.busy = true;
    state.submitted = submitted;
    state.hasResult = false;
    $("result-content").hidden = true;
    $("result-empty").hidden = false;
    panel.classList.remove("outdated");
    panel.setAttribute("aria-busy", "true");
    submit.disabled = true;
    spinner.hidden = false;
    $("button-label").textContent = "Classifying…";
    $("query-error").hidden = true;
    $("request-status").textContent = "Classifying your query…";
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 45000);
    try {
      const response = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: submitted }), signal: controller.signal });
      if (!response.ok) {
        if (response.status === 422) { showError("Check the query: use 1–4,000 characters with some text.", true); return; }
        throw new Error("API unavailable");
      }
      const result = await response.json();
      if (!validResponse(result)) throw new Error("Unexpected API response");
      setAPI(true);
      showPrediction(result, submitted);
      $("request-status").textContent = input.value === submitted
        ? "Classification complete."
        : "Classification complete for the submitted query. Your input has changed.";
    } catch (error) {
      setAPI(false);
      showError(error.name === "AbortError"
        ? "The classifier is taking longer than expected. Your query is still here; try again shortly."
        : "The classifier isn’t available right now. Your query is still here; try again shortly.");
      $("request-status").textContent = "No prediction was returned.";
    } finally {
      clearTimeout(timer);
      state.busy = false;
      submit.disabled = false;
      spinner.hidden = true;
      $("button-label").textContent = "Classify query";
      panel.setAttribute("aria-busy", "false");
      if (!state.hasResult && !$("query-error").hidden) $("request-status").textContent = "No prediction was returned.";
    }
  }

  async function loadEvaluation() {
    try {
      const response = await fetch("/evaluation");
      if (!response.ok) throw new Error("Evaluation unavailable");
      const report = await response.json();
      if (report.mode !== "full" || report.split !== "final_test" || !report.baseline || !report.distilbert) throw new Error("Invalid report");
      for (const name of ["baseline", "distilbert"]) {
        const metrics = report[name];
        if (!Number.isFinite(metrics.accuracy) || !Number.isFinite(metrics.macro_f1)) throw new Error("Invalid metrics");
        $(`${name}-accuracy`).textContent = `${(metrics.accuracy * 100).toFixed(2)}%`;
        $(`${name}-f1`).textContent = metrics.macro_f1.toFixed(4);
      }
      $("evaluation-context").textContent = `${report.distilbert.n.toLocaleString("en-US")} filtered held-out queries · 77 intents · one seeded experiment`;
      const delta = report.distilbert_macro_f1_delta;
      if (!Number.isFinite(delta)) throw new Error("Invalid delta");
      $("f1-delta").textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(4)} macro-F1 on this run`;
      $("evaluation-status").hidden = true;
    } catch (_) {
      $("evaluation-status").textContent = "Evaluation report unavailable. Open the experiment report below for the published evidence.";
    }
  }

  input.addEventListener("input", onEdit);
  examples.forEach((button) => button.addEventListener("click", () => { input.value = button.dataset.example; onEdit(); input.focus(); }));
  form.addEventListener("submit", classify);
  input.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); form.requestSubmit(); } });
  onEdit();
  fetch("/health", { signal: AbortSignal.timeout(8000) }).then((response) => setAPI(response.ok)).catch(() => setAPI(false));
  loadEvaluation();
})();
