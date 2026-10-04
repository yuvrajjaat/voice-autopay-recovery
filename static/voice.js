/* Browser voice demo.
 *
 * Flow:
 *   pick a customer
 *     -> POST /api/sessions                      (backend session)
 *     -> GET  /api/voice/connection/{session_id}  (signed URL + variables)
 *     -> mount <elevenlabs-convai>               (microphone conversation)
 *
 * No secret reaches this file. The signed URL is minted server-side and is
 * good for one conversation; the API key, the webhook secret and the tool
 * secret all stay on the server. The page never calls /tools/* either — the
 * agent does that from ElevenLabs' cloud.
 */

"use strict";

let customers = [];
let sessionId = null;
let widget = null;

const $ = (id) => document.getElementById(id);

function setStatus(text, kind) {
  const pill = $("status-pill");
  pill.textContent = text;
  pill.className = `pill ${kind || "pill-neutral"}`;
}

function showBanner(message, kind) {
  const banner = $("banner");
  banner.textContent = message;
  banner.className = kind === "info" ? "banner info" : "banner";
  banner.hidden = false;
  if (kind === "info") setTimeout(() => { banner.hidden = true; }, 5000);
}

async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch (_) {
    body = { message: text };
  }
  if (!response.ok) {
    throw new Error((body && (body.message || body.detail)) || response.statusText);
  }
  return body;
}

/* --------------------------------------------------------------- customers */

async function loadCustomers() {
  try {
    customers = await api("/api/customers");
  } catch (error) {
    showBanner(`Could not load customers: ${error.message}`);
    return;
  }

  $("customer-count").textContent = `${customers.length} fictional records`;
  $("customer").innerHTML =
    '<option value="">Select a customer…</option>' +
    customers
      .map(
        (customer) =>
          `<option value="${customer.customer_id}">` +
          `${customer.customer_id} — ${customer.name} — ` +
          `${customer.amount} ${customer.currency} (${customer.scenario_label})` +
          "</option>"
      )
      .join("");
  $("customer").addEventListener("change", describeCustomer);
}

function describeCustomer() {
  const customer = customers.find((entry) => entry.customer_id === $("customer").value);
  const note = $("customer-note");
  if (!customer) {
    note.hidden = true;
    return;
  }
  note.hidden = false;
  note.innerHTML =
    `<strong>Play this caller:</strong> ${customer.caller_intent} ` +
    `<span class="muted">(expected: <code>${customer.expected_path}</code>)</span>`;
}

/* ----------------------------------------------------------------- session */

function renderSession(connection) {
  const rows = [
    ["Customer", `${connection.customer_name} <span class="muted mono">${connection.customer_id}</span>`],
    ["Session", `<span class="mono">${connection.session_id}</span>`],
    ["Agent", connection.agent_id ? `<span class="mono">${connection.agent_id}</span>` : "&mdash;"],
    ["Connection", `<span class="mono">${connection.mode}</span>`],
  ];
  $("session").innerHTML = rows
    .map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`)
    .join("");
}

async function start() {
  const customerId = $("customer").value;
  if (!customerId) {
    showBanner("Pick a customer first.");
    return;
  }

  $("btn-start").disabled = true;
  setStatus("creating session…", "pill-neutral");

  let connection;
  try {
    const session = await api("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_id: customerId, channel: "web" }),
    });
    sessionId = session.session_id;
    connection = await api(`/api/voice/connection/${sessionId}`);
  } catch (error) {
    setStatus("failed", "pill-bad");
    showBanner(`Could not start: ${error.message}`);
    $("btn-start").disabled = false;
    return;
  }

  renderSession(connection);
  $("btn-end").disabled = false;

  if (connection.mode === "unavailable") {
    setStatus("session ready, agent unavailable", "pill-warn");
    showBanner(connection.message || "ElevenLabs is not configured.");
    return;
  }

  if (connection.message) showBanner(connection.message, "info");
  mountAgent(connection);
  setStatus("agent loaded", "pill-ok");
}

function mountAgent(connection) {
  unmountAgent();

  widget = document.createElement("elevenlabs-convai");
  if (connection.mode === "signed_url" && connection.signed_url) {
    widget.setAttribute("signed-url", connection.signed_url);
  } else {
    widget.setAttribute("agent-id", connection.agent_id);
  }
  // Carries the backend session into the conversation, so every tool call the
  // agent makes is bound to this customer and cannot be aimed elsewhere.
  widget.setAttribute("dynamic-variables", JSON.stringify(connection.dynamic_variables));

  const host = $("agent-host");
  host.hidden = false;
  host.appendChild(widget);
  $("mic-hint").hidden = false;
}

function unmountAgent() {
  if (widget && widget.parentNode) widget.parentNode.removeChild(widget);
  widget = null;
  $("agent-host").hidden = true;
  $("mic-hint").hidden = true;
}

function end() {
  unmountAgent();
  $("btn-end").disabled = true;
  $("btn-start").disabled = false;
  setStatus("ended", "pill-neutral");
  if (sessionId) {
    showBanner(
      "Conversation ended. Open the dashboard to see the session's tool timeline.",
      "info"
    );
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  $("btn-start").addEventListener("click", start);
  $("btn-end").addEventListener("click", end);
  await loadCustomers();
  setStatus("idle", "pill-neutral");
});
