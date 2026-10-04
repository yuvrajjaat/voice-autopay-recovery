/* Demo dashboard behaviour.
 *
 * Plain fetch + DOM. The page talks only to /api/*; it never calls /tools/*,
 * so the tool secret stays out of the browser entirely.
 *
 * The session id is kept in localStorage so a reload (or a mid-recording
 * refresh) does not lose the session you were demonstrating.
 */

"use strict";

const STORAGE_KEY = "autopay-demo-session";

let selectedCustomer = null;
let sessionId = null;
let customers = [];
let timer = null;

const $ = (id) => document.getElementById(id);

/* ----------------------------------------------------------------- utils */

function showBanner(message, kind) {
  const banner = $("banner");
  banner.textContent = message;
  banner.className = kind === "info" ? "banner info" : "banner";
  banner.hidden = false;
  if (kind === "info") {
    setTimeout(() => { banner.hidden = true; }, 4000);
  }
}

function clearBanner() {
  $("banner").hidden = true;
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
    const detail = (body && (body.message || body.detail)) || response.statusText;
    throw new Error(`${response.status} — ${detail}`);
  }
  return body;
}

function timeOf(isoString) {
  try {
    return new Date(isoString).toLocaleTimeString();
  } catch (_) {
    return isoString;
  }
}

function statusPill(value) {
  const ok = ["recovered", "active", "paid"];
  const warn = ["scheduled", "link_sent", "escalated"];
  const bad = ["do_not_call", "closed"];
  let cls = "pill-neutral";
  if (ok.includes(value)) cls = "pill-ok";
  else if (warn.includes(value)) cls = "pill-warn";
  else if (bad.includes(value)) cls = "pill-bad";
  return `<span class="pill ${cls}">${value}</span>`;
}

function yesNo(value, trueIsGood) {
  const good = trueIsGood ? value : !value;
  return `<span class="pill ${good ? "pill-ok" : "pill-neutral"}">${value ? "yes" : "no"}</span>`;
}

/* ------------------------------------------------------------- customers */

async function loadCustomers() {
  try {
    customers = await api("/api/customers");
  } catch (error) {
    showBanner(`Could not load customers: ${error.message}`);
    return;
  }

  $("customer-count").textContent = `${customers.length} fictional records`;
  $("customer-rows").innerHTML = customers.map((customer) => `
    <tr data-id="${customer.customer_id}">
      <td><input type="radio" name="pick" value="${customer.customer_id}"></td>
      <td class="mono">${customer.customer_id}</td>
      <td>${customer.name}</td>
      <td class="num">${customer.amount} ${customer.currency}</td>
      <td class="mono">${customer.failure_reason}</td>
      <td>${customer.scenario_label}</td>
      <td>${statusPill(customer.payment_status)}${customer.do_not_call ? ' <span class="pill pill-bad">do-not-call</span>' : ""}</td>
    </tr>
  `).join("");

  for (const row of document.querySelectorAll("#customer-rows tr")) {
    row.addEventListener("click", () => selectCustomer(row.dataset.id));
  }
  if (selectedCustomer) selectCustomer(selectedCustomer);
}

function selectCustomer(customerId) {
  selectedCustomer = customerId;
  for (const row of document.querySelectorAll("#customer-rows tr")) {
    const isPicked = row.dataset.id === customerId;
    row.classList.toggle("selected", isPicked);
    const radio = row.querySelector("input");
    if (radio) radio.checked = isPicked;
  }
  const customer = customers.find((entry) => entry.customer_id === customerId);
  if (customer) {
    $("caller-intent").hidden = false;
    $("caller-intent").innerHTML =
      `<strong>${customer.name}</strong> — ${customer.caller_intent} ` +
      `<span class="muted">(expected path: <code>${customer.expected_path}</code>)</span>`;
  }
  $("btn-create").disabled = false;
}

/* --------------------------------------------------------------- session */

async function createSession() {
  if (!selectedCustomer) return;
  clearBanner();
  try {
    const session = await api("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_id: selectedCustomer, channel: "web" }),
    });
    setSession(session.session_id);
    showBanner(`Session open for ${session.customer_name}.`, "info");
    await refreshAll();
  } catch (error) {
    showBanner(`Could not create session: ${error.message}`);
  }
}

function setSession(id) {
  sessionId = id;
  try {
    if (id) localStorage.setItem(STORAGE_KEY, id);
    else localStorage.removeItem(STORAGE_KEY);
  } catch (_) {
    /* private browsing or blocked storage: the dashboard still works */
  }
  const hasSession = Boolean(id);
  $("session-id-box").hidden = !hasSession;
  $("session-id").textContent = id || "";
  $("btn-refresh-state").disabled = !hasSession;
  $("btn-refresh-events").disabled = !hasSession;
  $("btn-reset").disabled = !hasSession;
  renderCurl();
}

async function refreshState() {
  if (!sessionId) return;
  let state;
  try {
    state = await api(`/api/sessions/${sessionId}`);
  } catch (error) {
    if (String(error.message).startsWith("404")) {
      showBanner("That session no longer exists — create a new one.");
      setSession(null);
      $("state").innerHTML = '<div class="kv-empty muted">No session.</div>';
      return;
    }
    showBanner(`Could not load state: ${error.message}`);
    return;
  }

  const rows = [
    ["Customer", `${state.customer_name} <span class="muted mono">${state.customer_id}</span>`],
    ["Session status", statusPill(state.status)],
    ["Channel", `<span class="mono">${state.channel}</span>`],
    ["Amount due", `<strong>${state.amount_due} ${state.currency}</strong>`],
    ["Payment status", statusPill(state.payment_status)],
    ["Identity verified", yesNo(state.identity_verified, true)],
    ["Verification attempts", `${state.verification_attempts} of ${state.verification_attempts_allowed}`],
    ["Retry attempts", String(state.retry_attempts)],
    ["Confirmation", state.last_confirmation_number ? `<span class="mono">${state.last_confirmation_number}</span>` : "&mdash;"],
    ["Scheduled retry", state.scheduled_for || "&mdash;"],
    ["Payment link", yesNo(state.payment_link_prepared, false)],
    ["Escalated", state.escalated ? `<span class="pill pill-warn">${state.escalation_ticket}</span>` : yesNo(false, false)],
    ["Disposition", state.disposition ? `<span class="mono">${state.disposition}</span>` : "&mdash;"],
    ["Notes", state.disposition_notes || "&mdash;"],
    ["Do not call", yesNo(state.do_not_call, false)],
    ["Tool calls", String(state.tool_calls)],
  ];

  $("state").innerHTML = rows
    .map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`)
    .join("");
}

async function refreshEvents() {
  if (!sessionId) return;
  let events;
  try {
    events = await api(`/api/sessions/${sessionId}/events`);
  } catch (error) {
    return; /* refreshState already surfaced a missing session */
  }

  $("event-count").textContent = events.length
    ? `${events.length} event${events.length === 1 ? "" : "s"}`
    : "";

  if (!events.length) {
    $("events").innerHTML = '<li class="muted">No events yet.</li>';
    return;
  }

  const bad = ["payment_retry_declined", "identity_verification_failed",
               "identity_verification_locked", "payment_retry_skipped"];
  const ok = ["identity_verified", "payment_retry_succeeded"];
  const warn = ["payment_scheduled", "payment_link_prepared",
                "human_escalation_created", "payment_details_withheld"];

  $("events").innerHTML = events.map((event) => {
    let cls = "";
    if (bad.includes(event.event_type)) cls = "ev-bad";
    else if (ok.includes(event.event_type)) cls = "ev-ok";
    else if (warn.includes(event.event_type)) cls = "ev-warn";

    const detail = Object.entries(event.detail || {})
      .map(([key, value]) => `${key}=${value}`)
      .join("  ");

    return `<li class="${cls}">
      <div class="ev-head">
        <span class="ev-seq">${event.sequence}</span>
        <span class="ev-type">${event.event_type}</span>
        <span class="ev-time">${timeOf(event.created_at)}</span>
      </div>
      <div class="ev-summary">${event.summary}</div>
      ${detail ? `<div class="ev-detail">${detail}</div>` : ""}
    </li>`;
  }).join("");
}

async function resetSession() {
  if (!sessionId) return;
  try {
    const result = await api(`/api/sessions/${sessionId}/reset`, { method: "POST" });
    showBanner(result.message, "info");
    await refreshAll();
    await loadCustomers();
  } catch (error) {
    showBanner(`Could not reset: ${error.message}`);
  }
}

async function refreshAll() {
  await refreshState();
  await refreshEvents();
}

/* ------------------------------------------------------- the curl helper */

function renderCurl() {
  if (!sessionId) {
    $("curl").textContent = "Create a session to see the commands.";
    return;
  }
  const tool = (name, body) =>
    `curl -s -X POST http://127.0.0.1:8000/tools/${name} \\\n` +
    `  -H "X-Tool-Secret: $env:TOOL_SHARED_SECRET" \\\n` +
    `  -H "Content-Type: application/json" \\\n` +
    `  -d '${JSON.stringify({ session_id: sessionId, ...body })}'`;

  $("curl").textContent = [
    "# 1. the agent looks up the failure (figures withheld until verified)",
    tool("get_failed_payment_details", {}),
    "",
    "# 2. confirm the postal code (read it from data/customers.json)",
    tool("verify_identity", { postal_code: "XXXXX" }),
    "",
    "# 3. retry the charge through the mock processor",
    tool("retry_payment", { payment_method: "primary" }),
    "",
    "# 4. close the call",
    tool("log_disposition", { disposition: "payment_recovered" }),
  ].join("\n");
}

/* ----------------------------------------------------------------- setup */

function startAutoRefresh() {
  if (timer) clearInterval(timer);
  timer = setInterval(async () => {
    if (!$("auto-refresh").checked || !sessionId) return;
    await refreshAll();
    $("tick").textContent = `· updated ${new Date().toLocaleTimeString()}`;
  }, 2000);
}

window.addEventListener("DOMContentLoaded", async () => {
  $("btn-create").addEventListener("click", createSession);
  $("btn-refresh-state").addEventListener("click", refreshState);
  $("btn-refresh-events").addEventListener("click", refreshEvents);
  $("btn-reset").addEventListener("click", resetSession);
  $("btn-copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(sessionId || "");
      showBanner("Session id copied.", "info");
    } catch (_) {
      showBanner("Copy failed — select the id manually.");
    }
  });

  await loadCustomers();

  let stored = null;
  try {
    stored = localStorage.getItem(STORAGE_KEY);
  } catch (_) {
    stored = null;
  }
  if (stored) {
    setSession(stored);
    await refreshAll();
  } else {
    setSession(null);
  }

  startAutoRefresh();
});
