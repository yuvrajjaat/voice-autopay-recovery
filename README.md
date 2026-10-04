# Voice Agent for Autopay Recovery

A voice agent that contacts customers whose automatic payment failed, explains
what happened, asks whether they want to retry, and triggers the retry — built
on **ElevenLabs Agents** with a **FastAPI** backend.

Everything about the money is fake and everything about the people is invented:

- **10 fictional customers** in `data/customers.json`. Names are made up, emails
  sit on the reserved `example.com` domain, phone numbers use the `555-0100`
  block reserved for fiction.
- **No real payments.** `app/payments/mock_processor.py` makes no network call
  and charges a stored method id — there is no parameter anywhere that accepts
  a card number.
- **No real messages.** A "payment link" is recorded with `delivered: false`
  and points at `example.test`, a TLD that can never resolve.
- **No telephony.** The demo runs through a browser microphone. Outbound calling
  is implemented only as a safety gate that refuses everything by default.

---

## Setup

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Run the tests — they all pass offline, with no accounts and no credentials:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### Try it with no accounts at all

The offline simulator drives the whole conversation against the real tool
endpoints, with a deterministic decision engine in place of the voice model:

```powershell
.\.venv\Scripts\python.exe scripts\simulate.py            # all 10 scenarios
.\.venv\Scripts\python.exe scripts\simulate.py -s CUST-003  # one transcript
.\.venv\Scripts\python.exe scripts\simulate.py --list
```

---

## The browser voice demo

### 1. Configure `.env`

| Variable | Needed for | How to get it |
|---|---|---|
| `ELEVENLABS_API_KEY` | everything | elevenlabs.io → Profile → API Keys |
| `TOOL_SHARED_SECRET` | tool calls | generate one, see below |
| `PUBLIC_BASE_URL` | tool calls | your tunnel's https address |
| `ELEVENLABS_AGENT_ID` | the voice page | written out by the provisioning script |
| `ELEVENLABS_WEBHOOK_SECRET` | post-call transcripts | optional; from the ElevenLabs webhook settings |

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

The free ElevenLabs plan includes roughly **15 agent minutes per month**, which
is enough for several demo conversations. Nothing in this project needs a paid
plan.

### 2. Start the server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

### 3. Start a tunnel

ElevenLabs' cloud calls the tool endpoints, so `localhost` has to be reachable.
Either of these works — run it in a second terminal:

```powershell
ngrok http 8000
# or
cloudflared tunnel --url http://localhost:8000
```

Copy the `https://…` address it prints into `.env` as `PUBLIC_BASE_URL`, with no
trailing path. An ngrok static domain is worth setting up if you have one: the
URL then survives restarts and you avoid re-provisioning.

### 4. Provision the agent

This pushes the prompt from `app/agent/prompt.py` and all seven tools from
`app/agent/tool_specs.py` into ElevenLabs. Nothing is configured by hand.

```powershell
.\.venv\Scripts\python.exe scripts\provision_agent.py --dry-run   # inspect first, no API call
.\.venv\Scripts\python.exe scripts\provision_agent.py             # create or update
.\.venv\Scripts\python.exe scripts\provision_agent.py --verify    # compare live agent with the repo
```

It prints an `ELEVENLABS_AGENT_ID`. Put that in `.env` and restart the server.
Re-run the script any time you change the prompt, a tool schema, or the tunnel
URL — it updates in place rather than creating duplicates.

### 5. Talk to it

Open **http://127.0.0.1:8000/voice**

1. Pick one of the ten fictional customers. The page shows what that caller
   wants, so you know what to play.
2. Press **Start conversation** — this creates the backend session and loads the
   agent.
3. Allow microphone access and press the widget's call button.
4. The agent will ask for the postal code on the account. **Read it from
   `data/customers.json`** — it is deliberately never sent to the browser,
   because it is the credential the agent checks.

Watch **http://127.0.0.1:8000/dashboard** in a second window while you talk: the
tool timeline and the ledger row update live as the agent calls each tool.

---

## How it fits together

```
  browser microphone                       (the demo path)
         |
  ElevenLabs Agents cloud
    speech-to-text -> Claude -> speech
    + 7 webhook tools
         |
   HTTPS via your tunnel, X-Tool-Secret header
         |
  FastAPI  POST /tools/*          the agent's seven tools
           GET  /api/*            dashboard + voice control plane
           POST /webhooks/*       post-call metadata
         |
  data/customers.json   (10 fictional records, read-only seed)
  data/runtime.json     (sessions, attempts, events; gitignored)
         |
  mock_processor.py     deterministic, offline, no network
```

### The seven tools

| Tool | What it does |
|---|---|
| `get_failed_payment_details` | amount, reason, whether a retry can work |
| `verify_identity` | checks the postal code; two attempts |
| `retry_payment` | the actual retry, through the mock processor |
| `schedule_retry` | records a future retry (a note, not a job) |
| `send_payment_link` | prepares a link; never sends anything |
| `escalate_to_human` | raises a ticket; contacts nobody |
| `log_disposition` | records the outcome and closes the session |

---

## Safety design

A prompt can be argued with, so the rules that matter are enforced in code:

- **Amounts are withheld until identity is verified.** `get_failed_payment_details`
  returns `verified: false` with every figure omitted if called early.
- **No charge without verification**, none on a settled balance, none past two
  attempts in a conversation, and none on a decline code where a retry cannot
  succeed.
- **`session_id` is injected by the platform, not chosen by the model.** It
  travels as an ElevenLabs dynamic variable, and the tool request schema has no
  customer field at all — so the agent cannot aim a tool call at a different
  account.
- **A recovery claim must be backed by an approved payment.** `log_disposition`
  refuses `payment_recovered` unless a retry actually returned `paid`.
- **The post-call webhook cannot change payment state.** It records duration and
  a transcript; the tool calls remain the only writers of the ledger.
- **Opt-out outlives the call.** `do_not_call` is stored against the customer.
- **Outbound calling is off by default.** `app/dial_safety.py` requires
  `ENABLE_OUTBOUND_CALLS=true`, a configured `DEMO_PHONE_NUMBER`, a destination
  equal to it, and a customer who has not opted out. The ten fictional phone
  numbers are never dialable.

No secret reaches the browser: the API key, the webhook secret, and the tool
secret are all server-side, and the voice page receives a short-lived signed URL
instead of a credential.

---

## Layout

```
app/
  main.py              FastAPI app
  config.py            typed settings, E.164 normalisation
  models.py            Pydantic models: seed, runtime, tool I/O
  store.py             JSON persistence, sessions, event log
  security.py          X-Tool-Secret dependency
  errors.py            one error envelope for every tool failure
  dial_safety.py       the outbound-call gate
  agent/
    prompt.py          the system prompt (source of truth)
    tool_specs.py      the seven tool specifications
  payments/
    mock_processor.py  deterministic fake authorizations
  providers/
    elevenlabs_client.py   all ElevenLabs-specific code
  routers/
    tools.py           the agent's seven tools
    demo.py            dashboard + voice control plane
    pages.py           /dashboard and /voice
    webhooks.py        post-call metadata
scripts/
  simulate.py          offline conversation simulator
  provision_agent.py   push prompt + tools to ElevenLabs
data/                  the 10 fictional records; runtime state
demo/transcripts/      simulator and voice transcripts
tests/                 the test suite
```

## Commands

```powershell
# tests
.\.venv\Scripts\python.exe -m pytest

# server
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# offline simulator
.\.venv\Scripts\python.exe scripts\simulate.py

# provision the ElevenLabs agent
.\.venv\Scripts\python.exe scripts\provision_agent.py
```

Dashboard: http://127.0.0.1:8000/dashboard ·
Voice demo: http://127.0.0.1:8000/voice ·
API docs: http://127.0.0.1:8000/docs
