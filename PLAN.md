# Voice Agent for Autopay Recovery — Implementation Plan (v2: Python / FastAPI / ElevenLabs)

**Assignment:** build a voice agent that contacts customers with failed autopay payments, identifies the failure, explains it, asks whether they want to retry, and triggers a mock payment retry. 10 fictional customers. Working end-to-end demonstration. Only call a number we control.

**Stack decided by the user:** Python · FastAPI · ElevenLabs Agents · Twilio only where telephony is required · JSON local storage · fully mocked payment processor.

**Status:** plan only — no code written yet.

---

## 0. The three findings that shaped this design

1. **ElevenLabs free plan = $0/month, 15 agent minutes, 4 concurrent calls, website widget included**, then $0.08/min. Fifteen minutes is roughly 4–6 short conversations, so the plan **budgets those minutes explicitly** and does all prompt debugging in a text harness that consumes none of them.
2. **Server (webhook) tools are invoked from ElevenLabs' cloud, not from the browser or the phone.** The agent's tool calls arrive at our FastAPI backend identically whether the human is on a browser mic or a PSTN call. That means the **browser demo is a genuine end-to-end demonstration** — mic → ElevenLabs → our tools → mock processor → spoken result — and telephony is a pure transport swap, not a different architecture. This is what makes the $0 web-first path viable.
3. **Twilio credentials are entered in the ElevenLabs dashboard when importing a number, never in our code.** Our repo therefore holds *no* Twilio secret, and the phone path adds exactly one env var (a phone-number ID) plus the destination number.

---

## 1. Updated architecture

```
                            YOU (the only permitted human)
                                     |
            +------------------------+------------------------+
            |                                                 |
   [PATH A - default, $0]                        [PATH B - only if required]
   Browser mic via the embedded                  Real phone call
   ElevenLabs widget, served at                  Twilio number imported
   http://localhost:8000/voice                   into ElevenLabs
            |                                                 |
            |                                       +---------v----------+
            |                                       | Twilio PSTN leg    |
            |                                       | -> DEMO_PHONE_     |
            |                                       |    NUMBER only     |
            |                                       +---------+----------+
            |                                                 |
            +------------------------+------------------------+
                                     |
                      +--------------v---------------+
                      |   ElevenLabs Agents cloud    |
                      |   STT -> LLM -> TTS,         |
                      |   turn-taking, interruption   |
                      |   system prompt + 7 webhook  |
                      |   tools + dynamic variables  |
                      +--------------+---------------+
                                     |
                      HTTPS (ngrok static domain)
                      POST /tools/*            <- agent calls our tools
                      POST /webhooks/elevenlabs/post-call
                                     |
       +-----------------------------v------------------------------+
       |              FastAPI backend  (localhost:8000)             |
       |                                                            |
       |  /tools/*      7 webhook tools, X-Tool-Secret required     |
       |  /api/*        dashboard control plane                     |
       |  /webhooks/*   post-call transcript, HMAC-verified         |
       |  /            dashboard (ledger + live tool trace)         |
       |  /voice        widget page for PATH A                      |
       +------------------+-------------------------+---------------+
                          |                         |
            +-------------v---------+   +-----------v--------------+
            | data/customers.json   |   | app/payments/            |
            | 10 fictional records  |   |   mock_processor.py      |
            | data/runtime.json     |   | deterministic, offline,  |
            | attempts, dispositions|   | zero network calls       |
            +-----------------------+   +--------------------------+
```

### Design decisions worth defending in the writeup

1. **Web-first, phone-capable.** Path A is the default demo and costs nothing beyond free quota. Path B is implemented and tested but gated behind `ENABLE_OUTBOUND_CALLS=true`. Both exercise identical backend code, so the phone demo proves no new logic — only a new transport.

2. **Session-bound tool calls, not LLM-supplied customer IDs.** When a conversation starts we mint a `session_id` and pass it as an ElevenLabs **dynamic variable**. Every webhook tool receives `session_id` as a platform-injected parameter (`value_type: dynamic_variable`) rather than something the LLM fills in. The backend resolves the customer from the session. Consequence: the model *cannot* hallucinate or be talked into a different customer's ID, so cross-customer data disclosure is structurally impossible rather than prompt-dependent. The LLM fills only genuine conversational arguments (the ZIP the caller states, the retry date they request).

3. **Tool endpoints are the only write path.** The LLM never mutates the ledger directly. Every state change passes a Pydantic-validated handler that appends an immutable audit entry to `runtime.json`.

4. **Dial-safety interlock (three independent layers).**
   - All 10 records store fictional `+1-555-01xx` numbers (the reserved fictional block). The dialer never reads them.
   - Outbound dials only `DEMO_PHONE_NUMBER` from `.env`; any other destination raises before an API call is made.
   - `ENABLE_OUTBOUND_CALLS` defaults to `false`, so a fresh clone physically cannot place a call. Covered by `tests/test_dial_safety.py`.

5. **Deterministic mock payments.** Outcome comes from each record's `mock_retry_outcome`, so CUST-001 always approves and CUST-002 always declines. Essential when recording a demo you can't retake — and it means the pytest suite asserts exact outcomes.

6. **The agent is defined in code, not in a dashboard.** `scripts/provision_agent.py` creates the 7 tools and the agent from `app/agent/`, printing the resulting `agent_id`. The submission is reproducible from a clone; nothing lives only in a web console.

7. **Live visibility during the call.** Tool calls hit our backend in real time, so the dashboard streams the tool trace live (`get_failed_payment → verify_identity → retry_payment: succeeded`) and flips the ledger row `failed → recovered` mid-conversation. The full transcript arrives afterward via the post-call webhook. The terminal mic script (`scripts/talk.py`) additionally prints live transcripts via SDK callbacks.

---

## 2. Updated folder structure

```
voice-autopay-recovery/
├── README.md                     what it is, setup, demo script, safety notes, limits
├── PLAN.md                       this document
├── requirements.txt              pinned, see §3
├── .env.example                  every variable, all values placeholders
├── .gitignore                    .env, data/runtime.json, demo/recordings/, __pycache__
├── pytest.ini                    (or [tool.pytest] — minimal config)
│
├── app/
│   ├── __init__.py
│   ├── main.py                   FastAPI app, router mounting, static + templates
│   ├── config.py                 pydantic-settings Settings; fails loudly on missing keys
│   ├── models.py                 Pydantic request/response schemas for all 7 tools
│   ├── store.py                  JSON load/save, customer queries, audit append, sessions
│   ├── security.py               X-Tool-Secret dependency + ElevenLabs HMAC verification
│   ├── routers/
│   │   ├── tools.py              POST /tools/*      <- the agent's 7 webhook tools
│   │   ├── demo.py               GET/POST /api/*    <- dashboard + CLI control plane
│   │   ├── webhooks.py           POST /webhooks/elevenlabs/post-call
│   │   └── pages.py              GET /  dashboard,  GET /voice  widget page
│   ├── agent/
│   │   ├── prompt.py             system prompt + first-message templates
│   │   └── tool_specs.py         the 7 tool definitions — SINGLE source of truth,
│   │                             consumed by provisioning AND the offline simulator
│   ├── payments/
│   │   └── mock_processor.py     deterministic fake authorizations, no network
│   └── providers/
│       └── elevenlabs_client.py  provision tools/agent, outbound call, signed URL
│
├── data/
│   ├── customers.json            the 10 fictional records (seed, read-only)
│   └── runtime.json              mutable ledger: attempts, dispositions, calls (gitignored)
│
├── scripts/
│   ├── provision_agent.py        create/update the 7 tools + the agent; prints agent_id
│   ├── talk.py                   local mic conversation (PATH A, terminal variant)
│   ├── place_call.py             outbound phone call (PATH B), honors the interlock
│   ├── simulate.py               offline text conversation over the same prompt + tools
│   └── reset_state.py            restore runtime.json from seed between demo takes
│
├── templates/
│   ├── dashboard.html            ledger table, live tool trace, reset button
│   └── voice.html                ElevenLabs widget + the scenario picker
├── static/
│   ├── app.js                    polls /api/calls, renders the trace
│   └── styles.css
│
├── tests/
│   ├── test_tools.py             each tool endpoint: happy path, auth, validation
│   ├── test_flows.py             all 10 records end-to-end through the tool layer
│   └── test_dial_safety.py       interlock refuses unknown numbers and disabled outbound
│
└── demo/
    ├── README.md                 what was demoed, when, with the results table
    ├── transcripts/              10 simulated + the recorded voice conversations
    ├── tool-traces/              JSON tool-call trace per demo conversation
    └── screenshots/
```

---

## 3. Required Python packages

Python 3.11.9 is already installed. `requirements.txt`:

```
fastapi==0.115.*            # backend; Pydantic v2 validation comes with it
uvicorn[standard]==0.32.*   # ASGI server
pydantic-settings==2.*      # typed env loading, fail-fast on missing keys
python-dotenv==1.*          # .env loading for the scripts
httpx==0.27.*               # ElevenLabs REST calls + FastAPI TestClient transport
jinja2==3.*                 # dashboard + widget templates
elevenlabs==2.*             # official SDK: provisioning, outbound call, Conversation
pytest==8.*                 # the offline test suite
```

Two optional extras, each isolated so a reviewer can skip them:

```
pyaudio==0.2.14             # ONLY for scripts/talk.py (terminal mic demo).
                            # Install via: pip install "elevenlabs[pyaudio]"
                            # Not needed at all for the browser widget demo.
anthropic==0.40.*           # ONLY for scripts/simulate.py (offline text harness)
```

Tunnel (not a Python package): **ngrok**, using its free tier's **one free static domain** so the tool URLs stay stable and the agent doesn't need re-provisioning after every restart. `cloudflared tunnel --url` is the fallback if you'd rather not sign up, at the cost of a rotating URL.

Deliberately **not** included: SQLAlchemy, Alembic, Celery, Redis, Docker, any frontend framework. At 10 records they add setup friction and zero assignment credit.

---

## 4. ElevenLabs integration approach

### 4.1 Provisioning (code, not clicks)

`scripts/provision_agent.py` runs once and is idempotent:

1. For each of the 7 definitions in `app/agent/tool_specs.py`, `POST /v1/convai/tools` with a `tool_config` describing: URL (`{PUBLIC_BASE_URL}/tools/<name>`), method `POST`, the `X-Tool-Secret` header, and the body parameter schema. Collect the returned tool IDs.
2. `POST /v1/convai/agents/create` with `conversation_config.agent.prompt` containing our system prompt, the LLM choice, and `tool_ids: [...]` from step 1. (Tools attach to an agent via `prompt.tool_ids`; inline definitions also work but tool IDs are the cleaner, updatable path.)
3. Print the `agent_id` and write it to `.env` for the operator to confirm.
4. Re-runs `PATCH` the existing objects instead of duplicating them, keyed on name.

### 4.2 The 7 webhook tools

Each is one FastAPI endpoint. Every one receives `session_id` as a **platform-injected dynamic variable**, so the model cannot select the customer:

| Tool | LLM-supplied args | Returns to the agent |
|---|---|---|
| `get_failed_payment_details` | — | amount, currency, due_date, failed_at, plain-English decline reason, card brand + last4, attempts_so_far, backup_method_available |
| `verify_identity` | `zip_code` | `{verified, attempts_remaining}` — locks after 2 failures |
| `retry_payment` | `payment_method` (`primary`\|`backup`) | `{status, reason_code, reason_text, confirmation_number?, amount}` |
| `schedule_retry` | `requested_date` | `{scheduled_for, confirmation_number}` |
| `send_payment_link` | `channel` (`sms`\|`email`) | `{sent_to_masked, link_id}` — **logged, never transmitted** |
| `escalate_to_human` | `reason` | `{ticket_id, callback_window}` |
| `log_disposition` | `outcome`, `notes` | `{ok: true}` — always called before hangup |

Do-not-call is `log_disposition(outcome="do_not_call")` rather than an eighth tool.

Tool auth: a shared secret in the `X-Tool-Secret` header, stored as an ElevenLabs workspace secret and verified by a FastAPI dependency. Unauthenticated hits get 401 and are logged — worth showing in the demo, since the tunnel is publicly reachable.

### 4.3 Per-conversation context

Customer facts reach the agent as **dynamic variables** (`customer_name`, `amount_due`, `due_date`, `card_last4`, `decline_reason`, `session_id`), referenced in the prompt as `{{customer_name}}`. The opening line is overridden per conversation via `conversation_config_override.agent.first_message`, so the agent greets the right person without a tool round-trip. Amounts and card details are *restated* by the agent only after `verify_identity` succeeds — the prompt forbids disclosure before that, and `get_failed_payment_details` returns `{"verified": false}` and withholds the figures if called early. Belt and braces: the rule is enforced in code, not only in the prompt.

### 4.4 Path A — the browser voice demo (default)

`GET /voice` serves a Jinja2 page embedding the ElevenLabs widget (`<elevenlabs-convai agent-id="...">`) with a dropdown to pick which of the 10 customers you're playing. Selecting one calls `POST /api/sessions` to mint the `session_id` and seed the dynamic variables, then starts the widget. For a private agent, the backend hands the browser a short-lived signed URL rather than exposing the API key client-side.

`scripts/talk.py` is the terminal equivalent using the SDK's `Conversation` + `DefaultAudioInterface`, with `callback_agent_response` / `callback_user_transcript` printing the dialogue live — which records much better in a screen capture than a widget alone.

### 4.5 Post-call webhook

`POST /webhooks/elevenlabs/post-call` handles `post_call_transcription` (full transcript, analysis, metadata), with the `elevenlabs-signature` header verified via the SDK's `webhooks.construct_event(...)`. The handler writes the transcript into `demo/transcripts/` and attaches duration plus cost to the call record — that is how demo artifacts get generated rather than hand-transcribed.

### 4.6 Free-minute budget (15 min/month)

| Allocation | Minutes |
|---|---|
| Prompt tuning over voice (after the text harness is already correct) | ~6 |
| Three recorded web demos (happy path, decline path, opt-out) | ~5 |
| One recorded phone demo, if required | ~3 |
| Reserve | ~1 |

Overage is $0.08/min, so even running 2× over costs about $1.20. The text simulator in §6 is what keeps us inside this. Note for the README: the free tier carries **no commercial licence** — fine for a coursework demo, and worth stating rather than glossing over.

---

## 5. Twilio integration approach (Path B, only if required)

Kept deliberately thin, because ElevenLabs owns the hard part.

1. Twilio trial: **$15.15 credit, no card required.** Verify your own mobile as a **Verified Caller ID** (SMS verification), then buy one local US number from the trial credit.
2. **Twilio's Account SID and Auth Token are entered in the ElevenLabs dashboard** (Phone Numbers → Import from Twilio). They are **never** committed, never placed in `.env`, and never touched by our code. ElevenLabs returns an `agent_phone_number_id`; that ID — not a credential — is the only thing our repo stores.
3. Outbound is then one SDK call:

```python
client.conversational_ai.twilio.outbound_call(
    agent_id=settings.elevenlabs_agent_id,
    agent_phone_number_id=settings.elevenlabs_agent_phone_number_id,
    to_number=settings.demo_phone_number,          # the ONLY permitted destination
    call_recording_enabled=True,
    conversation_initiation_client_data={
        "dynamic_variables": {...},                 # customer context + session_id
        "conversation_config_override": {
            "agent": {"first_message": greeting}
        },
    },
)
```

4. `scripts/place_call.py` refuses to run unless `ENABLE_OUTBOUND_CALLS=true` **and** the destination equals `DEMO_PHONE_NUMBER`. It never accepts a number as a CLI argument — only a `--customer CUST-001` scenario ID. You cannot dial an arbitrary number through this repo even deliberately.
5. A happy accident worth noting in the README: a Twilio **trial project can only call verified caller IDs**, so the platform itself enforces the assignment's "only a number you control" rule as a fourth layer underneath our three.

Cost: the PSTN leg is a few cents per minute from trial credit, plus ElevenLabs agent minutes. Expected total for one recorded phone demo: **well under $1, drawn from free credit.**

---

## 6. How the voice agent talks to the FastAPI backend

Two transports, **one** implementation — this is the core of the design.

```
PATH A (browser mic)              PATH B (phone)
       |                                 |
  WebRTC/WS audio               Twilio PSTN audio
       |                                 |
       +----------------+----------------+
                        |
             ElevenLabs Agents cloud
             (decides when to call a tool)
                        |
         HTTPS POST, server-side, via ngrok
                        |
       +----------------v-----------------+
       | FastAPI  POST /tools/<tool-name> |
       | 1. verify X-Tool-Secret          |
       | 2. Pydantic-validate the body    |
       | 3. resolve customer from         |
       |    session_id (never from LLM)   |
       | 4. enforce the gate:             |
       |    identity verified? attempts   |
       |    remaining? already recovered? |
       | 5. call mock_processor if needed  |
       | 6. append audit entry            |
       | 7. return a compact JSON object  |
       |    the agent can read aloud      |
       +----------------------------------+
                        |
            JSON result -> LLM -> TTS -> spoken
```

Because tool calls originate in ElevenLabs' cloud rather than on the client, the backend is identical for web and phone. The offline simulator (`scripts/simulate.py`) is a **third** transport over the same endpoints: it drives a text conversation with the Claude API using the identical system prompt and the tool schemas from `app/agent/tool_specs.py`, dispatching to the same HTTP endpoints. So a reviewer with no ElevenLabs account can still watch all 10 scenarios execute, and prompt iteration costs zero voice minutes.

Tool results are written for *speech*, not for screens: `{"status": "succeeded", "amount": "49.00", "confirmation_number": "C-8841"}` — short, unambiguous, no nested structures for the model to mangle aloud. Errors return a `reason_text` the agent can say verbatim, so a failure never produces improvised wording about money.

---

## 7. Required environment variables

`.env.example` is committed with **placeholders only**; `.env` is gitignored. Nothing real — no key, number, or secret — enters the repository at any point.

```ini
# --- ElevenLabs (required) ---
ELEVENLABS_API_KEY=sk_your_key_here
ELEVENLABS_AGENT_ID=                      # written by scripts/provision_agent.py
ELEVENLABS_WEBHOOK_SECRET=wsec_placeholder # post-call webhook HMAC verification

# --- Our own tool authentication (required) ---
TOOL_SHARED_SECRET=generate_a_random_string  # sent as X-Tool-Secret

# --- Public tunnel (required for tool calls to reach localhost) ---
PUBLIC_BASE_URL=https://your-static-domain.ngrok-free.app

# --- Dial safety (required before any phone call can be placed) ---
ENABLE_OUTBOUND_CALLS=false               # default off; must be flipped by hand
DEMO_PHONE_NUMBER=+10000000000            # E.164; the ONLY number that can be dialled

# --- Telephony (Path B only; a Twilio ID, NOT a credential) ---
ELEVENLABS_AGENT_PHONE_NUMBER_ID=

# --- Optional: offline text simulator ---
ANTHROPIC_API_KEY=

# --- App ---
APP_PORT=8000
LOG_LEVEL=INFO
```

Note the absence of `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` — by design. Those live only in the ElevenLabs dashboard (§5.2), so the most sensitive credential in the whole system never comes near our code. `app/config.py` validates this set at startup and raises a clear error naming the missing variable rather than failing deep inside a request.

---

## 8. Exact build order

Each phase ends in something demonstrable, and the account-free phases come first so no free quota is spent on code that isn't finished.

| # | Phase | Work | Done when | Accounts | Quota |
|---|---|---|---|---|---|
| 0 | **Skeleton** | venv, `requirements.txt`, `.gitignore`, `.env.example`, `git init`, `app/config.py` | `uvicorn app.main:app` serves `/healthz` | none | 0 |
| 1 | **Data + money** | `data/customers.json` (all 10), `store.py`, `mock_processor.py`, `tests/test_flows.py` | pytest proves all 10 retry outcomes | none | 0 |
| 2 | **Tool layer** | `models.py`, `tool_specs.py`, `routers/tools.py`, `security.py`, `tests/test_tools.py` | all 7 endpoints green, 401 on bad secret | none | 0 |
| 3 | **Dashboard** | `routers/demo.py`, `pages.py`, `templates/`, `static/` | ledger renders; tool trace updates live during a curl-driven run | none | 0 |
| 4 | **Prompt + simulator** | `agent/prompt.py`, `scripts/simulate.py` | all 10 scenarios produce sane transcripts; guardrails hold under adversarial prodding | Anthropic (optional) | 0 EL |
| 5 | **Dial safety** | `tests/test_dial_safety.py`, interlock in `place_call.py` | tests prove a fresh clone cannot dial | none | 0 |
| 6 | **Go live (Path A)** | ElevenLabs signup, ngrok static domain, `provision_agent.py`, `/voice` page, `talk.py` | **first real voice conversation completes a retry end-to-end** | ElevenLabs free | ~6 min |
| 7 | **Post-call capture** | `routers/webhooks.py`, HMAC verify, transcripts into `demo/` | transcript + duration + cost land automatically | — | ~0 |
| 8 | **Record Path A** | happy path (CUST-001), decline path (CUST-002), opt-out (CUST-010), plus the card-number refusal | three screen recordings + committed traces | — | ~5 min |
| 9 | **Telephony (if required)** | Twilio trial, verify caller ID, buy + import number, `place_call.py`, one recorded call | a real phone rings and the flow completes | Twilio trial | ~3 min |
| 10 | **Submission** | `README.md`, `demo/README.md` with the results table, honest limitations section | a clone reproduces everything from the README | — | 0 |

**Phases 0–5 need no accounts and cost nothing.** That is deliberate: by the time any voice minute is spent, the conversation logic, the guardrails, and all 10 scenarios are already proven by pytest and the text harness. Debugging a prompt over live audio is slow, and at 15 free minutes it's also the scarcest resource in the project.

Realistic effort: **5–7 hours**, with phases 4 and 8 (prompt tuning and recording) taking the largest share.

---

## 9. Unchanged from v1

The **conversation flow** (disclose → verify → explain → consent → retry → wrap, with branches for defer, new card, dispute, human, and opt-out), the **prompt guardrails** (never state a full card number or request one, no amounts before verification, explicit per-turn consent before any charge, max 2 retries, immediate opt-out honouring, admit to being an AI when asked), and the **10 fictional customer records** (CUST-001 … CUST-010, covering 3 recoveries, 3 hard declines, 1 verification failure, 1 dispute, 1 deferral, 1 opt-out) all carry over verbatim from v1 §4 and §6. Full tables are in the git history of this file; they will be reproduced in `README.md` at build time.

One addition prompted by the ElevenLabs design: because `get_failed_payment_details` now withholds figures until `verify_identity` has passed, the "no disclosure before verification" rule is enforced by the backend rather than trusted to the prompt.

---

## 10. Open questions

1. **Is the browser voice demo acceptable as the primary end-to-end demonstration**, with the phone call as a documented secondary path? That keeps the project at $0. If the assignment will be judged on a real PSTN call, say so and phase 9 becomes mandatory rather than conditional.
2. **ElevenLabs account** — do you want to sign up yourself (I'd supply exact steps), given that I can't create accounts for you? The API key then goes in `.env` only, never in chat.
3. **Tunnel** — ngrok with the free static domain (recommended, stable URL), or `cloudflared` to avoid another signup?
4. **Offline simulator** — include it (recommended: it protects the 15-minute budget and makes the repo reviewable without any account), or drop it to stay minimal?
5. **Repo location** — initialise git in `C:\Users\yuvra\Voice Call Agent` as-is, or create a `voice-autopay-recovery/` subfolder?
6. **Your demo number** goes in `.env` only. Confirm it's a US/Canada mobile that can receive SMS, which Twilio's caller-ID verification requires — relevant only if we do phase 9.
