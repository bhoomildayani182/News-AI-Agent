# AI News Research Agent

A Python application that researches a news topic using Gemini or OpenRouter,
explains it in plain English, and optionally emails the report.

## How the agent works

1. You enter a news topic or question.
2. The agent researches current information and collects source links.
3. With OpenRouter, it checks source diversity and same-story relevance, and decides whether to write or
   search for a missing fact. Research is limited to two rounds.
4. A reviewer checks claims against the excerpts, explicitly assesses each cited
   source's relevance to the event, country and period, and returns a corrected brief,
   requests targeted evidence, or stops publication. If the final output has an
   invalid citation or missing relevance check, one evidence-only repair attempt
   is allowed within the same spending guard. It cannot reuse a source already
   rejected as irrelevant or bypass the final checks.
   The reviewer also checks that the report answers every requested part—for
   example, the RBA's stated reasons, not merely the effects on mortgage holders.
5. A successful report appears in the terminal, an email preview, or an email,
   depending on the command you choose.

Reports contain **Summary**, **In simple terms**, **Why it matters**,
**What different sources are saying**, **What to watch next**, and **Sources**.
The brief targets 180–240 words, with a 300-word limit excluding the source list.
Emails include a formatted HTML version and a plain-text alternative.

With OpenRouter, the agent chooses an evidence policy for the question. Factual
summaries of an official report, statistical release, policy decision or product
announcement may use the original official issuer alone **only when its excerpts answer the question**.
A single-publisher report is visibly labelled in the terminal and both email formats:
"This brief relies on one official publisher. Independent corroboration is not included."
The official-domain check is a safeguard, not proof of relevance or accuracy;
the model must also assess issuer identity, actual findings and question coverage.
This exception covers `.gov.au`, `.gov`, `.gov.uk` and `.govt.nz` sources, plus
explicit first-party documentation/news hosts for Google Cloud (`cloud.google.com`,
`docs.cloud.google.com`, `developers.google.com`), AWS (`aws.amazon.com`,
`docs.aws.amazon.com`) and Microsoft (`learn.microsoft.com`, `docs.microsoft.com`,
`azure.microsoft.com`). Matching is by exact vendor hostname, not a loose substring.
Unknown commercial websites and user-hosted pages do not qualify automatically.
Vendor documentation describes the vendor's claims; it is not independent verification.
For latest product changes, the agent asks for dated release notes/announcements,
not an undated documentation page. Comparisons, disputed claims and
independent verification still require two relevant publishers covering the same story—not
unrelated articles that merely share a keyword. For example, UK inflation data
cannot substitute for another source covering an Australian ABS release unless
you asked for an international comparison. When evidence remains insufficient
after the bounded follow-up, the agent stops without sending an email.
For statistical releases, it asks for relevant headline/underlying context and
keeps release dates, reference periods and annual/monthly changes distinct.
Delivered citations are numbered consecutively, with matching links in the
HTML email, plain-text email and terminal report. Internal research IDs remain intact.

AI review does not guarantee
accuracy or independent sources. Check important claims against the linked
articles before relying on them.

## Start here: choose Docker or Python

Both options run the same local web agent at `http://127.0.0.1:8765`.
Use only one option at a time so they do not compete for the same port.
Keep the project anywhere; run commands from the folder containing `compose.yaml`
and `main.py`.

### Option A — Docker (no local Python installation needed)

1. Install Docker Desktop on macOS/Windows, or Docker Engine with the Compose
   plugin on Linux. Start Docker before running the command below. Use a current
   Docker Compose version supporting `--wait`.
2. If `.env` does not exist, copy `.env.example` to `.env`. Fill in your AI provider
   key and the SMTP/email settings described below. Do not overwrite an existing `.env`.
3. Start the agent with **one command**:

```bash
docker compose up --build --wait
```

The first run downloads Python and installs dependencies inside the image.
The command waits for the local page to be healthy and leaves the container running
in the background. Open **http://127.0.0.1:8765**, enter a topic and click
**Research and send email**. Use this exact address, not `0.0.0.0` or `localhost`.
Starting the container or its health check does not perform paid research or send email.
Submitting a topic can incur AI/search charges and sends email only after successful validation.

View status/logs:

```bash
docker compose ps
docker compose logs --tail=100 agent
```

Stop the agent without deleting its saved history/cache:

```bash
docker compose down
```

After changing `.env`, recreate the container to load the new settings:

```bash
docker compose up --build --force-recreate --wait
```

The image runs as a non-root user. API keys/passwords are supplied at runtime from
`.env`; that file is excluded from the image and build context. Docker administrators
can inspect container environment variables, so keep Docker access and `.env` private.
The web port is published only on your own computer, not your LAN or the internet.
This is not a public production hosting configuration.

Research history/cache is stored in the named `agent-data` Docker volume. It is
separate from a Python installation's local `.agent-data` folder and survives
ordinary container recreation. **Do not use `docker compose down -v` unless you
intend to permanently delete the Docker history/cache.** Do not stop/recreate the
container during active research; in-progress work is not resumed automatically.

Optional CLI report/history commands inside Docker:

```bash
docker compose exec agent python main.py
docker compose exec agent python main.py --history
```

The container's application filesystem is read-only. `--preview-email` writes an
HTML file, so use the Python option below for that preview command. Normal web
email delivery and terminal reports work without writing an HTML preview file.

### Option B — Without Docker (local Python)

Follow the Python setup and provider/email configuration below. Once configured,
activate your virtual environment and start the web agent with:

```bash
python web_app.py
```

Open **http://127.0.0.1:8765**. Keep the terminal open; press Ctrl+C to stop.

## 1. Open the project folder (Python option)

Install Python 3.10 or newer and open a terminal in the folder containing `main.py` and
`requirements.txt`. Run all commands below from that folder.

**There is no fixed installation path.** You can store the project anywhere.
Keep `.env` beside `main.py`; the application finds it there.

## 2. Set up Python (skip for Docker)

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the virtual environment already exists, skip its creation. Activate it again
whenever you open a new terminal. After activation, use `python` as shown below.

## 3. Choose an AI provider

If `.env` does not exist, copy `.env.example` and name the copy `.env`.
Do not overwrite an existing configuration.

For Gemini, create a key in [Google AI Studio](https://aistudio.google.com/app/apikey)
and set these lines in your private `.env`:

```dotenv
AI_PROVIDER=gemini
GEMINI_API_KEY=your_real_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_REVIEW_MODEL=gemini-3.7-flash
PREFERRED_SOURCES=RBA,ABC,Reuters,AFR
```

For OpenRouter, create a key in your [OpenRouter dashboard](https://openrouter.ai/keys)
and set these lines instead:

```dotenv
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=your_real_openrouter_key_here
OPENROUTER_MODEL=google/gemini-3.1-flash-lite
OPENROUTER_REVIEW_MODEL=google/gemini-3.1-flash-lite
OPENROUTER_MAX_RUN_COST_USD=0.03
RESEARCH_MAX_ROUNDS=2
REPORT_CACHE_MINUTES=15
PREFERRED_SOURCES=RBA,ABC,Reuters,AFR
```

Only the selected provider's key is used. Existing `GEMINI_*` lines may remain
in `.env` when OpenRouter is selected. Both tasks default to Flash-Lite to keep
cost low; you can explicitly choose another review model. Both models must be available to
your chosen account. Existing model values in `.env` override new defaults:
update both model lines above if migrating from the older, more expensive setup.
OpenRouter uses its [web search server tool](https://openrouter.ai/docs/guides/features/server-tools/web-search)
with Parallel's fast mode for source excerpts. Each research request permits one
search and up to five results. Search charges are additional to model usage.
The agent checks evidence sufficiency before proceeding to review. Missing core
findings trigger targeted research, not a review of a knowingly incomplete draft.
It prefers multiple relevant publishers, with the official-primary exception above. It strips
tracking parameters, merges duplicate links, removes identical syndicated excerpts,
and never treats a bare URL as evidence. A domain is a diversity check, not proof
of editorial independence. Article dates are displayed only when returned in metadata.
For supported official HTML pages, the agent also attempts a direct read of the
original page before drafting. Retrieval uses an exact-host allowlist, HTTPS-only
redirects to that allowlist, a four-MiB page-size limit, per-request timeouts and
at most two page attempts per research round. It adds no LLM request, but can add
latency and evidence tokens. If retrieval is unavailable, the search excerpts
remain available and the failed fetch is recorded in local run history.
Regional Google documentation is treated as one publisher, not independent voices.

Change the comma-separated `PREFERRED_SOURCES` list to your preferred publishers.
These are preferences, not a guarantee that every publisher will appear.
Preferences are saved in configuration; the agent does not learn them from
previous conversations.

Keep keys and passwords private. Never share or commit `.env`.
Existing terminal environment variables take priority over values in `.env`.

## 4. Run live research

### Use a simple local web page

After setting the selected provider and email values in `.env` (see section 6), start the
agent from the project folder:

```bash
python web_app.py
```

Open `http://127.0.0.1:8765` in a browser on the same computer. Enter a topic
and click **Research and send email**. The page shows when research is running
and whether the email was sent. Only one topic runs at a time, so repeated clicks
cannot start overlapping reports. The configured `EMAIL_TO` address receives the
report. Keep the terminal running while the agent works; press Ctrl+C to stop it.

This page is available only on your computer. It is not a public website or a
scheduled service. OpenRouter may reuse a recently verified report for a dated,
non-time-sensitive question within the configured cache period. Questions containing
"latest", "most recent", "current", "today", "tonight", "now" or "breaking" always
perform fresh research. Keep the server running while accepting topics.

### Use the command line

```bash
python main.py
```

At `Enter news topic:`, enter a question such as:

```text
What did the RBA announce in its most recent cash rate decision?
```

An OpenRouter run normally uses three model requests: search, decision/draft,
and review. If essential evidence is missing, it can perform one targeted follow-up
round, with at most six normal model requests and two search-enabled requests total.
One additional evidence-only repair is permitted for final validation problems
(at most seven calls before length recovery). It makes no new search, respects the
same run budget, and cannot relabel a rejected source to force success.
If a response is cut off by its output allowance (`finish_reason=length`), one
additional stage-only regeneration is permitted for the whole run, after checking
the budget. This gives a maximum of eight API calls with two rounds, one content
repair and one length recovery. A recovered draft/review does not search again.
An incomplete research response with usable source excerpts is reused without
paying for another search. A research retry without evidence still counts toward
the two-search-request maximum. HTTP failures and malformed JSON do not trigger
automatic paid retries.
Cached runs make no API requests. Gemini retains its original three-call mode
and one bounded correction. Live research may incur charges.

## 5. Preview the email

```bash
python main.py --preview-email
```

Enter your topic. This uses a recent verified OpenRouter report if available,
otherwise it runs **live research without sending email**.
It prints a text preview and saves `email-preview.html` in the project folder.
Open that file in your browser to see the formatted email.
Each preview replaces the previous preview file.

To check only the layout without API calls or charges:

```bash
python main.py --demo --preview-email
```

The demo contains fictional information; it does not research current news.

## 6. Configure and send email

Add your email provider's SMTP settings to `.env`:

```dotenv
SMTP_HOST=your_smtp_server
SMTP_PORT=465
SMTP_USERNAME=your_smtp_username
SMTP_PASSWORD=your_smtp_password_or_app_password
EMAIL_FROM=you@example.com
EMAIL_TO=recipient@example.com
EMAIL_SENDER_NAME=AI News Research Agent
EMAIL_SUBJECT_TITLE=News Update
```

- Port `465` uses SMTP over TLS; port `587` uses STARTTLS.
- Use an app password if your email provider requires one.
- `EMAIL_FROM` must be an address your provider allows you to send from.
- `EMAIL_TO` is the recipient's address.
- `EMAIL_SENDER_NAME` controls the displayed sender name.
- `EMAIL_SUBJECT_TITLE` prefixes the subject, for example: `News Update: <topic>`.

Run:

```bash
python main.py --send-email
```

Enter the topic. The agent checks email configuration, researches the topic,
and sends one email only after the report passes its checks.
Delivery still depends on your email provider.

With OpenRouter, repeating the exact topic and settings within the cache period
reuses the verified report. You can preview, inspect it, and then use `--send-email`
without paying for another research run. Each explicit send command still sends
an email; do not repeat it unless you want another copy. Preview and send options
cannot be combined. Use `--fresh` to deliberately research again:

```bash
python main.py --preview-email --fresh
```

The cache is scoped by topic, UTC date, provider, models, key fingerprint and
source preferences. Failed reports are never cached. Set `REPORT_CACHE_MINUTES=0`
to disable it. Emails show the original research timestamp.
Explicitly time-sensitive questions bypass this cache automatically, even when
`REPORT_CACHE_MINUTES` is greater than zero. Fresh research incurs normal charges.

## Cost controls and run history

### Balanced output allowances and clear limit notices

These defaults apply automatically even when an older `.env` does not contain them:

```dotenv
OPENROUTER_RESEARCH_MAX_TOKENS=2400
OPENROUTER_REPORT_MAX_TOKENS=4096
OPENROUTER_RECOVERY_MAX_TOKENS=8192
OPENROUTER_MAX_LENGTH_RETRIES=1
OPENROUTER_REASONING_EFFORT=low
```

Output allowances cover the full structured response and, where applicable,
internal reasoning—not only the short email. The email remains capped at 300
words. A larger allowance is a ceiling, not an automatic charge for that many tokens.
Low reasoning effort is requested only when the model advertises reasoning
support; use `provider_default` to omit that parameter. The selected model has
not been changed. The provider's advertised output cap is respected when available.

The app first tries to recover a cut-off response once using the same evidence,
a larger allowance and concise output instructions. It never concatenates partial
JSON, bypasses source checks, increases your budget automatically or retries SMTP.
Every attempt, including unsuccessful generation, can incur charges.

When an allowance is exhausted, the web page shows an amber **Agent limit reached**
notice—not a fabricated HTTP 502. It explains which setting to review:

- Output length: narrow the topic or increase `OPENROUTER_RECOVERY_MAX_TOKENS`.
- Run cost: review actual usage, then increase `OPENROUTER_MAX_RUN_COST_USD` only
  if you accept potentially higher spending. The default remains `$0.03`.
- Research allowance: narrow the question; `RESEARCH_MAX_ROUNDS` supports 1–2.
- Provider HTTP 402/429: check account credits or rate/quota limits; changing the
  app's budget does not fix an exhausted provider allowance.

Research/report limits support 256–16384 tokens; the recovery ceiling must cover
both limits and be at most 32768. Length retries support only 0 or 1 per run.
After changing `.env`, restart Python or run
`docker compose up --build --force-recreate --wait`. No email is sent when a
complete, verified report cannot be produced.

### Spending and diagnostics

OpenRouter checks published model prices before a paid request, caps output
tokens, disables the old account-level web plugin on writing/review requests,
and checks estimated spending against `OPENROUTER_MAX_RUN_COST_USD`. Requests
stop if the next estimate would exceed the configured budget.

This is an application budget guard, not a guaranteed billing cap: estimates
and search prices can differ from final charges, and in-flight costs cannot
be undone. For a provider-enforced limit, also set a spending limit on your
[OpenRouter API key](https://openrouter.ai/keys). The default budget is $0.03 USD,
not a promised price for a report. The application does not raise account limits.

The terminal and web result show API calls, elapsed time, provider-reported cost,
and reported searches. Missing usage is explicitly marked unavailable, never
assumed to be free. Counts describe application requests; OpenRouter may perform
internal model steps while executing its search tool.

View the latest five research runs without API calls:

```bash
python main.py --history
```

The local `.agent-data/runs.sqlite3` contains research stages, decisions, failures,
usage and cached reports. It is ignored by Git. Keys and raw API responses are
not logged. Email acceptance is recorded separately from research; an SMTP success message means the
mail server accepted the email, not that inbox delivery has been independently verified.
`--history` also shows actual HTTP status, finish reason, requested output limit,
reported output/reasoning tokens and whether a length recovery was used. These
fields help distinguish truncation from an actual upstream HTTP failure.

## Optional: more detailed research

```bash
python main.py --thorough
```

For Gemini, this separates the six research stages and costs more than its
default mode. OpenRouter always uses the bounded loop above, including when
`--thorough` is supplied. Neither mode has unlimited autonomous research.

## Main project files

| File | Purpose |
|---|---|
| `main.py` | Starts the agent, accepts a topic, and selects display, preview, or email delivery. |
| `web_app.py` | Provides a local topic form and sends approved reports to `EMAIL_TO`. |
| `config.py` | Loads the selected provider, API key, models, and source preferences. |
| `openrouter_client.py` | Bounds requests, checks prices, and collects OpenRouter usage. |
| `research_agent.py` | Runs the bounded OpenRouter research decision loop and source checks. |
| `run_tracking.py` | Stores local research history and recent verified reports. |
| `stages.py` | Researches, compares evidence, writes, and validates reports. |
| `email_format.py` | Creates the HTML email and plain-text alternative. |
| `email_delivery.py` | Loads email settings and sends through SMTP. |
| `demo.py` | Supplies fictional content for offline previews. |
| `requirements.txt` | Lists required Python packages. |
| `.env.example` | Template for your private `.env` configuration. |

## Common issues

- **Missing API key:** set the selected provider's API key in `.env`.
- **HTTP 402 from OpenRouter:** add credits or check the account's billing.
- **HTTP 429:** check the selected provider's quota, rate limits, and spending cap.
  Repeated retries will not fix an exhausted allowance.
- **Model unavailable / access denied:** check the selected model names,
  API key, and model access for your account.
- **Few or no source links:** the web search tool may not have run or may not
  return actual findings or sufficient relevant evidence; the agent makes at most one
  targeted follow-up before stopping without email.
- **Budget guard stopped:** inspect `--history` before increasing the budget.
- **Response cut off / agent limit reached:** one budget-checked length recovery
  is attempted when enabled. If it cannot finish, follow the notice and inspect
  `--history`; never assume it is a provider outage or increase every limit blindly.
- **Validation failed:** no email is sent. Try a more specific question with
  clearer official coverage, then inspect the sources.
- **Email failed:** check SMTP host, port, credentials, and allowed sender address.
- **Missing Python package:** activate `.venv` and install `requirements.txt`.

For all available command options, run `python main.py --help`.
