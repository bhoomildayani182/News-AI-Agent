# AI News Research Agent

A Python command-line application that researches a news topic using Gemini and
Google Search, explains it in plain English, and optionally emails the report.

## How the agent works

1. You enter a news topic or question.
2. The agent researches current information and collects source links.
3. It compares the evidence and writes a short, plain-English brief.
4. A review checks the draft against the collected evidence. A rejected draft
   can be corrected once and reviewed again.
5. A successful report appears in the terminal, an email preview, or an email,
   depending on the command you choose.

Reports contain **Summary**, **In simple terms**, **Why it matters**,
**What different sources are saying**, **What to watch next**, and **Sources**.
The brief targets 180–240 words, with a 300-word limit excluding the source list.
Emails include a formatted HTML version and a plain-text alternative.

The agent requires multiple source links, but AI review does not guarantee
accuracy or independent sources. Check important claims against the linked
articles before relying on them.

## 1. Open the project folder

Install Python 3 and open a terminal in the folder containing `main.py` and
`requirements.txt`. Run all commands below from that folder.

**There is no fixed installation path.** You can store the project anywhere.
Keep `.env` beside `main.py`; the application finds it there.

## 2. Set up Python

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

## 3. Configure Gemini

If `.env` does not exist, copy `.env.example` and name the copy `.env`.
Do not overwrite an existing configuration.

Create an API key in [Google AI Studio](https://aistudio.google.com/app/apikey),
then edit these settings in your private `.env`:

```dotenv
GEMINI_API_KEY=your_real_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_REVIEW_MODEL=gemini-3.7-flash
PREFERRED_SOURCES=RBA,ABC,Reuters,AFR
```

The application uses the **Gemini Developer API**, not Vertex AI or OpenRouter.
Research and drafting use `GEMINI_MODEL`; the final evidence review uses
`GEMINI_REVIEW_MODEL`. Your Google account must have access to the configured
models, and the research model must support Google Search.

Change the comma-separated `PREFERRED_SOURCES` list to your preferred publishers.
These are preferences, not a guarantee that every publisher will appear.
Preferences are saved in configuration; the agent does not learn them from
previous conversations.

Keep keys and passwords private. Never share or commit `.env`.
Existing terminal environment variables take priority over values in `.env`.

## 4. Run live research

### Use a simple local web page

After setting the Gemini and email values in `.env` (see section 6), start the
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
scheduled service. Every submission starts fresh research and uses Gemini API
quota. You do not need to keep the page running between requests.

### Use the command line

```bash
python main.py
```

At `Enter news topic:`, enter a question such as:

```text
What did the RBA announce in its most recent cash rate decision?
```

The default mode researches, writes, and reviews the report using three model
requests, before any retries or correction. A correction and re-review can add
two requests. Live research uses your Google API quota and may incur charges.

## 5. Preview the email

```bash
python main.py --preview-email
```

Enter your topic. This runs **live research but does not send email**.
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

**Each live command starts fresh research.** Running `--send-email` after a
preview does not send the saved preview; it generates a new report and uses
API quota again. Preview and send options cannot be combined.

## Optional: more detailed research

```bash
python main.py --thorough
```

This separates planning, research, source evaluation, comparison, writing, and
review into six model requests. It is slower and uses more quota than the
default mode. It can also be combined with `--preview-email` or `--send-email`.

## Main project files

| File | Purpose |
|---|---|
| `main.py` | Starts the agent, accepts a topic, and selects display, preview, or email delivery. |
| `web_app.py` | Provides a local topic form and sends approved reports to `EMAIL_TO`. |
| `config.py` | Loads the API key, models, and source preferences. |
| `stages.py` | Researches, compares evidence, writes, and validates reports. |
| `email_format.py` | Creates the HTML email and plain-text alternative. |
| `email_delivery.py` | Loads email settings and sends through SMTP. |
| `demo.py` | Supplies fictional content for offline previews. |
| `requirements.txt` | Lists required Python packages. |
| `.env.example` | Template for your private `.env` configuration. |

## Common issues

- **Missing API key:** set `GEMINI_API_KEY` in `.env`.
- **HTTP 429:** check the Google project's quota, rate limits, and spending cap.
  Repeated retries will not fix an exhausted allowance.
- **Model unavailable / access denied:** check the configured model names,
  API key, and model access for your Google project.
- **Validation failed:** no email is sent. Try a more specific question with
  clearer official coverage, then inspect the sources.
- **Email failed:** check SMTP host, port, credentials, and allowed sender address.
- **Missing Python package:** activate `.venv` and install `requirements.txt`.

For all available command options, run `python main.py --help`.
