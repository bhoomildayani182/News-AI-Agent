"""Local, on-demand web page for researching and emailing one news topic."""

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from secrets import token_urlsafe
import smtplib
from threading import Lock, Thread
from urllib.parse import parse_qs

import httpx
from google.genai import errors

from config import ConfigurationError, get_settings
from email_delivery import EmailConfigurationError, send_email, validate_email_settings
from stages import ResearchError, research_topic
from openrouter_client import OpenRouterError


HOST = '127.0.0.1'
PORT = 8765
MAX_FORM_BYTES = 4096


class JobState:
    """Keep one active run so repeated clicks do not create duplicate charges or emails."""

    def __init__(self):
        self.lock = Lock()
        self.token = token_urlsafe(32)
        self.topic = ''
        self.status = 'idle'
        self.message = ''

    def snapshot(self):
        with self.lock:
            return self.topic, self.status, self.message

    def start(self, topic):
        with self.lock:
            if self.status == 'running':
                return False
            self.topic = topic
            self.status = 'running'
            self.message = 'Researching sources and preparing your email. This can take a few minutes.'
        Thread(target=self._run, args=(topic,), daemon=True).start()
        return True

    def _run(self, topic):
        try:
            settings = get_settings()
            validate_email_settings()
            result = research_topic(topic, settings)
            send_email(topic, result)
        except (ConfigurationError, EmailConfigurationError) as error:
            status, message = 'failed', str(error)
        except ResearchError:
            status, message = 'failed', 'The report did not pass research checks. No email was sent.'
        except errors.APIError as error:
            code = getattr(error, 'code', None)
            status, message = 'failed', (
                'Gemini quota or rate limit reached. No email was sent.' if code == 429
                else 'Gemini request failed. Check model access and try again later. No email was sent.'
            )
        except OpenRouterError as error:
            status, message = 'failed', f'OpenRouter HTTP {error.code}: {error} No email was sent.'
        except (httpx.HTTPError, smtplib.SMTPException, OSError):
            status, message = 'failed', 'Research or email connection failed. Check your network and email settings.'
        except Exception:
            status, message = 'failed', 'The run failed unexpectedly. No success confirmation is available.'
        else:
            status, message = 'sent', 'Report completed and email sent to the address in EMAIL_TO.'
        with self.lock:
            self.status, self.message = status, message
        print(message, flush=True)


STATE = JobState()


def page(topic='', status='idle', message='', *, error=''):
    """Render only escaped user content; the page remains local to this computer."""
    safe_topic = escape(topic, quote=True)
    notice = f'<p class="notice {status}">{escape(message)}</p>' if message else ''
    if error:
        notice = f'<p class="notice failed">{escape(error)}</p>'
    refresh = '<meta http-equiv="refresh" content="4;url=/status">' if status == 'running' else ''
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI News Research Agent</title>{refresh}
<style>
body {{font:16px/1.6 system-ui,sans-serif;background:#eef2f7;color:#17243c;margin:0;padding:48px 16px}}
main {{max-width:620px;margin:auto;background:white;padding:32px;border-radius:16px;box-shadow:0 8px 28px #1b315014}}
h1 {{font-size:1.7rem;margin:0 0 8px}} p {{margin:0 0 20px}} label {{display:block;font-weight:650;margin-bottom:8px}}
textarea {{width:100%;box-sizing:border-box;padding:12px;font:inherit;border:1px solid #9aa8bd;border-radius:8px;min-height:100px}}
button {{margin-top:16px;background:#164aa2;color:white;border:0;border-radius:8px;padding:12px 20px;font:inherit;cursor:pointer}}
button:disabled {{background:#98a5b9;cursor:wait}} .notice {{padding:14px;border-radius:8px;margin:20px 0 0}}
.running {{background:#e7f0ff}} .sent {{background:#e3f6e8}} .failed {{background:#fff0ed}}
small {{display:block;margin-top:18px;color:#52627b}}
</style></head><body><main><h1>AI News Research Agent</h1>
<p>Enter a news topic. The agent will research it and email a short, sourced report.</p>
<form method="post" action="/run">
<input type="hidden" name="token" value="{STATE.token}">
<label for="topic">News topic or question</label>
<textarea id="topic" name="topic" minlength="5" maxlength="500" required>{safe_topic}</textarea>
<button type="submit" {'disabled' if status == 'running' else ''}>Research and send email</button>
</form>{notice}
<small>Runs one topic at a time. Keep this page open to see the result. Each submitted topic uses your selected provider's API quota.</small>
</main></body></html>'''.encode('utf-8')


class Handler(BaseHTTPRequestHandler):
    def _respond(self, code, content):
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path not in ('/', '/status'):
            self._respond(404, page(error='Page not found.'))
            return
        topic, status, message = STATE.snapshot()
        self._respond(200, page(topic, status, message))

    def do_POST(self):
        if self.path != '/run':
            self._respond(404, page(error='Page not found.'))
            return
        if self.headers.get('Host') != f'{HOST}:{PORT}' or self.headers.get('Origin') not in (None, f'http://{HOST}:{PORT}'):
            self._respond(403, page(error='Request not allowed.'))
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_FORM_BYTES or self.headers.get('Content-Type', '').split(';')[0] != 'application/x-www-form-urlencoded':
            self._respond(400, page(error='Invalid form submission.'))
            return
        fields = parse_qs(self.rfile.read(length).decode('utf-8', errors='replace'))
        if fields.get('token', [''])[0] != STATE.token:
            self._respond(403, page(error='Invalid form token. Refresh the page and try again.'))
            return
        topic = fields.get('topic', [''])[0].strip()
        if not 5 <= len(topic) <= 500 or '\n' in topic or '\r' in topic:
            self._respond(400, page(topic, error='Enter a single-line topic between 5 and 500 characters.'))
            return
        if not STATE.start(topic):
            current_topic, status, message = STATE.snapshot()
            self._respond(409, page(current_topic, status, message, error='A report is already running.'))
            return
        self.send_response(303)
        self.send_header('Location', '/status')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def log_message(self, format, *args):
        # Avoid logging submitted topics and form content.
        pass


def main():
    try:
        get_settings()
        validate_email_settings()
    except (ConfigurationError, EmailConfigurationError) as error:
        print(f'Setup error: {error}')
        return 1
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError:
        print(f'Cannot start the local page on {HOST}:{PORT}. Check whether the port is in use.')
        return 1
    print(f'Open http://{HOST}:{PORT} in your browser. Press Ctrl+C to stop.')
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nStopped.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
