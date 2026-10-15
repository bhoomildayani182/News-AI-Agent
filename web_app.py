"""Local, on-demand web page for researching and emailing one news topic."""

from html import escape
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from secrets import token_urlsafe
import smtplib
from threading import Lock, Thread
from urllib.parse import parse_qs

import httpx
from google.genai import errors

from config import ConfigurationError, get_settings
from email_delivery import EmailConfigurationError, send_email, validate_email_settings
from stages import ResearchError, research_topic
from openrouter_client import OpenRouterError, ProviderResponseError, RunLimitReached
from run_tracking import record_delivery, usage_text


HOST = '127.0.0.1'
PORT = 8765
MAX_FORM_BYTES = 4096
LOCAL_AUTHORITIES = {f'{HOST}:{PORT}', f'localhost:{PORT}'}


class JobState:
    """Keep one active run so repeated clicks do not create duplicate charges or emails."""

    def __init__(self):
        self.lock = Lock()
        self.token = token_urlsafe(32)
        self.topic = ''
        self.status = 'idle'
        self.message = ''
        self.summary = {}
        self.continuation = None
        self.override = None

    def snapshot(self):
        with self.lock:
            return self.topic, self.status, self.message

    def start(self, topic, *, continue_run=False):
        with self.lock:
            if self.status == 'running':
                return False
            if continue_run:
                if not self.continuation or self.continuation['topic'] != topic:
                    return False
                self.override = self.continuation
            else:
                self.override = None
            self.continuation = None
            self.topic = topic
            self.summary = {}
            self.status = 'running'
            self.message = 'Researching sources and preparing your email. This can take a few minutes.'
        Thread(target=self._run, args=(topic,), daemon=True).start()
        return True

    def progress(self, message):
        with self.lock:
            self.message = message

    def run_summary(self):
        with self.lock:
            return dict(self.summary)

    def _run(self, topic):
        summary = {}
        delivery_started = False
        try:
            settings = get_settings()
            validate_email_settings()
            if settings.provider == 'openrouter':
                if self.override:
                    settings = replace(settings,
                        max_run_cost_usd=self.override['budget'],
                        max_research_rounds=self.override['rounds'],
                        research_max_tokens=self.override['research_tokens'],
                        report_max_tokens=self.override['report_tokens'],
                        recovery_max_tokens=self.override['tokens'])
                    result = research_topic(topic, settings, on_progress=self.progress,
                                            fresh=True, checkpoint=self.override['checkpoint'])
                else:
                    result = research_topic(topic, settings, on_progress=self.progress)
            else:
                result = research_topic(topic, settings)
            summary = getattr(result, 'run_summary', {})
            delivery_started = True
            record_delivery(summary, 'sending')
            send_email(topic, result)
            record_delivery(summary, 'accepted_by_smtp')
        except (ConfigurationError, EmailConfigurationError) as error:
            status, message = 'failed', str(error)
        except RunLimitReached as error:
            summary = getattr(error, 'run_summary', {})
            status, message = 'limit_reached', 'Agent limit reached. ' + str(error)[:600] + ' No email was sent.'
            sources = getattr(error, 'resume_sources', [])
            # One optional continuation, server-calculated bounds; never edit .env.
            used_kinds = self.override['used_kinds'] if self.override else []
            if (settings.provider == 'openrouter' and len(used_kinds) < 2
                    and error.kind not in used_kinds
                    and error.kind in {'cost', 'output', 'research'}
                    and getattr(error, 'checkpoint', None)):
                spent = max(0, float(summary.get('estimated_budget_used_usd',
                                    summary.get('reported_cost_usd', 0)) or 0))
                spent += self.override['previous_estimated_spend'] if self.override else 0
                cap = self.override['total_cap'] if self.override else settings.max_run_cost_usd
                # Extra search rounds do NOT authorize extra money. Only a cost
                # stop offers a higher TOTAL cap, explicitly approved in the UI.
                total_cap = min(cap * 2, 0.30) if error.kind == 'cost' else cap
                rounds = (min(settings.max_research_rounds + 2, 4)
                          if error.kind == 'research' else settings.max_research_rounds)
                output_factor = 2 if error.kind == 'output' else 1
                self.continuation = {'topic': topic, 'sources': sources,
                    'checkpoint': error.checkpoint,
                    'budget': max(0, total_cap - spent), 'total_cap': total_cap,
                    'previous_cost': float(summary.get('reported_cost_usd', 0) or 0)
                                     + (self.override['previous_cost'] if self.override else 0),
                    'previous_estimated_spend': spent,
                    'used_kinds': used_kinds + [error.kind],
                    'kind': error.kind, 'rounds': rounds,
                    'research_tokens': min(settings.research_max_tokens * output_factor, 16384),
                    'report_tokens': min(settings.report_max_tokens * output_factor, 16384),
                    'tokens': min(settings.recovery_max_tokens * output_factor, 32768)}
                if (self.continuation['budget'] <= 0 or total_cap < cap
                        or (error.kind == 'research' and rounds == settings.max_research_rounds)):
                    self.continuation = None
            if error.kind == 'research':
                message = ('Research incomplete: required coverage could not be verified within '
                           'the search allowance. Increasing the money budget alone will not fix '
                           'missing articles. ' + str(error)[:450] + ' No email was sent.')
            if sources:
                # A discovery bulletin is not a factual summary or a validated report.
                # Links only: do not send unchecked model claims or scraped passages.
                from stages import ResearchResult, Source, SOURCE_CITATION
                bulletin = ResearchResult(
                    '## Partial research — not a verified answer\n'
                    'The agent reached a limit before completing verification. '
                    'These are discovered reading links, not proof that they answer your question. '
                    'No completed comparison or verified conclusions are available.\n\n'
                    '## Why research stopped\n' + SOURCE_CITATION.sub('', str(error)[:600]) + '\n\n'
                    '## Reading links\n' + '\n'.join(
                        f'- Discovered source [S{i}]' for i in range(1, len(sources) + 1)),
                    [Source(s['title'], s['url'], s['publisher'], s.get('published_at', ''))
                     for s in sources], {**summary, 'partial': True})
                try:
                    record_delivery(summary, 'sending_partial')
                    send_email(topic, bulletin)
                    record_delivery(summary, 'partial_accepted_by_smtp')
                    message = message.replace('No email was sent.',
                        'A partial reading-links email was sent, clearly marked as unverified.')
                except Exception:
                    record_delivery(summary, 'partial_delivery_unconfirmed')
                    message = message.replace('No email was sent.',
                        'Partial email delivery is unconfirmed. Check your inbox before retrying.')
        except ResearchError as error:
            summary = getattr(error, 'run_summary', {})
            detail = str(error).replace(' No report or email produced.', '')[:600]
            status, message = 'failed', ('Research stopped safely: ' + detail +
                                         ' No email was sent. The app is still running.')
        except errors.APIError as error:
            code = getattr(error, 'code', None)
            status, message = 'failed', (
                'Gemini quota or rate limit reached. No email was sent.' if code == 429
                else 'Gemini request failed. Check model access and try again later. No email was sent.'
            )
        except OpenRouterError as error:
            summary = getattr(error, 'run_summary', {})
            if error.code in (402, 429):
                status = 'limit_reached'
                message = ('The provider account has insufficient credits. Check OpenRouter billing.'
                           if error.code == 402 else
                           'The provider rate/quota limit was reached. Wait before retrying and check OpenRouter usage limits.')
                message += ' No email was sent. Changing the agent run budget does not fix provider limits.'
            else:
                status, message = 'failed', f'OpenRouter could not complete the request (HTTP {error.code}). {error} No email was sent.'
        except ProviderResponseError as error:
            summary = getattr(error, 'run_summary', {})
            status, message = 'failed', str(error)[:600] + ' No email was sent.'
        except (httpx.HTTPError, smtplib.SMTPException, OSError) as error:
            summary = getattr(error, 'run_summary', summary)
            if delivery_started:
                record_delivery(summary, 'unconfirmed_check_inbox_before_retry')
            status, message = 'failed', ('Email delivery could not be confirmed. Check your inbox before retrying.'
                if delivery_started else 'Research connection failed. Check your network; no automatic paid retry was made.')
        except Exception as error:
            summary = getattr(error, 'run_summary', summary)
            status, message = 'failed', 'The run failed unexpectedly. No success confirmation is available.'
        else:
            status, message = 'sent', 'Report completed and email sent to the address in EMAIL_TO.'
        if self.override:
            summary['previous_reported_cost_usd'] = self.override['previous_cost']
            summary['total_reported_cost_usd'] = round(
                self.override['previous_cost'] + float(summary.get('reported_cost_usd', 0) or 0), 8)
            message += f' Combined reported cost: ${summary["total_reported_cost_usd"]:.6f}.'
            if summary.get('run_id'):
                record_delivery(summary, summary.get('delivery_status', 'not_started'))
        with self.lock:
            self.status, self.message = status, message
            self.summary = summary
        print(message, flush=True)


STATE = JobState()


def page(topic='', status='idle', message='', *, error='', summary=None):
    """Render only escaped user content; the page remains local to this computer."""
    safe_topic = escape(topic, quote=True)
    notice = f'<p class="notice {status}">{escape(message)}</p>' if message else ''
    if error:
        notice = f'<p class="notice failed">{escape(error)}</p>'
    if summary:
        notice += f'<p class="notice">{escape(usage_text(summary))}</p>'
    refresh = '<meta http-equiv="refresh" content="4;url=/status">' if status == 'running' else ''
    continuation = ''
    with STATE.lock:
        offer = dict(STATE.continuation) if STATE.continuation else None
    if status == 'limit_reached' and offer and topic == offer['topic']:
        button = ('Approve additional budget and continue' if offer['kind'] == 'cost'
                  else 'Continue within existing budget')
        continuation = (
            '<div class="notice limit_reached"><strong>Optional additional research</strong>'
            f'<p>Reason: {escape(message)} </p>'
            f'<p>Total budget guard for this request: ${offer["total_cap"]:.3f}, including '
            f'earlier work. Remaining allowance: ${offer["budget"]:.4f}. '
            f'Maximum total research rounds: {offer["rounds"]}. '
            f'Recovery output ceiling: {offer["tokens"]} tokens. '
            'Previous charges still apply. Completed steps are reused without additional API '
            'calls; the interrupted step resumes. A complete answer is not guaranteed. '
            'This checkpoint expires on a date/model change. Defaults stay unchanged.</p>'
            '<form method="post" action="/run">'
            f'<input type="hidden" name="token" value="{STATE.token}">'
            f'<input type="hidden" name="topic" value="{safe_topic}">'
            '<input type="hidden" name="continue" value="yes">'
            f'<button type="submit">{button}</button></form></div>')
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
.limit_reached {{background:#fff8df;color:#713f12;border:1px solid #f1d88a}}
small {{display:block;margin-top:18px;color:#52627b}}
</style></head><body><main><h1>AI News Research Agent</h1>
<p>Enter a news topic. The agent will research it and email a short, sourced report.</p>
<form method="post" action="/run">
<input type="hidden" name="token" value="{STATE.token}">
<label for="topic">News topic or question</label>
<textarea id="topic" name="topic" minlength="5" maxlength="500" required>{safe_topic}</textarea>
<button type="submit" {'disabled' if status == 'running' else ''}>Research and send email</button>
</form>{notice}{continuation}
<small>Runs one topic at a time. Keep this page open to see the result. Recent verified reports may be reused; new research uses API credits.</small>
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
        self._respond(200, page(topic, status, message, summary=STATE.run_summary()))

    def do_POST(self):
        if self.path != '/run':
            self._respond(404, page(error='Page not found.'))
            return
        # Support both local browser addresses without allowing cross-site posts.
        # The form token below remains required, including when Origin is absent.
        authority = self.headers.get('Host', '').lower()
        if authority not in LOCAL_AUTHORITIES or self.headers.get('Origin') not in (None, f'http://{authority}'):
            self._respond(403, page(error=(
                'Open http://127.0.0.1:8765 or http://localhost:8765 on this computer '
                'and refresh before submitting. Requests from other sites are blocked.')))
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
        continue_run = fields.get('continue', [''])[0] == 'yes'
        started = STATE.start(topic, continue_run=True) if continue_run else STATE.start(topic)
        if not started:
            current_topic, status, message = STATE.snapshot()
            self._respond(409, page(current_topic, status, message,
                error='This continuation is unavailable or a report is already running. Refresh the page.'))
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
    # Container networking needs all container interfaces, but browser Host/Origin
    # checks above still require a local browser address. Compose publishes loopback only.
    bind_host = os.getenv('APP_BIND_HOST', HOST).strip() or HOST
    if bind_host not in {HOST, '0.0.0.0'}:
        print('Setup error: APP_BIND_HOST must be 127.0.0.1 or 0.0.0.0.')
        return 1
    try:
        server = ThreadingHTTPServer((bind_host, PORT), Handler)
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
