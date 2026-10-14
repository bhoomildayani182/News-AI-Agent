"""Private local run history and a short cache; no keys or raw prompts are saved."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from config import ConfigurationError


DATA_PATH = Path(__file__).resolve().parent / '.agent-data' / 'runs.sqlite3'


class RunStore:
    def __init__(self, path=None):
        self.path = Path(path or DATA_PATH)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as db:
                db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
                db.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, created REAL, data TEXT)')
        except (OSError, sqlite3.Error):
            raise ConfigurationError('Cannot open .agent-data run history. Use a writable project folder.') from None

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def save_run(self, summary):
        try:
            with self.connect() as db:
                db.execute('INSERT OR REPLACE INTO runs VALUES (?, ?)',
                           (summary['run_id'], json.dumps(summary)))
        except sqlite3.Error:
            raise ConfigurationError('Cannot write .agent-data run history. Use a writable project folder.') from None

    def get_cached(self, key, minutes):
        if not minutes:
            return None
        with self.connect() as db:
            row = db.execute('SELECT created, data FROM cache WHERE key=?', (key,)).fetchone()
        if not row or time.time() - row[0] > minutes * 60 or row[0] > time.time():
            return None
        try:
            return json.loads(row[1])
        except (ValueError, TypeError):
            return None

    def cache(self, key, result):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO cache VALUES (?, ?, ?)',
                       (key, time.time(), json.dumps(result)))
            db.execute('DELETE FROM cache WHERE created < ?', (time.time() - 3600,))

    def recent(self, limit=5):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                'SELECT data FROM runs ORDER BY rowid DESC LIMIT ?', (limit,))]


def cache_key(topic, settings):
    # Scope caches by date, provider, models, preferences and credential fingerprint.
    value = ['v2', ' '.join(topic.casefold().split()), settings.provider, settings.model,
             settings.review_model, settings.preferred_sources,
             datetime.now(timezone.utc).date().isoformat(),
             hashlib.sha256(settings.api_key.encode()).hexdigest()]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


@dataclass
class RunLog:
    store: RunStore
    model: str
    reviewer: str
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    started: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage: str = 'starting'
    status: str = 'running'
    calls: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    cache_hit: bool = False
    report_generated_at: str = ''
    source_count: int = 0
    publishers: list = field(default_factory=list)
    error: str = ''
    elapsed_seconds: float = 0
    clock_started: float = field(default_factory=time.monotonic, repr=False)

    def summary(self):
        reported = [call['cost_usd'] for call in self.calls if call['cost_usd'] is not None]
        return dict(run_id=self.run_id, started=self.started, provider='openrouter',
                    model=self.model, reviewer=self.reviewer, stage=self.stage,
                    status=self.status, calls=self.calls, decisions=self.decisions,
                    api_calls=len(self.calls), cache_hit=self.cache_hit,
                    report_generated_at=self.report_generated_at or self.started,
                    source_count=self.source_count, publishers=self.publishers,
                    reported_cost_usd=round(sum(reported), 8),
                    cost_complete=all(call['cost_usd'] is not None for call in self.calls),
                    estimated_budget_used_usd=round(sum(
                        call['cost_usd'] if call['cost_usd'] is not None else call['reserved_usd']
                        for call in self.calls), 8),
                    input_tokens=sum(call.get('input_tokens', 0) for call in self.calls),
                    output_tokens=sum(call.get('output_tokens', 0) for call in self.calls),
                    search_requests=sum(call.get('search_requests') or 0 for call in self.calls),
                    search_count_complete=all(not call['search_enabled'] or
                        call.get('search_requests') is not None for call in self.calls),
                    elapsed_seconds=round(time.monotonic() - self.clock_started, 2),
                    error=self.error)

    def save(self):
        self.store.save_run(self.summary())


def usage_text(summary):
    cost = summary.get('reported_cost_usd', 0)
    cost_text = f'reported cost ${cost:.6f}'
    if not summary.get('cost_complete'):
        cost_text += ' (partial; some charges unavailable—check OpenRouter Activity)'
    searches = f"{summary.get('search_requests', 0)} searches"
    if not summary.get('search_count_complete', True):
        searches = (searches + ' reported (some unavailable)' if summary.get('search_requests')
                    else 'search count unavailable')
    return (f"Run {summary['run_id']}: {summary.get('api_calls', 0)} API calls; "
            f"{searches}; {cost_text}; "
            f"{summary.get('elapsed_seconds', 0):.1f}s" +
            ('; reused a recent verified report' if summary.get('cache_hit') else ''))


def record_delivery(summary, status):
    """Keep email acceptance separate from the research outcome."""
    if summary:
        summary['delivery_status'] = status
        try:
            RunStore().save_run(summary)
        except (OSError, sqlite3.Error, ConfigurationError):
            print('Could not update the local email status; check the SMTP result shown separately.')
