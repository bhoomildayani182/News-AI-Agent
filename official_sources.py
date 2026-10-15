"""Read a small number of known official pages; no LLM calls or arbitrary URL fetching."""

from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit

import httpx


# Exact hosts only: no user-supplied domain expansion, HTTP, credentials or custom ports.
FETCH_HOSTS = frozenset({
    'cloud.google.com', 'docs.cloud.google.com', 'developers.google.com',
    'aws.amazon.com', 'docs.aws.amazon.com', 'learn.microsoft.com',
    'docs.microsoft.com', 'azure.microsoft.com',
    'rba.gov.au', 'www.rba.gov.au', 'abs.gov.au', 'www.abs.gov.au',
    'cyber.gov.au', 'www.cyber.gov.au',
})
MAX_BYTES = 4 * 1024 * 1024
MAX_CHARS = 7000


def latest_rba_statement(topic, *, include_discovery=False):
    """Discover the latest statement from RBA's own index, never a fixed date/URL."""
    if not (re.search(r'\bRBA\b|Reserve Bank of Australia', topic, re.I)
            and re.search(r'\blatest\b|most recent|current', topic, re.I)
            and re.search(r'decision|cash rate|interest rate', topic, re.I)):
        return []
    index = 'https://www.rba.gov.au/monetary-policy/int-rate-decisions/'
    try:
        with httpx.Client(timeout=httpx.Timeout(12, connect=5), follow_redirects=False) as client:
            with client.stream('GET', index) as response:
                response.raise_for_status()
                if 'text/html' not in response.headers.get('content-type', '').lower():
                    return []
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_BYTES:
                        return []
        links = re.findall(r'href=[\"\'](/media-releases/\d{4}/mr-\d{2}-\d+\.html)[\"\']',
                           content.decode('utf-8', errors='replace'))
        if not links:
            return []
        source = dict(id='S1', title='RBA monetary policy decision (official statement)',
                      url=urljoin(index, links[0]), publisher='rba.gov.au', published_at='', excerpt='')
        # This date comes from the official index label, not a guessed URL date.
        label = re.search(re.escape(links[0]) + r'[\"\'][^>]*>([^<]+)</a>',
                          content.decode('utf-8', errors='replace'), re.I)
        if label:
            try:
                source['event_date'] = datetime.strptime(label.group(1).strip(), '%d %B %Y').date().isoformat()
            except ValueError:
                pass
        refresh_sources([source], topic)
        # Never publish a discovered URL or placeholder as retrieved evidence.
        if source.get('evidence_origin') == 'official_page':
            return [source]
        # A URL discovered on the index is a search target, NOT factual evidence.
        return [source] if include_discovery else []
    except (httpx.HTTPError, ValueError, UnicodeError):
        return []


def safe_url(url):
    try:
        parts = urlsplit(url)
        return (parts.scheme == 'https' and parts.hostname in FETCH_HOSTS
                and not parts.username and not parts.password and parts.port in (None, 443))
    except (ValueError, TypeError):
        return False


class PageText(HTMLParser):
    """Prefer main/article content and omit navigation, code and executable markup."""

    SKIP = {'script', 'style', 'nav', 'header', 'footer', 'noscript', 'svg'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = []
        self.inside = 0
        self.main = []
        self.body = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.hidden.append(tag)
        if tag in {'main', 'article'}:
            self.inside += 1

    def handle_endtag(self, tag):
        if tag in self.hidden:
            self.hidden = self.hidden[:self.hidden.index(tag)]
        if tag in {'p', 'div', 'h1', 'h2', 'h3', 'li', 'tr', 'section', 'br'}:
            self.handle_data('\n')
        if tag in {'main', 'article'}:
            self.inside = max(0, self.inside - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.body.append(data)
            if self.inside:
                self.main.append(data)

    def text(self):
        text = ''.join(self.main or self.body)
        return '\n'.join(' '.join(line.split()) for line in text.splitlines() if line.strip())


def select_excerpt(text, topic):
    if re.search(r'\breport\b', topic, re.I) and re.search(r'findings|statistics|trends', topic, re.I):
        for heading in ('year in review', 'key findings', 'key statistics', 'executive summary'):
            matches = list(re.finditer(r'^' + heading + r'\s*$', text, re.I | re.M))
            if matches and matches[-1].start() > 1200:
                start = matches[-1].start()
                return text[:1200] + '\n[Intervening text omitted]\n' + text[start:start + 5700]
    return text[:MAX_CHARS]


def refresh_sources(pool, topic):
    """Replace search snippets with page evidence when safe; never silently claim a fetch succeeded.

    At most two pages and two allowlisted redirects per page, with bounded size/time.
    Failure preserves the search evidence and is recorded, not retried indefinitely.
    """
    attempts = []
    candidates = [s for s in pool if safe_url(s['url']) and '.pdf' not in urlsplit(s['url']).path.lower()]
    # Product release notes take precedence over general marketing/product pages.
    candidates.sort(key=lambda s: 'release-notes' not in s['url'])
    if not candidates:
        return attempts
    with httpx.Client(timeout=httpx.Timeout(12, connect=5), follow_redirects=False) as client:
        for source in candidates[:2]:
            attempt = dict(source_id=source['id'], status='unavailable')
            attempts.append(attempt)
            url = source['url']
            try:
                for _ in range(3):
                    if not safe_url(url):
                        break
                    with client.stream('GET', url, headers={'User-Agent': 'AI-News-Research/1.0'}) as response:
                        attempt['http_status'] = response.status_code
                        if response.status_code in {301, 302, 303, 307, 308}:
                            url = urljoin(url, response.headers.get('location', ''))
                            continue
                        response.raise_for_status()
                        if 'text/html' not in response.headers.get('content-type', '').lower():
                            break
                        content = bytearray()
                        for chunk in response.iter_bytes():
                            content.extend(chunk)
                            if len(content) > MAX_BYTES:
                                raise ValueError('Page exceeds size limit')
                        parser = PageText()
                        parser.feed(content.decode('utf-8', errors='replace'))
                        text = parser.text()
                        if len(text) < 100:
                            break
                        source['excerpt'] = select_excerpt(text, topic)
                        source['retrieved_at'] = datetime.now(timezone.utc).isoformat()
                        source['evidence_origin'] = 'official_page'
                        attempt['status'] = 'retrieved'
                        attempt['characters'] = len(source['excerpt'])
                        break
            except (httpx.HTTPError, ValueError, UnicodeError):
                pass
    return attempts
