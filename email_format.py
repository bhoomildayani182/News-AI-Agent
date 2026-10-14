"""Small, escaped HTML renderer for the agent's fixed report format."""

from html import escape
import re
from urllib.parse import urlsplit

from stages import ResearchResult, SOURCE_CITATION, cited_source_numbers, format_report


def plain_email(result: ResearchResult) -> str:
    """Readable fallback for email clients that cannot display HTML."""
    text = format_report(result)
    text = re.sub(r'^#{1,2} ', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    return re.sub(r'^\s*\*\s+', '- ', text, flags=re.MULTILINE)


def html_email(topic: str, result: ResearchResult, title: str) -> str:
    """Render only our supported Markdown; never trust model-generated HTML."""
    format_report(result)  # Check that every citation has a source before rendering.
    cited = cited_source_numbers(result.report_text)
    urls = {}
    for number in cited:
        url = result.sources[number - 1].url
        parsed = urlsplit(url)
        if parsed.scheme not in {'https', 'http'} or not parsed.netloc:
            raise ValueError('Source links must be HTTP or HTTPS URLs.')
        urls[number] = escape(url, quote=True)

    def inline(text: str) -> str:
        text = escape(text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        def citation(match):
            numbers = [int(n) for n in re.findall(r'S(\d+)', match.group())]
            return ' ' + ' '.join(
                f'<a href="{urls[n]}" style="color:#2563eb;text-decoration:none;'
                f'font-size:12px;white-space:nowrap">[{n}]</a>' for n in numbers)
        return SOURCE_CITATION.sub(citation, text)

    def body_html(body: str) -> str:
        parts, paragraph, in_list = [], [], False
        def flush():
            if paragraph:
                parts.append('<p style="margin:0 0 12px">' + inline(' '.join(paragraph)) + '</p>')
                paragraph.clear()
        for line in body.strip().splitlines():
            item = re.match(r'^\s*[-*]\s+(.+)', line)
            if item:
                flush()
                if not in_list:
                    parts.append('<ul style="margin:0 0 12px;padding-left:22px">')
                    in_list = True
                parts.append('<li style="margin:0 0 8px">' + inline(item[1]) + '</li>')
            else:
                if in_list:
                    parts.append('</ul>')
                    in_list = False
                if line.strip():
                    paragraph.append(line.strip())
                else:
                    flush()
        flush()
        if in_list:
            parts.append('</ul>')
        return ''.join(parts)

    chunks = re.split(r'^#{1,2} (.+?)\s*$', result.report_text, flags=re.MULTILINE)
    sections = []
    for heading, body in zip(chunks[1::2], chunks[2::2]):
        highlight = 'background:#eff6ff;border-radius:10px;padding:20px;' if heading == 'Summary' else ''
        sections.append(f'<tr><td style="padding:0 28px 20px"><div style="{highlight}">'
                        f'<h2 style="font-size:17px;color:#0f172a;margin:0 0 10px">{escape(heading)}</h2>'
                        f'{body_html(body)}</div></td></tr>')
    sources = ''.join(
        f'<p style="margin:0 0 7px"><a href="{urls[n]}" '
        f'style="color:#2563eb;text-decoration:underline">[{n}] '
        f'{escape(result.sources[n - 1].title)}</a>'
        + (f' <span style="color:#64748b">({escape(result.sources[n - 1].published_at)})</span>'
           if result.sources[n - 1].published_at else '') + '</p>' for n in sorted(cited))
    generated = result.run_summary.get('report_generated_at', '')
    researched = (f'<p style="font-size:12px;color:#64748b">Researched: {escape(generated)}</p>'
                  if generated else '')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(title)}</title></head>'
            '<body style="margin:0;background:#f1f5f9;color:#334155;font-family:Arial,sans-serif;'
            'font-size:15px;line-height:1.6">'
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            'style="background:#f1f5f9"><tr><td align="center" style="padding:24px 8px">'
            '<table role="presentation" width="640" cellpadding="0" cellspacing="0" '
            'style="width:100%;max-width:640px;background:#ffffff;border:1px solid #e2e8f0">'
            '<tr><td style="background:#0f172a;padding:24px 28px;color:#ffffff">'
            f'<p style="margin:0;font-size:12px;letter-spacing:2px;text-transform:uppercase">{escape(title)}</p>'
            '<p style="margin:6px 0 0;color:#cbd5e1;font-size:14px">Your news brief</p></td></tr>'
            f'<tr><td style="padding:24px 28px"><h1 style="margin:0;font-size:22px;line-height:1.35;'
            f'color:#0f172a">{escape(topic)}</h1></td></tr>' + ''.join(sections) +
            '<tr><td style="padding:20px 28px;background:#f8fafc;border-top:1px solid #e2e8f0">'
            '<h2 style="font-size:15px;margin:0 0 10px;color:#0f172a">Read the sources</h2>' + sources +
            '<p style="font-size:12px;color:#64748b;margin:16px 0 0">AI-assisted research. '
            'Source links let you check the original reporting.</p>' + researched +
            '</td></tr></table></td></tr></table></body></html>')
