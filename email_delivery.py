"""Build and optionally send a news report email using the Python standard library."""

from email.message import EmailMessage
from email.utils import formataddr, parseaddr
import os
from pathlib import Path
import smtplib
import ssl
from textwrap import shorten

from dotenv import load_dotenv
from stages import ResearchResult
from email_format import html_email, plain_email


class EmailConfigurationError(ValueError):
    """Raised for missing or invalid SMTP configuration."""


def _load_email_env() -> None:
    """Make demo email previews use the same private settings as live runs."""
    load_dotenv(Path(__file__).resolve().parent / '.env', override=False)


def build_email(topic: str, result: ResearchResult, sender: str, recipient: str) -> EmailMessage:
    """Build an HTML email with a readable plain-text fallback."""
    _load_email_env()
    clean_topic = shorten(' '.join(topic.split()), width=90, placeholder='…') or 'News report'
    sender_name = os.getenv('EMAIL_SENDER_NAME', '').strip()
    subject_title = os.getenv('EMAIL_SUBJECT_TITLE', 'News').strip() or 'News'
    if any(c in sender_name + subject_title for c in '\r\n'):
        raise EmailConfigurationError('Email name and title must be single-line text.')
    message = EmailMessage()
    prefix = '[Partial research] ' if result.run_summary.get('partial') else ''
    message['Subject'] = f'{prefix}{subject_title}: {clean_topic}'
    message['From'] = formataddr((sender_name, sender)) if sender_name else sender
    message['To'] = recipient
    message.set_content(plain_email(result))
    message.add_alternative(html_email(topic, result, subject_title), subtype='html')
    return message


def validate_email_settings() -> tuple[str, int, str, str, str, str]:
    """Fail before paid research if email settings are incomplete."""
    _load_email_env()
    host = os.getenv('SMTP_HOST', '').strip()
    username = os.getenv('SMTP_USERNAME', '').strip()
    password = os.getenv('SMTP_PASSWORD', '')
    sender = os.getenv('EMAIL_FROM', '').strip()
    recipient = os.getenv('EMAIL_TO', '').strip()
    port_text = os.getenv('SMTP_PORT', '465').strip()

    if not all((host, username, password, sender, recipient)):
        raise EmailConfigurationError(
            'Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM and EMAIL_TO in .env.'
        )
    if parseaddr(sender)[1] != sender:
        raise EmailConfigurationError('EMAIL_FROM must contain one plain email address.')
    if parseaddr(recipient)[1] != recipient:
        raise EmailConfigurationError('EMAIL_TO must contain one plain email address.')
    try:
        port = int(port_text)
    except ValueError:
        raise EmailConfigurationError('SMTP_PORT must be 465 or 587.') from None
    if port not in (465, 587):
        raise EmailConfigurationError('SMTP_PORT must be 465 or 587.')

    return host, port, username, password, sender, recipient


def send_email(topic: str, result: ResearchResult) -> None:
    """Send once via authenticated SMTP after an explicit user request."""
    host, port, username, password, sender, recipient = validate_email_settings()
    message = build_email(topic, result, sender, recipient)
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=20, context=context) as server:
            server.login(username, password)
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls(context=context)
            server.login(username, password)
            server.send_message(message)
