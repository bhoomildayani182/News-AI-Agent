"""Beginner-friendly CLI for fast or thorough news research."""

import argparse
import os
from pathlib import Path
import smtplib
import sys

import httpx
from dotenv import load_dotenv
from google.genai import errors

from config import ConfigurationError, get_settings
from demo import DEMO_TOPIC, demo_result
from email_delivery import (EmailConfigurationError, build_email, send_email,
                            validate_email_settings)
from stages import ResearchError, format_report, research_topic
from openrouter_client import OpenRouterError
from run_tracking import RunStore, record_delivery, usage_text


def get_topic() -> str:
    """Keep asking until the user provides a topic."""
    while True:
        topic = input('Enter news topic: ').strip()
        if topic:
            return topic
        print('Please enter a news topic.')


def _print_api_error(error: errors.APIError) -> None:
    """Never print raw exception details, which may include credentials."""
    code = getattr(error, 'code', None)
    messages = {
        400: 'Request rejected. Check your key and model support for Google Search.',
        401: 'Check GEMINI_API_KEY in .env.',
        403: 'Access denied. Check your Google project, credentials and model access.',
        404: 'Model unavailable. Check GEMINI_MODEL in .env.',
        429: 'Quota or rate limit reached. Check usage or wait before retrying.',
    }
    message = messages.get(code, 'Gemini request failed. Please try again later.')
    if isinstance(code, int) and code >= 500:
        message = 'Gemini is temporarily unavailable or busy. Please try again later.'
    if isinstance(code, int):
        message = f'HTTP {code}: {message}'
    print(f'Error: {message}', file=sys.stderr)


def main() -> int:
    delivery_started = False
    try:
        # Demo previews can use the email name/title without requiring an API key.
        load_dotenv(Path(__file__).resolve().parent / '.env', override=False)
        parser = argparse.ArgumentParser(description='Research a news topic with Gemini or OpenRouter.')
        parser.add_argument('--thorough', action='store_true',
                            help='Gemini: six stages. OpenRouter always uses its bounded research loop.')
        parser.add_argument('--fresh', action='store_true',
                            help='OpenRouter: bypass the recent verified-report cache.')
        parser.add_argument('--history', action='store_true',
                            help='Show the five most recent local OpenRouter runs without an API call.')
        parser.add_argument('--demo', action='store_true',
                            help='Show a fictional sample locally; makes zero API calls.')
        email_action = parser.add_mutually_exclusive_group()
        email_action.add_argument('--preview-email', action='store_true',
                            help='Print the email subject and body; does not send it.')
        email_action.add_argument('--send-email', action='store_true',
                            help='Send a successful live report using SMTP settings in .env.')
        args = parser.parse_args()
        if args.history:
            if any((args.send_email, args.preview_email, args.demo, args.fresh, args.thorough)):
                parser.error('--history must be used on its own.')
            history = RunStore().recent()
            for item in history:
                print(usage_text(item))
                print(f"  {item['status']} at {item['stage']}: {item.get('error') or 'OK'}")
                if item.get('delivery_status'):
                    print('  Email: ' + item['delivery_status'])
            if not history:
                print('No local OpenRouter runs yet.')
            return 0
        if args.demo and args.send_email:
            parser.error('--demo cannot be combined with --send-email.')
        if args.demo:
            topic = DEMO_TOPIC
            print('OFFLINE DEMO — fictional sample; no API request or current news research.\n')
            result = demo_result()
        else:
            settings = get_settings()
            if args.send_email:
                validate_email_settings()  # Check before paid API requests.
            topic = get_topic()
            mode = 'bounded' if settings.provider == 'openrouter' else ('six-stage' if args.thorough else 'fast')
            print(f'\nStarting {mode} research...\n')
            if args.fresh:
                result = research_topic(topic, settings, thorough=args.thorough, fresh=True)
            else:
                result = research_topic(topic, settings, thorough=args.thorough)
        if args.preview_email:
            sender = os.getenv('EMAIL_FROM', '').strip() or 'sender@example.com'
            recipient = os.getenv('EMAIL_TO', '').strip() or 'recipient@example.com'
            message = build_email(topic, result, sender, recipient)
            print('\nEMAIL PREVIEW — not sent\n')
            print(f'From: {message["From"]}')
            print(f'To: {message["To"]}')
            print(f'Subject: {message["Subject"]}\n')
            print(message.get_body(preferencelist=('plain',)).get_content())
            preview_path = Path(__file__).resolve().parent / 'email-preview.html'
            preview_path.write_text(message.get_body(preferencelist=('html',)).get_content(), encoding='utf-8')
            print(f'\nFormatted email preview: {preview_path}')
        else:
            print('\n' + format_report(result))
        if args.send_email:
            delivery_started = True
            record_delivery(result.run_summary, 'sending')
            send_email(topic, result)
            record_delivery(result.run_summary, 'accepted_by_smtp')
            print('\nEmail sent successfully.')
        return 0
    except (ConfigurationError, ResearchError, EmailConfigurationError) as error:
        print(f'Error: {error}', file=sys.stderr)
    except (smtplib.SMTPException, OSError):
        if delivery_started:
            record_delivery(result.run_summary, 'unconfirmed_check_inbox_before_retry')
        message = ('Email delivery could not be confirmed. Check SMTP settings and inbox before sending again.' if delivery_started
                   else 'A local file or connection operation failed. Check network and project folder permissions.')
        print(message, file=sys.stderr)
    except errors.APIError as error:
        _print_api_error(error)
    except OpenRouterError as error:
        print(f'Error: HTTP {error.code}: {error}', file=sys.stderr)
    except httpx.TimeoutException:
        print('Request timed out. Please try again.', file=sys.stderr)
    except httpx.TransportError:
        print('Connection failed. Check your internet connection or proxy.', file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print('\nCancelled.', file=sys.stderr)
    except Exception as error:
        print(f'Unexpected error: {type(error).__name__}. Check your setup.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
