"""Offline checks for demo output and email; no API or SMTP network calls."""

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from demo import DEMO_TOPIC, demo_result
from email_delivery import EmailConfigurationError, build_email, send_email
from main import main
from stages import ResearchError, ResearchResult, Source
from email_format import html_email


class DeliveryTests(unittest.TestCase):
    def test_preview_and_send_cannot_be_combined(self):
        with patch('sys.argv', ['main.py', '--preview-email', '--send-email']), \
             patch('main.research_topic') as research, \
             patch('main.send_email') as send, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        research.assert_not_called()
        send.assert_not_called()

    def test_failed_research_never_sends_email(self):
        with patch('sys.argv', ['main.py', '--send-email']), \
             patch('main.get_settings'), patch('main.validate_email_settings'), \
             patch('main.get_topic', return_value='Example topic'), \
             patch('main.research_topic', side_effect=ResearchError('Audit failed')), \
             patch('main.send_email') as send, \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(), 1)
        send.assert_not_called()

    def test_demo_preview_without_key_or_network(self):
        output = io.StringIO()
        with patch.dict(os.environ, {'GEMINI_API_KEY': ''}), \
             patch('main.research_topic', side_effect=AssertionError('API called')), \
             patch('main.send_email', side_effect=AssertionError('SMTP called')), \
             patch('main.Path.write_text'), \
             patch('sys.argv', ['main.py', '--demo', '--preview-email']), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertIn('OFFLINE DEMO', output.getvalue())
        self.assertIn('EMAIL PREVIEW', output.getvalue())
        self.assertIn('fictional', output.getvalue().lower())
        self.assertEqual(output.getvalue().count('\nSummary\n'), 1)
        self.assertNotIn('## ', output.getvalue())

    def test_email_subject_and_sources(self):
        with patch.dict(os.environ, {'EMAIL_SENDER_NAME': 'Research Desk',
                                     'EMAIL_SUBJECT_TITLE': 'Daily Briefing'}):
            msg = build_email(DEMO_TOPIC, demo_result(), 'sender@example.com', 'to@example.com')
        self.assertTrue(msg['Subject'].startswith('Daily Briefing: '))
        self.assertEqual(str(msg['From']), 'Research Desk <sender@example.com>')
        self.assertIn('Example news topic', msg['Subject'])
        plain = msg.get_body(preferencelist=('plain',)).get_content()
        html = msg.get_body(preferencelist=('html',)).get_content()
        self.assertIn('Summary', plain)
        self.assertNotIn('## ', plain)
        self.assertIn('https://example.org/official', plain)
        self.assertIn('href="https://example.org/official"', html)
        self.assertIn('<h2 ', html)
        self.assertEqual(msg.get_content_type(), 'multipart/alternative')

    def test_html_escapes_untrusted_text_and_rejects_unsafe_links(self):
        result = ResearchResult('## Summary\n<script>alert(1)</script> [S1]',
                                [Source('<img src=x>', 'https://example.org/?a=1&b=2')])
        html = html_email('<unsafe>', result, 'News')
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img ', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('a=1&amp;b=2', html)
        unsafe = ResearchResult('## Summary\nClaim [S1]', [Source('Unsafe', 'javascript:alert(1)')])
        with self.assertRaises(ValueError):
            html_email('Topic', unsafe, 'News')

    def test_email_name_and_title_reject_line_breaks(self):
        with patch.dict(os.environ, {'EMAIL_SENDER_NAME': 'Name\nInjected'}):
            with self.assertRaises(EmailConfigurationError):
                build_email(DEMO_TOPIC, demo_result(), 'sender@example.com', 'to@example.com')

    def test_send_requires_settings_before_connection(self):
        with patch.dict(os.environ, {'SMTP_HOST': '', 'SMTP_USERNAME': '',
                                     'SMTP_PASSWORD': '', 'EMAIL_FROM': '', 'EMAIL_TO': ''}):
            with self.assertRaises(EmailConfigurationError):
                send_email(DEMO_TOPIC, demo_result())

    def test_smtp_ssl_sends_once(self):
        env = {'SMTP_HOST': 'mail.example.com', 'SMTP_PORT': '465',
               'SMTP_USERNAME': 'sender@example.com', 'SMTP_PASSWORD': 'dummy',
               'EMAIL_FROM': 'sender@example.com', 'EMAIL_TO': 'to@example.com'}
        with patch.dict(os.environ, env), patch('email_delivery.smtplib.SMTP_SSL') as smtp:
            send_email(DEMO_TOPIC, demo_result())
            connection = smtp.return_value.__enter__.return_value
            connection.login.assert_called_once()
            connection.send_message.assert_called_once()


if __name__ == '__main__':
    unittest.main()
