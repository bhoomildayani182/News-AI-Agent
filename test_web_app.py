"""Offline checks for the on-demand topic and email workflow."""

import contextlib
import io
import os
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode

from stages import ResearchError
from openrouter_client import RunLimitReached, OpenRouterError
from web_app import HOST, PORT, STATE, Handler, JobState, main, page


class WebAppTests(unittest.TestCase):
    def test_output_limit_is_an_actionable_notice_and_never_sends_email(self):
        error = RunLimitReached('Increase OPENROUTER_RECOVERY_MAX_TOKENS and restart.',
                                kind='output', setting='OPENROUTER_RECOVERY_MAX_TOKENS')
        error.run_summary = {'run_id': 'limit-test', 'api_calls': 4,
                             'reported_cost_usd': 0.006, 'cost_complete': True}
        state = JobState()
        with patch('web_app.get_settings'), patch('web_app.validate_email_settings'), \
             patch('web_app.research_topic', side_effect=error), patch('web_app.send_email') as send:
            state._run('Topic')
        _, status, message = state.snapshot()
        self.assertEqual(status, 'limit_reached')
        self.assertIn('Agent limit reached', message)
        self.assertIn('OPENROUTER_RECOVERY_MAX_TOKENS', message)
        self.assertNotIn('502', message)
        self.assertIn('notice limit_reached', page(status=status, message=message).decode())
        self.assertEqual(state.run_summary()['reported_cost_usd'], 0.006)
        send.assert_not_called()

    def test_provider_credit_and_rate_limits_do_not_advise_increasing_agent_budget(self):
        for code in (402, 429):
            state = JobState()
            with patch('web_app.get_settings'), patch('web_app.validate_email_settings'), \
                 patch('web_app.research_topic', side_effect=OpenRouterError(code)), \
                 patch('web_app.send_email') as send:
                state._run('Topic')
            self.assertEqual(state.snapshot()[1], 'limit_reached')
            self.assertIn('Changing the agent run budget does not fix provider limits', state.snapshot()[2])
            send.assert_not_called()

    def test_container_binds_all_interfaces_but_advertises_local_browser_url(self):
        output = io.StringIO()
        with patch.dict(os.environ, {'APP_BIND_HOST': '0.0.0.0'}), \
             patch('web_app.get_settings'), patch('web_app.validate_email_settings'), \
             patch('web_app.ThreadingHTTPServer') as server, contextlib.redirect_stdout(output):
            self.assertEqual(main(), 0)
        server.assert_called_once_with(('0.0.0.0', PORT), Handler)
        self.assertIn('http://127.0.0.1:8765', output.getvalue())

    def test_python_default_bind_remains_loopback(self):
        with patch.dict(os.environ, {'APP_BIND_HOST': ''}), \
             patch('web_app.get_settings'), patch('web_app.validate_email_settings'), \
             patch('web_app.ThreadingHTTPServer') as server, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        server.assert_called_once_with((HOST, PORT), Handler)

    def test_invalid_bind_setting_does_not_start_server(self):
        with patch.dict(os.environ, {'APP_BIND_HOST': 'unexpected-host'}), \
             patch('web_app.get_settings'), patch('web_app.validate_email_settings'), \
             patch('web_app.ThreadingHTTPServer') as server, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 1)
        server.assert_not_called()

    def test_container_bind_does_not_relax_browser_host_and_origin_checks(self):
        for headers in ({'Host': 'attacker.example', 'Origin': f'http://{HOST}:{PORT}'},
                        {'Host': f'{HOST}:{PORT}', 'Origin': 'http://attacker.example'},
                        {'Host': f'0.0.0.0:{PORT}', 'Origin': f'http://0.0.0.0:{PORT}'},
                        {'Host': f'localhost:{PORT}', 'Origin': 'null'},
                        {'Host': f'localhost:{PORT}', 'Origin': f'http://{HOST}:{PORT}'},
                        {'Host': f'localhost:{PORT}', 'Origin': 'http://localhost:9999'},
                        {'Host': f'localhost.attacker.example:{PORT}'},
                        {}):
            handler = Handler.__new__(Handler)
            handler.path = '/run'
            handler.headers = headers
            handler._respond = Mock()
            with patch.dict(os.environ, {'APP_BIND_HOST': '0.0.0.0'}), patch('web_app.STATE.start') as start:
                handler.do_POST()
            self.assertEqual(handler._respond.call_args.args[0], 403)
            start.assert_not_called()

    def test_local_form_submissions_work_with_both_addresses(self):
        for authority in (f'{HOST}:{PORT}', f'localhost:{PORT}'):
            for origin in (None, f'http://{authority}'):
                with self.subTest(authority=authority, origin=origin):
                    handler = self._form_handler(authority, origin, STATE.token)
                    with patch('web_app.STATE.start', return_value=True) as start:
                        handler.do_POST()
                    start.assert_called_once_with('Example news topic')
                    handler.send_response.assert_called_once_with(303)
                    handler._respond.assert_not_called()

    def test_localhost_still_requires_valid_form_token(self):
        handler = self._form_handler(f'localhost:{PORT}', f'http://localhost:{PORT}', 'stale-token')
        with patch('web_app.STATE.start') as start:
            handler.do_POST()
        self.assertEqual(handler._respond.call_args.args[0], 403)
        start.assert_not_called()

    @staticmethod
    def _form_handler(authority, origin, token):
        body = urlencode({'topic': 'Example news topic', 'token': token}).encode()
        handler = Handler.__new__(Handler)
        handler.path = '/run'
        handler.headers = {'Host': authority, 'Content-Length': str(len(body)),
                           'Content-Type': 'application/x-www-form-urlencoded'}
        if origin is not None:
            handler.headers['Origin'] = origin
        handler.rfile = io.BytesIO(body)
        handler._respond = Mock()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        return handler

    def test_success_researches_and_sends_one_email(self):
        state = JobState()
        with patch('web_app.get_settings') as settings, \
             patch('web_app.validate_email_settings'), \
             patch('web_app.research_topic', return_value='report') as research, \
             patch('web_app.send_email') as send:
            state._run('RBA decision')
        research.assert_called_once_with('RBA decision', settings.return_value)
        send.assert_called_once_with('RBA decision', 'report')
        self.assertEqual(state.snapshot()[1], 'sent')

    def test_failed_research_never_sends(self):
        state = JobState()
        with patch('web_app.get_settings'), patch('web_app.validate_email_settings'), \
             patch('web_app.research_topic', side_effect=ResearchError('Rejected')), \
             patch('web_app.send_email') as send:
            state._run('RBA decision')
        send.assert_not_called()
        self.assertEqual(state.snapshot()[1], 'failed')

    def test_running_job_rejects_second_submission(self):
        state = JobState()
        with state.lock:
            state.status = 'running'
        with patch('web_app.Thread') as worker:
            self.assertFalse(state.start('Second topic'))
        worker.assert_not_called()

    def test_page_escapes_topic_and_message(self):
        html = page('<script>alert(1)</script>', 'failed', '<unsafe>').decode()
        self.assertNotIn('<script>', html)
        self.assertNotIn('<unsafe>', html)
        self.assertIn('&lt;unsafe&gt;', html)

    def test_research_failure_keeps_cost_and_exact_reason_visible(self):
        error = ResearchError('Missing independent source evidence.')
        error.run_summary = {'run_id': 'test-run', 'api_calls': 2,
                             'reported_cost_usd': 0.002, 'cost_complete': True,
                             'elapsed_seconds': 3, 'search_requests': 1}
        state = JobState()
        with patch('web_app.get_settings'), patch('web_app.validate_email_settings'), \
             patch('web_app.research_topic', side_effect=error), \
             patch('web_app.send_email') as send:
            state._run('Topic')
        self.assertIn('Missing independent', state.snapshot()[2])
        self.assertIn('$0.002000', page(summary=state.run_summary()).decode())
        send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
