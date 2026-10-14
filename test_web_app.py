"""Offline checks for the on-demand topic and email workflow."""

import unittest
from unittest.mock import patch

from stages import ResearchError
from web_app import JobState, page


class WebAppTests(unittest.TestCase):
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
