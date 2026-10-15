"""Offline checks for bounded, allowlisted original-page retrieval."""

import unittest
from unittest.mock import Mock, patch

import httpx

from official_sources import MAX_BYTES, PageText, refresh_sources, safe_url, select_excerpt, latest_rba_statement
from research_agent import publisher


def source(url='https://docs.cloud.google.com/run/docs/release-notes', number=1):
    return dict(id=f'S{number}', url=url, excerpt='Old search evidence is preserved on fetch failure.')


def response(body, status=200, **headers):
    result = Mock(status_code=status)
    result.headers = {'content-type': 'text/html; charset=utf-8', **headers}
    result.iter_bytes.return_value = iter([body.encode() if isinstance(body, str) else body])
    context = Mock()
    context.__enter__ = Mock(return_value=result)
    context.__exit__ = Mock(return_value=False)
    return context


class OfficialSourceTests(unittest.TestCase):
    def test_latest_rba_discovery_uses_live_index_and_actual_statement_text(self):
        with patch('official_sources.httpx.Client') as constructor:
            client = constructor.return_value.__enter__.return_value
            client.stream.side_effect = [
                response('<a href="/media-releases/2026/mr-26-27.html">29 September 2026</a>'),
                response('<main><h1>Monetary Policy Decision</h1><p>' +
                         'The Reserve Bank announced its decision and explained its rationale. ' * 5 + '</p></main>')]
            pool = latest_rba_statement('Compare ABC and Reuters on the latest RBA decision')
        self.assertEqual(len(pool), 1)
        self.assertTrue(pool[0]['url'].endswith('mr-26-27.html'))
        self.assertEqual(pool[0]['evidence_origin'], 'official_page')
        self.assertEqual(pool[0]['event_date'], '2026-09-29')
        self.assertIn('explained its rationale', pool[0]['excerpt'])

    def test_official_discovery_failure_does_not_create_placeholder_evidence(self):
        with patch('official_sources.httpx.Client') as constructor:
            client = constructor.return_value.__enter__.return_value
            client.stream.side_effect = httpx.ConnectError('Offline')
            self.assertEqual(latest_rba_statement('Latest RBA decision'), [])
        with patch('official_sources.httpx.Client') as constructor:
            self.assertEqual(latest_rba_statement('Latest Australian GDP release'), [])
            constructor.assert_not_called()

    def test_blocked_statement_url_is_only_a_search_target_not_evidence(self):
        with patch('official_sources.httpx.Client') as constructor, \
             patch('official_sources.refresh_sources', return_value=[]):
            client = constructor.return_value.__enter__.return_value
            client.stream.return_value = response(
                '<a href="/media-releases/2026/mr-26-27.html">Latest</a>')
            discovered = latest_rba_statement('Latest RBA decision', include_discovery=True)
        self.assertEqual(discovered[0]['excerpt'], '')
        self.assertNotIn('evidence_origin', discovered[0])

    def test_only_explicit_https_hosts_are_fetchable(self):
        self.assertTrue(safe_url('https://docs.cloud.google.com/run/docs/release-notes'))
        for url in ('http://docs.cloud.google.com/a', 'https://docs.cloud.google.com.evil.example/a',
                    'https://user:pass@docs.cloud.google.com/a', 'https://docs.cloud.google.com:444/a',
                    'https://127.0.0.1/a', 'https://169.254.169.254/a', 'https://unknown.gov.au/a',
                    'file:///etc/passwd', 'https://[invalid'):
            self.assertFalse(safe_url(url), url)

    def test_current_official_page_replaces_stale_search_excerpt(self):
        pool = [source()]
        html = ('<nav>Ignore navigation</nav><main><h1>Cloud Run release notes</h1>'
                '<h2>October 12, 2026</h2><p>A newer product feature is generally available.</p>'
                '<h2>October 7, 2026</h2><p>An older feature remains in preview.</p>'
                '<script>Ignore executable text</script></main><footer>Ignore footer</footer>')
        with patch('official_sources.httpx.Client') as constructor:
            client = constructor.return_value.__enter__.return_value
            client.stream.return_value = response(html)
            attempts = refresh_sources(pool, 'Latest Cloud Run announcement')
        self.assertIn('October 12, 2026', pool[0]['excerpt'])
        self.assertNotIn('navigation', pool[0]['excerpt'])
        self.assertNotIn('executable', pool[0]['excerpt'])
        self.assertNotIn('footer', pool[0]['excerpt'])
        self.assertEqual(pool[0]['id'], 'S1')
        self.assertEqual(pool[0]['evidence_origin'], 'official_page')
        self.assertEqual(attempts[0]['status'], 'retrieved')

    def test_redirect_to_unapproved_host_is_not_followed(self):
        pool = [source()]
        with patch('official_sources.httpx.Client') as constructor:
            client = constructor.return_value.__enter__.return_value
            client.stream.return_value = response('', 302, location='http://169.254.169.254/secret')
            result = refresh_sources(pool, 'Latest update')
        self.assertEqual(client.stream.call_count, 1)
        self.assertEqual(result[0]['status'], 'unavailable')
        self.assertNotIn('evidence_origin', pool[0])

    def test_network_and_oversized_pages_preserve_search_evidence(self):
        for failure in (True, False):
            pool = [source()]
            old = pool[0]['excerpt']
            with patch('official_sources.httpx.Client') as constructor:
                client = constructor.return_value.__enter__.return_value
                if failure:
                    client.stream.side_effect = httpx.ConnectError('Offline')
                else:
                    client.stream.return_value = response(b'x' * (MAX_BYTES + 1))
                result = refresh_sources(pool, 'Topic')
            self.assertEqual(pool[0]['excerpt'], old)
            self.assertEqual(result[0]['status'], 'unavailable')

    def test_two_page_cap_and_pdf_exclusion(self):
        pool = [source('https://www.cyber.gov.au/report.pdf')]
        pool += [source(f'https://docs.cloud.google.com/run/page{i}', i) for i in range(1, 5)]
        with patch('official_sources.httpx.Client') as constructor:
            client = constructor.return_value.__enter__.return_value
            client.stream.side_effect = httpx.ConnectError('Offline')
            result = refresh_sources(pool, 'Topic')
        self.assertEqual(client.stream.call_count, 2)
        self.assertEqual(len(result), 2)

    def test_report_focus_keeps_period_and_key_statistics(self):
        text = 'Report 2024-25\n' + 'Foreword material. ' * 150
        text += '\nYear in review\nOver 1,200 incidents, up 11%.\n'
        excerpt = select_excerpt(text, 'Main findings of the latest report')
        self.assertIn('Report 2024-25', excerpt)
        self.assertIn('1,200 incidents', excerpt)
        self.assertLessEqual(len(excerpt), 7000)

    def test_regional_google_docs_are_not_two_independent_publishers(self):
        self.assertEqual(publisher('https://docs.cloud.google.cn/release-notes'),
                         publisher('https://docs.cloud.google.com/run/docs/release-notes'))


if __name__ == '__main__':
    unittest.main()
