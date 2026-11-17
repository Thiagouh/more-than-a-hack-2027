"""Pinned source download and cache checks with synthetic in-memory responses."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.download_ball_quality_external import fetch_match


class ExternalDownloadTests(unittest.TestCase):
    def test_verified_download_cache_and_corruption_repair(self):
        tracking=b'{"frame":0}\n';digest=hashlib.sha256(tracking).hexdigest()
        pointer=f'version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(tracking)}\n'.encode()
        def response(url,**kwargs):
            if 'media.githubusercontent.com' in url:return io.BytesIO(tracking)
            if url.endswith('_match.json'):return io.BytesIO(json.dumps(dict(id=1874553)).encode())
            return io.BytesIO(pointer)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('urllib.request.urlopen',side_effect=response) as network:
                source=fetch_match('1874553',tmp)
                self.assertEqual(network.call_count,3)
                self.assertEqual(source['tracking_sha256'],digest)
            with patch('urllib.request.urlopen',side_effect=AssertionError('cache must avoid network')):
                self.assertEqual(fetch_match('1874553',tmp),source)
            (Path(tmp)/'1874553/tracking.jsonl').write_bytes(b'corrupt')
            with patch('urllib.request.urlopen',side_effect=response):fetch_match('1874553',tmp)
            self.assertEqual((Path(tmp)/'1874553/tracking.jsonl').read_bytes(),tracking)

    def test_unverified_bytes_are_not_promoted_to_final_tracking(self):
        pointer=b'version https://git-lfs.github.com/spec/v1\noid sha256:'+b'0'*64+b'\nsize 3\n'
        def response(url,**kwargs):
            if 'media.githubusercontent.com' in url:return io.BytesIO(b'bad')
            if url.endswith('_match.json'):return io.BytesIO(b'{"id":1874553}')
            return io.BytesIO(pointer)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('urllib.request.urlopen',side_effect=response):
                with self.assertRaisesRegex(ValueError,'differs from the source'):fetch_match('1874553',tmp)
            self.assertFalse((Path(tmp)/'1874553/tracking.jsonl').exists())
            self.assertFalse((Path(tmp)/'1874553/source.json').exists())


if __name__=='__main__':unittest.main()
