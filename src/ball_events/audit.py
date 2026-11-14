"""Attempt lifecycle starts before semantic validation; operational timestamps stay separate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import uuid


class Attempt:
    def __init__(self, out, command=None, hypothesis=None):
        self.out = Path(out)
        self.path = self.out.parent / '_experiments' / (uuid.uuid4().hex + '.json')
        # Do not serialize arbitrary API inputs before validating their shape.
        self.record = {'status': 'started', 'stage': 'validation', 'cache_status': 'disabled',
                       'command': command if isinstance(command, (list, tuple)) and all(isinstance(x, str) for x in command) else None,
                       'hypothesis': hypothesis if isinstance(hypothesis, str) else None,
                       'identity': None, 'started_at_utc': datetime.now(timezone.utc).isoformat()}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.record, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        os.replace(tmp, self.path)

    def update(self, **fields):
        self.record.update(fields)
        self.save()

    def __enter__(self):
        self.save()
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            stage = self.record['stage']
            status = 'rejected' if stage == 'validation' else 'failed_publication' if stage == 'publication' else 'failed_processing'
            self.record.update(status=status, error_type=kind.__name__, error_code=f'{stage}_error')
        else:
            self.record['status'] = 'completed'
        self.record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        try:
            self.save()
        except OSError:
            # A post-publication audit fault must not turn a valid output into a false failure.
            if error is not None or self.record['stage'] != 'published':
                raise
            import warnings
            warnings.warn('Output completed; final audit update failed', RuntimeWarning)
        return False
