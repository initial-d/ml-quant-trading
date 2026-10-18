"""Two single-attempt AkShare probes; only run during explicit data validation.

This is the source of the October 17 stdin probe. It does not fall back to
synthetic or another provider. Downloaded prices, if any, stay in artifacts/.
"""
import datetime
import json
from pathlib import Path
import time
import traceback

import torch
from mlquant.data.akshare_loader import load_akshare_panel

base = Path('artifacts/revalidation-20261018')
base.mkdir(parents=True, exist_ok=True)
rows = []
for ticker in ('600519', '000001'):
    started = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    print('probe', ticker, started, flush=True)
    timer = time.perf_counter()
    row = {'ticker': ticker, 'started_at': started, 'retries': 1,
           'start': '2021-01-01', 'end': '2025-01-01', 'adjust': 'qfq'}
    try:
        panel = load_akshare_panel((ticker,), '2021-01-01', '2025-01-01',
                                  device='cpu', retries=1)
        panel.assert_consistent()
        row.update(status='success', dates=panel.n_dates,
                   tradable_ratio=float(panel.mask.float().mean()))
        torch.save(panel, base / f'probe-{ticker}.pt')
    except Exception as exc:
        row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    row['wall_seconds'] = time.perf_counter() - timer
    rows.append(row)
    (base / 'provider-probe.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    print(row, flush=True)
