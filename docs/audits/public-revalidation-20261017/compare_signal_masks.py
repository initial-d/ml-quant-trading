"""Offline same-input comparison; synthetic engineering evidence only.

Run from the repository root:
    .venv/bin/python docs/audits/public-revalidation-20261017/compare_signal_masks.py
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
import scripts.public_data_validation as corrected

BASE = 'ccdcf16a7c0768dbb31fa22f91ae02b692f0fd9b'
OUT = ROOT / 'artifacts/revalidation-20261017'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
source = subprocess.check_output(
    ['git', 'show', f'{BASE}:scripts/public_data_validation.py'], cwd=ROOT, text=True,
)
old = types.ModuleType('archived_validation')
old.__file__ = str(ROOT / 'scripts/public_data_validation.py')
sys.modules[old.__name__] = old
exec(compile(source, old.__file__, 'exec'), old.__dict__)

# A stock is observed today but is halted tomorrow. Zero-filled data on the
# halt day mirrors loader behavior; both versions receive this identical panel.
panel = make_synthetic_panel(SyntheticConfig(n_dates=90, n_stocks=12, halt_prob=0, seed=3))
mask = panel.mask.clone()
mask[31, 0] = False
fields = {name: getattr(panel, name).masked_fill(~mask, 0)
          for name in ('open', 'high', 'low', 'close', 'volume', 'vwap', 'amount')}
panel = replace(panel, mask=mask, **fields)
panel.assert_consistent()
raw_mask = panel.mask.clone()
hash_state = hashlib.sha256()
for name in ('open', 'high', 'low', 'close', 'volume', 'vwap', 'amount', 'mask'):
    hash_state.update(name.encode())
    hash_state.update(getattr(panel, name).numpy().tobytes())

result = {
    'category': 'synthetic engineering comparison; no predictive or investment evidence',
    'base_commit': BASE,
    'corrected_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'synthetic_config': {'n_dates':90, 'n_stocks':12, 'seed':3, 'halt_prob':0},
    'perturbation': 'zero-filled halt at date index 31, stock index 0',
    'panel_hash_method': 'SHA-256 over field names and C-order tensor bytes for open/high/low/close/volume/vwap/amount/mask',
    'panel_sha256': hash_state.hexdigest(),
    'threads': {'torch':1,'interop':1},
    'costs_bps':5, 'slippage_bps':2,
    'versions': {},
}
for label, module in (('old', old), ('corrected', corrected)):
    module.load_validation_panel = lambda cfg: panel
    captured = {}
    summarise = module._summarise_strategy

    def capture(name, weights, returns, _captured=captured, _summarise=summarise, **kwargs):
        _captured[name] = weights.copy()
        return _summarise(name, weights, returns, **kwargs)

    module._summarise_strategy = capture
    cfg = module.ValidationConfig(
        source='synthetic', preset='us-large-100', tickers=(), start='2021-01-01',
        end='2022-01-01', max_tickers=12, device='cpu', costs_bps=5, slippage_bps=2,
        cost_grid_bps=(0,7,15,30), bootstrap_samples=100, bootstrap_block_size=20,
        train_window=40, test_window=20, step=20, top_quantile=0.25, seed=3,
        epochs=1, batch_size=128, hidden=16,
        models=('equal_weight','momentum_20','alpha101_mean'),
        output_dir=OUT / label, synthetic_dates=90, synthetic_stocks=12,
        command=(sys.executable, str(Path(__file__).relative_to(ROOT))),
    )
    rows = module.run_validation(cfg)
    result['versions'][label] = {
        'equal_weight_before_halt': float(captured['equal_weight'][30,0]),
        'equal_weight_on_halt': float(captured['equal_weight'][31,0]),
        'results': rows,
    }
    module._summarise_strategy = summarise
    assert torch.equal(panel.mask,raw_mask)
assert result['versions']['old']['equal_weight_before_halt'] == 0
assert result['versions']['corrected']['equal_weight_before_halt'] > 0
assert result['versions']['corrected']['equal_weight_on_halt'] == 0
(OUT / 'same-input-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({version:{key:row[key] for key in ('equal_weight_before_halt','equal_weight_on_halt')}
                  for version,row in result['versions'].items()},indent=2))
