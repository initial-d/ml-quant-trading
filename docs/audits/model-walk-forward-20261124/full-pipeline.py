import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from click.testing import CliRunner
import yaml
from mlquant.cli.main import cli

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
root = Path('/tmp/mlquant-walk-forward-20261124/full')
root.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load(Path('configs/small.yaml').read_text())
config['artifacts_dir'] = str(root)
config['train']['save_dir'] = str(root / 'static-checkpoints')
config_path = root / 'config.yaml'
config_path.write_text(yaml.safe_dump(config))
stages = [ ['gen-data'], ['features'], ['train', '--walk-forward', '--test-dates', '50'],
           ['portfolio'], ['backtest', '--accounting', 'constrained'],
           ['rebalance', '--currency', 'CNY', '--data-category', 'synthetic', '--data-source', 'synthetic-demo'] ]
records = []
runner = CliRunner()
for stage in stages:
    args = [*stage, '--config', str(config_path)]
    start = time.perf_counter()
    result = runner.invoke(cli, args)
    records.append({'args': args, 'elapsed_seconds': time.perf_counter() - start, 'exit_code': result.exit_code})
    print(' '.join(args), flush=True)
    print(result.output, flush=True)
    if result.exit_code:
        if result.exception:
            raise result.exception
        raise RuntimeError(result.output)
manifest = {'category': 'synthetic engineering full-213-factor pipeline',
            'command': f'{sys.executable} /tmp/mlquant-walk-forward-20261124/full-pipeline.py',
            'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'config': config, 'stages': records,
            'training': json.loads((root / 'walk-forward.json').read_text()),
            'constrained_backtest': json.loads((root / 'summary.json').read_text()),
            'cost_aware_backtest': json.loads((root / 'rebalance/summary.json').read_text())}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
