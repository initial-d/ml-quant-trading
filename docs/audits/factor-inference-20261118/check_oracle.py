"""Independent Statsmodels checks on all frozen IC/RankIC series and test families.

Run after producing an inference JSON. No downloads or factor recomputation.
Statsmodels is an audit-only dependency, not required by mlquantx.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import sys

import numpy as np
import scipy
from scipy.stats import norm
import statsmodels
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac_simple


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    payload = gzip.decompress(raw) if args.source.suffix == '.gz' else raw
    source = json.loads(payload)
    report = json.loads(args.inference.read_text())
    if report['input_report_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('frozen source changed')
    errors = {'mean': [], 'standard_error': [], 'p_value': []}
    family = []
    for original, row in zip(source['factors'], report['factors']):
        if original['name'] != row['name']:
            raise ValueError('factor family changed')
        for metric in ('ic', 'rank_ic'):
            cell = row[metric]; family.append(cell['p_value'])
            values = np.asarray(original['daily_' + metric], dtype=float)
            observed = np.isfinite(values); n = int(observed.sum())
            if cell['status'] != 'ok':
                continue
            # Zero design rows are missing-date placeholders, not zero ICs.
            # X'X=n; scores=X*(y-X*beta) retain the complete original time grid.
            model = OLS(np.where(observed, values, 0), observed.astype(float)[:, None]).fit()
            # Statsmodels' built-in correction uses total design rows. Apply
            # the declared observed-date correction n/(n-1) explicitly instead.
            covariance = cov_hac_simple(model, nlags=cell['max_lag'], use_correction=False)
            se = math.sqrt(float(covariance[0, 0]) * n / (n - 1))
            mean = float(model.params[0])
            p = float(2 * norm.sf(abs(mean / se)))
            for key, value, expected in [('mean', mean, cell['mean']),
                                        ('standard_error', se, cell['hac_standard_error']),
                                        ('p_value', p, cell['p_value'])]:
                errors[key].append(abs(value - expected))
                if not np.isclose(value, expected, atol=1e-13, rtol=1e-11):
                    raise ValueError('Statsmodels mismatch: ' + original['name'] + ' ' + metric + ' ' + key)
    if len(family) != report['multiple_testing']['family_size']:
        raise ValueError('incomplete family')
    method_map = {'by': 'fdr_by', 'bh': 'fdr_bh', 'holm': 'holm'}
    p = np.asarray(family, dtype=float)
    valid = np.isfinite(p)
    filled = np.where(valid, p, 1.)
    adjusted = multipletests(filled, method=method_map[report['multiple_testing']['method']])[1]
    actual = np.array([row[k]['adjusted_p_value'] for row in report['factors'] for k in ('ic', 'rank_ic')], dtype=float)
    np.testing.assert_allclose(actual[valid], adjusted[valid], rtol=1e-12, atol=1e-14)
    # Nontrivial controls include tied, zero, unit and missing probabilities.
    from mlquant.backtest.ic_inference import adjust_pvalues
    control = np.array([.01, .04, .03, .002, np.nan, 0., 1., .04])
    correction_errors = {}
    for method, oracle_method in method_map.items():
        expected = multipletests(np.nan_to_num(control, nan=1.), method=oracle_method)[1]
        actual_control = adjust_pvalues(control, method=method)
        keep = np.isfinite(control)
        np.testing.assert_allclose(actual_control[keep], expected[keep], rtol=1e-12, atol=1e-14)
        correction_errors[method] = float(np.max(np.abs(actual_control[keep] - expected[keep])))
    result = {
        'status': 'passed', 'category': source['category'], 'source_commit': source['commit'],
        'inference_commit': report['environment']['commit'], 'implementation_sha256': report['environment']['implementation_sha256'],
        'command': shlex.join([sys.executable, *sys.argv]), 'python': sys.version,
        'numpy': np.__version__, 'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__,
        'os': platform.platform(), 'cpu': report['environment']['cpu'],
        'pytorch': report['environment']['pytorch'], 'device': 'cpu', 'gpu': 'unused',
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'inference_sha256': hashlib.sha256(args.inference.read_bytes()).hexdigest(),
        'factors': len(source['factors']), 'tests': len(family),
        'checked_hac_series': len(errors['standard_error']),
        'maximum_absolute_error': {key: max(values, default=0) for key, values in errors.items()},
        'correction_control_max_error': correction_errors,
        'multiple_testing': report['multiple_testing'],
        'data_source': 'Frozen full synthetic 500-date/200-stock/213-factor report, no factor recomputation',
        'cost_assumptions': 'No transactions, data downloads, GPU or paid APIs; IC excludes execution costs',
        'limitations': ['Checks implementation against Statsmodels, not finite-sample calibration or market alpha.',
                        'Mean HAC compares full original-grid OLS scores with explicit observed-date correction.',
                        'The synthetic source remains a full-sample engineering diagnostic.'],
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'tests': len(family), 'max_errors': result['maximum_absolute_error']}))


if __name__ == '__main__':
    main()
