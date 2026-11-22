"""Generate coverage tables from the JSON-compatible YAML 1.2 manifest. No job submission."""
import csv,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def main():
    cfg=json.loads((BASE/'reproduction.yaml').read_text())
    rows=[]
    for suite in cfg['suites']:
        for task in suite['tasks']:
            variants=[(m,'default') for m in suite['methods']]+[(m,'4') for m in suite['rank4_variants']]
            for method,rank in variants:
                for profile,settings in cfg['profiles'].items():
                    for seed in settings['seeds']:
                        rows.append({'suite':suite['id'],'model':suite['model'],'task':task,'method':method,'rank_variant':rank,'integration':cfg['integration_by_method'].get(method,'none'),'profile':profile,'seed':seed,'configuration':suite['config'],'execution_status':'not inferred from manifest'})
    with (BASE/'experiment_matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['| Suite | Model | Tasks | Methods / variants | Configuration |','|---|---|---|---|---|']
    for s in cfg['suites']:
        lines.append('| '+' | '.join([s['id'],s['model'],', '.join(s['tasks']),', '.join(s['methods'])+'; rank 4: '+', '.join(s['rank4_variants']),s['config']])+' |')
    (BASE/'coverage_table.md').write_text('\n'.join(lines)+'\n\nSeed 42 is the reported profile. Seeds 42/43/44 define a planned extension, not completed repetitions. Rows declare coverage, not observed run status.\n')
    print(f'{len(rows)} coverage rows generated; no jobs submitted.')
if __name__=='__main__':main()
