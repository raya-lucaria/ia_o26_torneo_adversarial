"""Audita semilla técnica propuesta, sin leer el lote256/test/S13 ni jugar."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess

from fase8a_soporte import ROOT, Budget, sha, save

OUT=ROOT/'resultados/medicion/fase8B2_runtime_20261029'
SEED=20261221


def main():
    budget=Budget()
    output=OUT/'AUDITORIA_SEMILLA_20261221.json'
    if output.exists():
        raise FileExistsError('Auditoría ya congelada; no sobrescribir')
    roots={ROOT.resolve()}
    inventories=[]
    for parent in [ROOT.parent, Path('/Users/josema/.codex/worktrees')]:
        if not parent.exists():continue
        if parent==ROOT.parent:
            roots.update(p.resolve() for p in parent.iterdir() if p.is_dir() and p.name.startswith('BreakthroughBotIA'))
        else:
            for container in parent.iterdir():
                if container.is_dir():
                    roots.update(p.resolve() for p in container.iterdir() if p.is_dir() and p.name.startswith('BreakthroughBotIA'))
    git_registries=[]
    for root in sorted(roots.copy()):
        if (root/'.git').exists():
            result=subprocess.run(['rtk','proxy','git','-C',str(root),'worktree','list','--porcelain'],capture_output=True,text=True)
            git_registries.append({'repository':str(root),'exit':result.returncode,'output':result.stdout,'stderr':result.stderr})
            if result.returncode==0:
                roots.update(Path(line[9:]).resolve() for line in result.stdout.splitlines() if line.startswith('worktree '))
        budget.wait()
    inventory=ROOT/'resultados/medicion/fase6_cobertura_20261029/inventario.json'
    frozen=json.loads(inventory.read_text())
    previous_uses=[{'path':r['path'],'seed':r['seed'],'sha256':r['sha256'],'population':r['population']} for r in frozen['traces'] if r['seed']==SEED]
    hits=[]
    skipped=[]
    scanned=0
    pattern=re.compile(rb'(?<![0-9])20261221(?![0-9])')
    excluded_prefixes=('fase5','fase6','fase7','fase8','validacion','test','examen','shards','dataset','datos','train')
    for root in sorted(roots):
        result_root=root/'resultados'
        if not result_root.is_dir():continue
        for folder, dirs, files in os.walk(result_root):
            base=Path(folder)
            kept=[]
            for name in dirs:
                lower=name.lower()
                if lower.startswith(excluded_prefixes) or 's13' in lower or lower in ['.git','.venv','__pycache__']:
                    skipped.append({'path':str(base/name),'reason':'sealed_or_analysis_or_dataset_subtree_not_read'})
                elif not (base/name).is_symlink():kept.append(name)
            dirs[:]=sorted(kept)
            for name in sorted(files):
                path=base/name
                if path.is_symlink() or path.suffix.lower() not in ['.json','.jsonl','.csv','.log','.md','.txt']:continue
                if path.stat().st_size>2*1024*1024:
                    skipped.append({'path':str(path),'reason':'large_nontrace_text_not_read'});continue
                raw=path.read_bytes();scanned+=1
                if pattern.search(raw):
                    item={'path':str(path),'sha256':sha(path),'bytes':len(raw),'classification':'ambiguous_reference'}
                    if path.suffix=='.json':
                        try:
                            d=json.loads(raw)
                            if isinstance(d,dict) and d.get('version')==1 and isinstance(d.get('meta'),dict) and d['meta'].get('semilla_maestra')==SEED:
                                item['classification']='completed_or_attempted_game_seed'
                        except (ValueError,UnicodeDecodeError):pass
                    hits.append(item)
                budget.wait()
    available=not previous_uses and not hits
    report={'at':datetime.now().astimezone().isoformat(),'seed':SEED,'decision_sha256':sha(OUT/'DECISION_TECNICA.json'),
            'available_in_audited_scope':available,'known_trace_inventory':{'path':str(inventory),'sha256':sha(inventory),'traces':len(frozen['traces'])},
            'known_trace_uses':previous_uses,'roots_discovered':[str(p) for p in sorted(roots)],'git_worktree_registries':git_registries,
            'text_files_scanned':scanned,'matching_files':hits,'excluded_subtrees_and_large_files':skipped,
            'sealed_lot256_test_and_S13_not_read':True,'no_games_started':True,
            'limit':'Known live repository/worktree result metadata and prior890trace inventory; excluded sealed training/test/S13 data. No universal claim about unavailable external archives.',
            'use_gate':'Only usable if available_in_audited_scope true, runtime/image frozen, exclusive window accepted BEFORE first game'}
    save(output,report)
    budget.wait()
    print(json.dumps({'seed':SEED,'roots':len(roots),'files_scanned':scanned,'known_uses':len(previous_uses),
                      'matching_files':hits,'available_in_scope':available},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
