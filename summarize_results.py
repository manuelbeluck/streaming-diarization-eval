import os
import csv
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
LANGUAGES = ['callhome_eng', 'callhome_deu', 'callhome_jpn', 'callhome_spa', 'callhome_zho']
SYSTEMS = ['diart_default', 'diart_custom', 'streaming_sortformer']
LANG_DISPLAY = {'callhome_eng': 'English', 'callhome_deu': 'German',
                'callhome_jpn': 'Japanese', 'callhome_spa': 'Spanish', 'callhome_zho': 'Mandarin'}

# Accumulate per (lang, system): list of metric values
data = defaultdict(lambda: defaultdict(list))

for lang in LANGUAGES:
    csv_path = os.path.join(RESULTS_DIR, lang, 'metrics.csv')
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sys = row['system']
            for col in ['DER', 'false_alarm', 'missed_detection', 'confusion',
                        'JER', 'latency_mean_ms', 'latency_std_ms', 'peak_latency_ms',
                        'peak_gpu_mem_mb', 'wall_time_s', 'total_speech_time']:
                val = row.get(col, '')
                if val and val.strip():
                    try:
                        data[(lang, sys)][col].append(float(val))
                    except ValueError:
                        pass

def mean(lst):
    return sum(lst) / len(lst) if lst else float('nan')

def fmt(v, decimals=1):
    return f'{v:.{decimals}f}' if v == v else 'N/A'

# Print per-language per-system summary
print('=== PER-LANGUAGE PER-SYSTEM SUMMARY ===\n')
for lang in LANGUAGES:
    print(f'--- {LANG_DISPLAY[lang]} ---')
    for sys in SYSTEMS:
        d = data[(lang, sys)]
        n = len(d.get('der', []))
        der = mean(d.get('DER', []))
        jer = mean(d.get('JER', []))
        fa = mean(d.get('false_alarm', []))
        miss = mean(d.get('missed_detection', []))
        conf = mean(d.get('confusion', []))
        lat = mean(d.get('latency_mean_ms', []))
        lat_std = mean(d.get('latency_std_ms', []))
        peak_lat = mean(d.get('peak_latency_ms', []))
        gpu = mean(d.get('peak_gpu_mem_mb', []))
        wall = mean(d.get('wall_time_s', []))
        print(f'  {sys} (n={n}):')
        print(f'    DER={fmt(der*100)}%  FA={fmt(fa*100)}%  Miss={fmt(miss*100)}%  Conf={fmt(conf*100)}%')
        print(f'    JER={fmt(jer*100)}%')
        print(f'    Latency mean={fmt(lat)}ms  std={fmt(lat_std)}ms  peak={fmt(peak_lat)}ms')
        print(f'    GPU={fmt(gpu,0)}MB  wall={fmt(wall,1)}s')
    print()

# Overall (macro-average across all languages)
print('=== OVERALL (macro-average across languages) ===\n')
for sys in SYSTEMS:
    all_der, all_jer, all_fa, all_miss, all_conf = [], [], [], [], []
    all_lat, all_peak_lat, all_gpu = [], [], []
    for lang in LANGUAGES:
        d = data[(lang, sys)]
        if d.get('DER'):
            all_der.extend(d['DER'])
            all_jer.extend(d.get('JER', []))
            all_fa.extend(d.get('false_alarm', []))
            all_miss.extend(d.get('missed_detection', []))
            all_conf.extend(d.get('confusion', []))
            all_lat.extend(d.get('latency_mean_ms', []))
            all_peak_lat.extend(d.get('peak_latency_ms', []))
            all_gpu.extend(d.get('peak_gpu_mem_mb', []))
    print(f'{sys} (n={len(all_der)}):'  )
    print(f'  DER={fmt(mean(all_der)*100)}%  FA={fmt(mean(all_fa)*100)}%  Miss={fmt(mean(all_miss)*100)}%  Conf={fmt(mean(all_conf)*100)}%')
    print(f'  JER={fmt(mean(all_jer)*100)}%')
    print(f'  Latency mean={fmt(mean(all_lat))}ms  peak={fmt(mean(all_peak_lat))}ms')
    print(f'  GPU={fmt(mean(all_gpu),0)}MB')
    print()

# Per-language DER table for LaTeX
print('=== LaTeX DER TABLE ===')
print('Language & DiArt Default & DiArt Custom & Sortformer \\\\')
for lang in LANGUAGES:
    vals = []
    for sys in SYSTEMS:
        d = data[(lang, sys)]
        der = mean(d.get('DER', []))
        vals.append(fmt(der*100))
    print(f'{LANG_DISPLAY[lang]} & {vals[0]} & {vals[1]} & {vals[2]} \\\\')
row = []
for sys in SYSTEMS:
    all_der = []
    for lang in LANGUAGES:
        all_der.extend(data[(lang, sys)].get('DER', []))
    row.append(fmt(mean(all_der)*100))
print(f'Overall & {row[0]} & {row[1]} & {row[2]} \\\\')
print()

# Per-language DER breakdown (FA/Miss/Conf as fractions of total_speech_time)
print('=== LaTeX DER BREAKDOWN TABLE ===')
print('System & Lang & FA\\% & Miss\\% & Conf\\% & DER\\% \\\\')
for sys in SYSTEMS:
    for lang in LANGUAGES:
        d = data[(lang, sys)]
        total = d.get('total_speech_time', [])
        fa_abs = d.get('false_alarm', [])
        miss_abs = d.get('missed_detection', [])
        conf_abs = d.get('confusion', [])
        der = mean(d.get('DER', []))
        if total:
            fa_frac = mean([f/t for f,t in zip(fa_abs, total)])
            miss_frac = mean([m/t for m,t in zip(miss_abs, total)])
            conf_frac = mean([c/t for c,t in zip(conf_abs, total)])
        else:
            fa_frac = miss_frac = conf_frac = float('nan')
        print(f'{sys} & {LANG_DISPLAY[lang]} & {fmt(fa_frac*100)} & {fmt(miss_frac*100)} & {fmt(conf_frac*100)} & {fmt(der*100)} \\\\')
print()

# Per-language JER table
print('=== LaTeX JER TABLE ===')
print('Language & DiArt Default & DiArt Custom & Sortformer \\\\')
for lang in LANGUAGES:
    vals = []
    for sys in SYSTEMS:
        d = data[(lang, sys)]
        jer = mean(d.get('JER', []))
        vals.append(fmt(jer*100))
    print(f'{LANG_DISPLAY[lang]} & {vals[0]} & {vals[1]} & {vals[2]} \\\\')
row = []
for sys in SYSTEMS:
    all_jer = []
    for lang in LANGUAGES:
        all_jer.extend(data[(lang, sys)].get('JER', []))
    row.append(fmt(mean(all_jer)*100))
print(f'Overall & {row[0]} & {row[1]} & {row[2]} \\\\')
print()

# Latency table
print('=== LaTeX LATENCY TABLE ===')
print('System & Mean (ms) & Std (ms) & Peak (ms) \\\\')
for sys in SYSTEMS:
    all_lat, all_std, all_peak = [], [], []
    for lang in LANGUAGES:
        d = data[(lang, sys)]
        all_lat.extend(d.get('latency_mean_ms', []))
        all_std.extend(d.get('latency_std_ms', []))
        all_peak.extend(d.get('peak_latency_ms', []))
    print(f'{sys} & {fmt(mean(all_lat),1)} & {fmt(mean(all_std),1)} & {fmt(mean(all_peak),0)} \\\\')
print()

# GPU memory
print('=== GPU MEMORY (Sortformer only) ===')
for lang in LANGUAGES:
    d = data[(lang, 'streaming_sortformer')]
    gpu = mean(d.get('peak_gpu_mem_mb', []))
    print(f'{LANG_DISPLAY[lang]}: {fmt(gpu,0)} MB')
