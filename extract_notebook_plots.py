"""Extract all image/png outputs from notebooks and save to chapters/images/."""
import json
import base64
import os

NOTEBOOKS = [
    ('callhome_eng_analysis', 'analysis/callhome_eng_analysis.ipynb'),
    ('callhome_deu_analysis', 'analysis/callhome_deu_analysis.ipynb'),
    ('callhome_jpn_analysis', 'analysis/callhome_jpn_analysis.ipynb'),
    ('callhome_spa_analysis', 'analysis/callhome_spa_analysis.ipynb'),
    ('callhome_zho_analysis', 'analysis/callhome_zho_analysis.ipynb'),
    ('cross_language', 'analysis/cross_language_comparison.ipynb'),
]

OUT_DIR = '../../chapters/images/notebook_plots'
os.makedirs(OUT_DIR, exist_ok=True)

total = 0
for nb_name, nb_path in NOTEBOOKS:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    code_cell_index = 0
    for cell in cells:
        if cell.get('cell_type') == 'code':
            code_cell_index += 1
            outputs = cell.get('outputs', [])
            img_index = 0
            for output in outputs:
                data = output.get('data', {})
                if 'image/png' in data:
                    img_data = data['image/png']
                    # img_data may be a list of strings or a single string
                    if isinstance(img_data, list):
                        img_data = ''.join(img_data)
                    img_bytes = base64.b64decode(img_data)
                    fname = f'{nb_name}_cell{code_cell_index:02d}_{img_index:02d}.png'
                    out_path = os.path.join(OUT_DIR, fname)
                    with open(out_path, 'wb') as f_out:
                        f_out.write(img_bytes)
                    print(f'Saved: {fname}')
                    img_index += 1
                    total += 1

print(f'\nTotal images extracted: {total}')
print(f'Output directory: {os.path.abspath(OUT_DIR)}')
