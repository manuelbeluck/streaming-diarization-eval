import re

with open(r'evaluation_pipeline_overview.svg', 'r', encoding='utf-8') as f:
    svg = f.read()

# Replace light-dark(val1, val2) with val1 (light-mode value)
svg_fixed = re.sub(
    r'light-dark\((#[0-9a-fA-F]+),\s*#[0-9a-fA-F]+\)',
    r'\1', svg
)
svg_fixed = re.sub(
    r'light-dark\((rgb\(\d+,\s*\d+,\s*\d+\)),\s*rgb\(\d+,\s*\d+,\s*\d+\)\)',
    r'\1', svg_fixed
)
# Handle light-dark(#hex, var(--ge-dark-color, #hex))
svg_fixed = re.sub(
    r'light-dark\((#[0-9a-fA-F]+),\s*var\([^)]+\)\)',
    r'\1', svg_fixed
)

remaining = len(re.findall(r'light-dark', svg_fixed))
print('Remaining light-dark:', remaining)

with open(r'evaluation_pipeline_fixed.svg', 'w', encoding='utf-8') as f:
    f.write(svg_fixed)
print('Done. Written', len(svg_fixed), 'chars')



