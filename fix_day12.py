import json

with open('src/day12.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell-1: KV Cache complexity wording
cell1 = nb['cells'][1]
for i, line in enumerate(cell1['source']):
    if '没有 KV Cache：每步重新计算所有 K,V → O(n²) 计算量' in line:
        cell1['source'][i] = line.replace(
            'O(n²) 计算量',
            '每步 O(n)，n 步总计 O(n²)'
        )
        print(f'Fixed cell-1 line {i}: complexity without cache')
    if '有 KV Cache：只计算新 Token 的 K,V，查表获取历史 → O(n) 计算量' in line:
        cell1['source'][i] = line.replace(
            'O(n) 计算量',
            '每步 O(1)，n 步总计 O(n)'
        )
        print(f'Fixed cell-1 line {i}: complexity with cache')

# Fix cell-2: Qwen2.5-1.5B num_kv_heads from 4 to 2
cell2 = nb['cells'][2]
in_1_5b_block = False
for i, line in enumerate(cell2['source']):
    if 'Qwen2.5-1.5B' in line:
        in_1_5b_block = True
    if in_1_5b_block and 'num_kv_heads' in line and ': 4' in line:
        cell2['source'][i] = line.replace(': 4', ': 2')
        print(f'Fixed cell-2 line {i}: 1.5B num_kv_heads 4 -> 2')
        in_1_5b_block = False

with open('src/day12.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Saved day12.ipynb')
