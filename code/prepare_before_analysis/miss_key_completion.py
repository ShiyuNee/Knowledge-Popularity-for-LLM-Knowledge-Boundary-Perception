import json

def copy_popularity_by_line(target_path, reference_path, output_path=None, popularity_field='popularity'):
    """
    generate的时候可能没保存question的popularity, 这里额外保存一下。
    """
    out_path = output_path or (target_path + '.filled')
    updated = 0
    total = 0

    # 先读取所有数据
    with open(target_path, 'r', encoding='utf-8') as ft:
        target_lines = ft.readlines()
    with open(reference_path, 'r', encoding='utf-8') as fr:
        reference_lines = fr.readlines()

    assert len(target_lines) == len(reference_lines), "两个文件行数不一致！"

    output_lines = []

    # 处理数据
    for line_t, line_r in zip(target_lines, reference_lines):
        total += 1
        obj_t = json.loads(line_t)
        obj_r = json.loads(line_r)

        pop = obj_r.get("popularity")
        obj_t[popularity_field] = pop
        updated += 1

        output_lines.append(json.dumps(obj_t, ensure_ascii=False))

    # 最后统一写入
    with open(out_path, 'w', encoding='utf-8') as fo:
        for line in output_lines:
            fo.write(line + '\n')

    print(f'Processed {total} lines, updated {updated}, output -> {out_path}')
    return out_path


for dataset in ['movies', 'songs', 'basketball']:
    for model_name in ['Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']:
        target = f'../../res/{dataset}/{dataset}_{model_name}_temperature1.jsonl'
        reference = f'../../res/{dataset}/{dataset}_qwen2_temperature1.jsonl'
        output = f'../../res/{dataset}/{dataset}_{model_name}_temperature1.jsonl'
        copy_popularity_by_line(target, reference, output)