import json
import os

"""
简单脚本：解析 `llm_pop_generation/` 下的 JSONL 输出文件，提取 `Res` 与 `pop_value`，并汇总统计。
用法示例：
python parse_llm_outputs.py ./llm_pop_generation/movies/
"""


def parse_file(path):
    stats = {'total': 0, 'parsed': 0, 'unparsed': 0, 'values': []}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            stats['total'] += 1
            pv = obj.get('pop_value')
            if pv is None:
                # 尝试从 Res 解析
                res = obj.get('Res', '')
                import re
                m = re.search(r"(-?\d+)", str(res))
                if m:
                    try:
                        v = int(m.group(1))
                        v = max(1, min(10, v))
                        pv = v
                    except Exception:
                        pv = None
            if pv is None:
                stats['unparsed'] += 1
            else:
                stats['parsed'] += 1
                stats['values'].append(pv)
    return stats


def main(dirpath):
    results = {}
    for root, dirs, files in os.walk(dirpath):
        for fn in files:
            if fn.endswith('.jsonl'):
                p = os.path.join(root, fn)
                stats = parse_file(p)
                results[fn] = stats
    # print summary
    for fn, s in results.items():
        vals = s['values']
        avg = sum(vals)/len(vals) if vals else None
        print(f"{fn}: total={s['total']} parsed={s['parsed']} unparsed={s['unparsed']} avg={avg}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python parse_llm_outputs.py <dirpath>')
    else:
        main(sys.argv[1])
