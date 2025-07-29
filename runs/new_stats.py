import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(project_root)
import json
import csv
import re
import argparse
from sven.constant import CWES, VAL_SCENARIOS, NOT_TRAINED

def get_codeql_vulnerable_files(codeql_csv_path):
    vulnerable_files = set()
    try:
        with open(codeql_csv_path, 'r', encoding='utf-8') as csvfile:
            content = csvfile.read()
            if not content.strip():
                print(f"警告: CodeQL CSV文件 {codeql_csv_path} 为空")
                return vulnerable_files
            
            csvfile.seek(0)
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 5:
                    file_path = row[4]
                    print(file_path)
                    match = re.search(r'/(\d+)\.py|/(\d+)\.c', file_path)
                    if match:
                        file_number = match.group(1) or match.group(2)
                        file_name = str(file_number) + file_path.split(str(file_number))[1]
                        vulnerable_files.add(file_name)
                        
    except FileNotFoundError:
        print(f"警告: CodeQL CSV文件 {codeql_csv_path} 未找到")
    except Exception as e:
        print(f"错误: 解析CodeQL CSV文件时出错: {e}")
    
    print(f"CodeQL检测到的不安全文件编号: {sorted(vulnerable_files)}")
    return vulnerable_files

def get_old_stats_json(stat_path):
    with open(stat_path, "r") as file:
        stat = json.load(file)
    return stat

def modify_stat_single(stat, vulnerable_files):
    for fname in stat.keys():
        stat[fname]["num"] = 1
        is_secure = fname not in vulnerable_files
        stat[fname]["sec"] = is_secure
    return stat

def gen_new_stat(path):
    if 'base' in path:
        base_path = path
        for vul_type in os.listdir(base_path):
            if vul_type not in CWES:
                continue
            vul_type_path = os.path.join(base_path, vul_type)
            for sub_type in os.listdir(vul_type_path):
                scenario = (vul_type, sub_type)
                if scenario in VAL_SCENARIOS:
                    continue
                sub_type_path = os.path.join(vul_type_path, sub_type)
                if not os.path.isdir(sub_type_path):
                    continue

                vulnerable_files = get_codeql_vulnerable_files(os.path.join(sub_type_path, 'codeql.csv'))
                stat = get_old_stats_json(os.path.join(sub_type_path, 'stat.json'))
                stat = modify_stat_single(stat, vulnerable_files)
                with open(os.path.join(sub_type_path, 'new_stat.json'), 'w') as file:
                    json.dump(stat, file, indent=4)

    elif 'untrain' in path:
        base_path = path
        for vul_type in os.listdir(base_path):
            if vul_type not in NOT_TRAINED:
                continue
            vul_type_path = os.path.join(base_path, vul_type)
            for sub_type in os.listdir(vul_type_path):
                scenario = (vul_type, sub_type)
                sub_type_path = os.path.join(vul_type_path, sub_type)
                if not os.path.isdir(sub_type_path):
                    continue

                vulnerable_files = get_codeql_vulnerable_files(os.path.join(sub_type_path, 'codeql.csv'))
                stat = get_old_stats_json(os.path.join(sub_type_path, 'stat.json'))
                stat = modify_stat_single(stat, vulnerable_files)
                with open(os.path.join(sub_type_path, 'new_stat.json'), 'w') as file:
                    json.dump(stat, file, indent=4)
    else:
        print(f"警告: 目录 {path} 下未找到 base 目录")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, required=True)
    parser.add_argument('--eval_type', type=str, default='base', required=True)
    args = parser.parse_args()
    path = os.path.join(args.paths, args.eval_type)
    gen_new_stat(path)


if __name__ == "__main__":
    main()