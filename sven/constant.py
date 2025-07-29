SEC_LABEL = 'sec'
VUL_LABEL = 'vul'
BINARY_LABELS = [SEC_LABEL, VUL_LABEL]

CWES = [
    'cwe-020',
    'cwe-022',
    'cwe-078',
    'cwe-079',
    'cwe-089',
    'cwe-094',
    'cwe-095',
    'cwe-113',
    'cwe-117',
    'cwe-119',
    'cwe-120',
    'cwe-121',
    'cwe-125',
    'cwe-131',
    'cwe-134',
    'cwe-170',
    'cwe-190',
    'cwe-193',
    'cwe-215',
    'cwe-295',
    'cwe-312',
    'cwe-326',
    'cwe-327',
    'cwe-329',
    'cwe-347',
    'cwe-377',
    'cwe-400',
    'cwe-416',
    'cwe-468',
    'cwe-476',
    'cwe-502',
    'cwe-601',
    'cwe-611',
    'cwe-643',
    'cwe-676',
    'cwe-732',
    'cwe-760',
    'cwe-776',
    'cwe-730',
    'cwe-787',
    'cwe-918',
    'cwe-943',
]

CWES_TRAINED = [
    'cwe-022',
    'cwe-078',
    'cwe-079',
    'cwe-089',
    'cwe-125',
    'cwe-190',
    'cwe-416',
    'cwe-476',
    'cwe-787',
]

NOT_TRAINED = [
    'cwe-020',
    'cwe-119',
    'cwe-502',
    'cwe-732',
]

VAL_SCENARIOS = {
    ('cwe-078', '2-py'),
    ('cwe-089', '2-py'),
    ('cwe-125', '2-c'),
    ('cwe-190', '2-c'),
    ('cwe-022', '2-py'),
    ('cwe-787', '2-c'),
}


CWES_DICT = {
    'base': CWES_TRAINED,
    'untrain': NOT_TRAINED,
}

LANGUAGE_MAPS = {
    'py': 'Python',
    'c': 'C'
}


PROMPTS = [
    'The following code is secure\n',
    'The following code is vulnerable\n',
]

MODEL_DIRS = {
    'deepseek-1.3b':'deepseek-ai/deepseek-coder-1.3b-base',
    'deepseek-6.7b':'deepseek-ai/deepseek-coder-6.7b-base',
    'qwen2.5-7b': 'Qwen/Qwen2.5-Coder-7B',
    'qwen2.5-3b': 'Qwen/Qwen2.5-Coder-3B',
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-Coder-0.5B',
    'seedcoder-8b': 'ByteDance-Seed/Seed-Coder-8B-Base',
}