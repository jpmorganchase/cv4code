# SPDX-License-Identifier: Apache-2.0
import re

def tokenize(codestr, ascii_only=False):
    if ascii_only:
        codestr = codestr.encode('ascii', errors='ignore').decode()
    tokens = []
    for line in codestr.splitlines():
        indent = re.search(r'^\s+', line)
        if indent:
            tokens.append(line[:indent.span()[1]])
            line = line[indent.span()[1]+1:]
        tokens.extend([x for x in re.split('(\W)', line) if x != '' and x != ' '])
    return tokens