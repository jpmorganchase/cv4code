# SPDX-License-Identifier: Apache-2.0
import os
import argparse
from torchtext.vocab import build_vocab_from_iterator
from modules.tokenizer import tokenize

def yield_tokens(paths, srcdir, ascii_only=False):
    srcdir = [srcdir] if isinstance(srcdir, str) else srcdir
    for path in paths:
        for src in srcdir:
            fullpath = os.path.join(src, path if not path.startswith('/') else path[1:])
            if not os.path.isfile(fullpath):
                continue
            with open(fullpath, 'r') as fd:
                codestr = fd.read()
                yield tokenize(codestr, ascii_only=ascii_only)

def get_filepaths(datadir, subset, srcdir):
    dataids = set()
    filepaths = set()
    with open(os.path.join(datadir, f'{subset}.csv'), 'r') as fd:
        for idx, line in enumerate(fd):
            if idx == 0:
                continue
            info = line.strip().split(',')
            dataid= info[0]
            dataids.add(dataid)
    
    for root, _, files in os.walk(srcdir):
        for f in files:
            if f.split('.')[0] in dataids:
                filepaths.add(os.path.join(root, f))
    return filepaths

def main(args):
    filepaths = get_filepaths(args.dataset_dir, args.subset, args.srcdir)
    vocab = build_vocab_from_iterator(
        yield_tokens(filepaths, args.srcdir, ascii_only=args.ascii_only), 
        min_freq=args.min_freq,
        specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    with open(os.path.join(args.dataset_dir, 'vocab'), 'w') as fd:
        for wrd in vocab.get_itos():
            print(wrd, file=fd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help='dataset directory')
    parser.add_argument('srcdir', type=str, help='source directory of raw code data')
    parser.add_argument('--subset', type=str, default='train', help='source directory of the image data')
    parser.add_argument('--min_freq', type=int, default=1, help='vocab min frequency')
    parser.add_argument('--ascii_only', action='store_true', default=False, help='keep only ascii')
    args = parser.parse_args()
    main(args)