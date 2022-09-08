# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import concurrent.futures as ft
from PIL import Image
from time import time

def source_encode(code_str, height=None, width=None, ascii_only=False):
    lines = [x for x in code_str.splitlines(keepends=False) if x != '']
    try:
        if height is None:
            height = len(lines)
        if width is None:
            width = max([len(x) for x in lines])
    except Exception:
        raise RuntimeError('input code has invalid format')
    image = np.zeros((3 if not ascii_only else 1, height, width), dtype=np.uint8)
    for h_idx, line in enumerate(lines):
        if h_idx >= height:
            break
        for w_idx, ch in enumerate(line):
            if w_idx >= width:
                break
            if not ascii_only:
                for ch_idx, bt in enumerate(ch.encode('utf-8')):
                    if ch_idx >= 2:
                        break
                    image[ch_idx, h_idx, w_idx] = bt
            else:
                for codepoint in [ord(x) for x in ch]:
                    if codepoint > 127:
                        codepoint = 128
                    image[0, h_idx, w_idx] = codepoint
    return image

def array2image(arr, img_path):
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    if img_path.split('.')[-1] == 'npy':
        np.save(img_path, arr)
        return img_path
    im = Image.fromarray(arr.transpose((1, 2, 0)))
    im.save(img_path)
    return img_path

def code2image(code_filepath, image_path, height=None, width=None, ascii_only=False):
    with open(code_filepath, 'r') as fd:
        code = fd.read()
        arr = source_encode(code, height, width, ascii_only=ascii_only)
        return array2image(arr, image_path)

def batch_convert(codepaths, imagepaths, height=None, width=None, ascii_only=False):
    n_success = 0
    tasks = []
    with ft.ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
        for idx, (src, dest) in enumerate(zip(codepaths, imagepaths)):
            tasks.append(
                executor.submit(code2image, src, dest, height, width, ascii_only)
            )
            batch_time = time()
            if len(tasks) >= 10000 or idx == len(codepaths) - 1:
                for task in ft.as_completed(tasks):
                    try:
                        image_path = task.result()
                        n_success += 1
                        if n_success % 1000 == 0 and n_success != 0:
                            print(f'[{time()-batch_time:.3f}s] code2image successfully converted {n_success}/{len(codepaths)} -> {image_path}')
                            batch_time = time()
                    except Exception as e:
                        print(f'exception caught - {e}')
                tasks = []
    return n_success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input directory or file (source code)')
    parser.add_argument('output', type=str, help='output directory')
    parser.add_argument('--problemid-include', type=str, default='', help='only problems to keep')
    parser.add_argument('--ascii-only', action='store_true', default=False, help='ascii characters only')
    parser.add_argument('--max-jobs', type=int, default=os.cpu_count(), help='number of jobs to process in parallel')
    parser.add_argument('--keep-source-structure', action='store_true', default=False, help='keep source dataset structrue')
    parser.add_argument('--fix-width', type=int, default=-1, help='fix output width')
    parser.add_argument('--fix-height', type=int, default=-1, help='fix output height')
    parser.add_argument('--image-format', type=str, default='png', help='image format (encoder)')
    parser.add_argument('--keyword', type=str, default='', help='path filtering keyword')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    width = None if args.fix_width < 0 else args.fix_width
    height = None if args.fix_height < 0 else args.fix_height

    problemid_include = None
    if args.problemid_include != '':
        with open(args.problemid_include, 'r') as fd:
            problemid_include = {x.strip() for x in fd}

    if os.path.isfile(args.input):
        # single file input (for testing)
        input_filepath = os.path.abspath(args.input)
        output_filepath = os.path.join(
            args.output, 
            '.'.join(os.path.basename(input_filepath).split('.')[:-1]) + '.' + args.image_format.lower())
        time_1 = time()
        code2image(input_filepath, output_filepath, height, width, args.ascii_only)
        time_2 = time()
        print(f'time consumption: {time_2-time_1}s')
    elif os.path.isdir(args.input):
        # batch processing
        src_files = []
        out_files = []
        for root, _, files in os.walk(os.path.abspath(args.input)):
            include = False if problemid_include else True
            if args.keyword != '':
                if args.keyword not in root:
                    continue
            if not include:
                for problemid in problemid_include:
                    if problemid in root:
                        include = True 
                        break
            if not include:
                print(f'skip {root}')
                continue
            src_files.extend(
                [os.path.join(root, x) for x in files]
            )
            # os.makedirs(os.path.join(args.output, root))
            dest_dir = os.path.abspath(args.output)
            if args.keep_source_structure:
                dest_dir = os.path.join(
                    dest_dir, 
                    root.split(os.path.abspath(args.input))[1].lstrip('/').lstrip('\\'))
            out_files.extend(
                [
                    os.path.join(
                        dest_dir,
                        '.'.join(x.split('.')[:-1]) + '.' + args.image_format
                     ) for x in files]
            )
        print(f'Converting {len(src_files)} source files')
        total_converted = batch_convert(
            src_files, 
            out_files, 
            height=height,
            width=width,
            ascii_only=args.ascii_only
            )
        print(f'Converted {total_converted} in total')
        