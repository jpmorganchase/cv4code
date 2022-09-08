# SPDX-License-Identifier: Apache-2.0
import os
import sys
import csv
import random
import logging
import argparse
import numpy as np
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils.helper import logging_handle

random.seed(0)

logger = logging.getLogger("prepare_codenet")

_problem_list_metadata_fields = [
    ("id", str),
    ("time_limit", float),
    ("memory_limit", float),
]

_problem_metadata_fields = [
    ("problem_id", str),
    ("language", str),
    ("status", str),
    ("cpu_time", float),
    ("memory", float),
    ("code_size", float),
]

def read_existing_set(csvpath, langs):
    dataset = defaultdict(dict)
    problem2submissions = defaultdict(int)
    with open(os.path.join(csvpath), 'r') as fd:
        reader = csv.DictReader(fd)
        tasks = [x for x in reader.fieldnames if x != 'submission_id']
        for row in reader:
            subid = row['submission_id']
            if (isinstance(langs, list) and len(langs) > 0) and row['language'] not in langs:
                continue
            for task in tasks:
                v = row[task]
                try:
                    v = float(v)
                except Exception:
                    pass
                dataset[subid][task] = v
            problem2submissions[dataset[subid]['problem_id']] += 1

    return dataset, list(problem2submissions.keys()), problem2submissions

def read_duplicated_problems(duplicate_file):
    prob2clusters = dict()
    with open(duplicate_file, "r") as fd:
        cluster_pids_list = [line.strip().split(",") for line in fd]
        for cluster_pids in cluster_pids_list:
            cluster_id = "_".join(cluster_pids)
            for pid in cluster_pids:
                prob2clusters[pid] = cluster_id
    return prob2clusters


def _read_duplicate_cluster(cluster_filepath, cluster_prefix):
    submission2cluster = dict()
    cid = 0
    cluster_size = 0
    with open(cluster_filepath, "r") as fd:
        for line in fd:
            if line.strip() == "":
                cid += 1
                cluster_size = 0
                continue
            cluster_size += 1
            if cluster_size > 1:
                subid = line.strip().split()[0].split('/')[-1].split('.')[0]
                submission2cluster[subid] = f'{cluster_prefix}_{cid}'
    return submission2cluster

def read_near_duplicated_submissions(duplicate_dir):
    langs = ["C", "C++", "Java", "Python"]
    sub2clusters = dict()
    for lang in langs:
        cluster_dir = os.path.join(duplicate_dir, lang, "clusters")
        for cluster in os.listdir(cluster_dir):
            pid = cluster.split("-")[0]
            cluster_filepath = os.path.join(cluster_dir, cluster)
            sub2clusters.update(
                _read_duplicate_cluster(cluster_filepath, f'{lang}_{pid}')
                )
    return sub2clusters

def _parse_csv(filepath, fields, key_str, duplicate_clusters=None):
    meta = dict()
    with open(filepath, "r") as fd:
        reader = csv.DictReader(fd)
        uniq_fields = set([x[0] for x in fields])
        if len(set(uniq_fields).intersection(reader.fieldnames)) < len(uniq_fields):
            raise ValueError(f"{filepath} does not have valid annotations")
        for row in reader:
            dataid = row[key_str]
            mapped = {k: t(row[k]) for (k, t) in fields if row[k] != ""}
            if len(mapped) == len(uniq_fields):
                meta[dataid] = mapped
                if duplicate_clusters is not None:
                    if dataid in duplicate_clusters:
                        meta[dataid]['cluster'] = duplicate_clusters[dataid]
    return meta

def parse_problem_csv(filepath, duplicate_clusters):
    return _parse_csv(filepath, _problem_metadata_fields, "submission_id", duplicate_clusters)


def parse_problemlist_csv(filepath):
    return _parse_csv(filepath, _problem_list_metadata_fields, "id")


def read_metadata(directory, duplicate_dir, excluded_problem_ids=[], included_problem_ids=[], lang=[]):
    metadata = dict()
    n_processed = 0
    problemid2submissions = defaultdict(int)

    problemlist_meta = None
    n_skipped = 0

    pid_clusters = dict()
    if duplicate_dir != '':
        logger.info('read identical problem clusters')
        pid_clusters = read_duplicated_problems(
            os.path.join(duplicate_dir, "identical_problem_clusters")
        )

    sub_clusters = dict()
    problem_ids_skipped = 0
    if duplicate_dir != '':
        logger.info('read near identical submissions')
        sub_clusters = read_near_duplicated_submissions(duplicate_dir)
    for f in [os.path.join(directory, x) for x in os.listdir(directory)]:
        if os.path.basename(f) == "problem_list.csv":
            problemlist_meta = parse_problemlist_csv(
                os.path.join(directory, "problem_list.csv")
            )
            continue
        try:
            datapoints = parse_problem_csv(f, sub_clusters)
            problem_id = os.path.basename(f).split(".")[0]
            if problem_id in excluded_problem_ids:
                problem_ids_skipped += 1
                continue
            if len(included_problem_ids) > 0 and problem_id not in included_problem_ids:
                problem_ids_skipped += 1
                continue
            if problem_id in pid_clusters:
                for v in datapoints.values():
                    v["problem_id"] = pid_clusters[problem_id]
                problemid2submissions[pid_clusters[problem_id]] += len(datapoints)
            else:
                problemid2submissions[problem_id] += len(datapoints)
            if len(lang) > 0:
                metadata.update({dataid : point for dataid, point in datapoints.items() if point['language'] in lang})
            else:
                metadata.update(datapoints)
        except ValueError as e:
            logger.warning(f'parsing failed {e}')
            n_skipped += 1
            continue
        n_processed += 1
        if n_processed % 100 == 0:
            logger.debug(f"Parsed {n_processed} metadata fields")
    logger.info(f"Parsed {n_processed} metadata fields from {directory}")
    if n_skipped > 0:
        logger.warning(f"Skipped {n_skipped} metadata entries")
    if problem_ids_skipped > 0:
        logger.info(f"Skipped {problem_ids_skipped} problems")
    return metadata, problemlist_meta, problemid2submissions


def get_subset(train_dict, size, problem2submission=None, no_dup=False, max_loop=10):
    problem2target_num = None
    if problem2submission is not None:
        problem2target_num = {
            k: int(v / len(train_dict) * size) for k, v in problem2submission.items()
        }
    val_dict = dict()
    loop_idx = 0
    while len(val_dict) < size and loop_idx < max_loop:
        dataids = list(train_dict.keys())
        val_dataids = random.sample(dataids, size)
        for dataid in val_dataids:
            if no_dup and 'cluster' in train_dict[dataid]:
                continue
            if problem2target_num is not None:
                if problem2target_num[train_dict[dataid]['problem_id']] < 1:
                    continue
                problem2target_num[train_dict[dataid]['problem_id']] -= 1
            val_dict[dataid] = train_dict.pop(dataid)
            if len(val_dict) >= size:
                break
        loop_idx += 1
        logger.info(f'sampled from {len(train_dict) + len(val_dict)}, {len(val_dict)}/{size}, {loop_idx}/{max_loop}')
    return val_dict

def get_field_info(data_dict):
    problem_ids = defaultdict(int)
    languages = defaultdict(int)
    status = defaultdict(int)
    cpu_times = list()
    memory = list()
    code_size = list()

    info = dict()
    for data in data_dict.values():
        problem_ids[data["problem_id"]] += 1
        languages[data["language"]] += 1
        status[data["status"]] += 1
        cpu_times.append(data["cpu_time"])
        memory.append(data["memory"])
        code_size.append(data["code_size"])
    info["problem_ids/total_#instances"] = len(problem_ids)
    info["problem_id/mean_#instances"] = np.mean(list(problem_ids.values()))
    info["problem_id/min_#instances"] = np.min(list(problem_ids.values()))
    info["problem_id/max_#nstances"] = np.max(list(problem_ids.values()))
    info["problem_id/std_#instances"] = np.std(list(problem_ids.values()))
    info["languages/total_#languages"] = len(languages)
    info["language/mean_#instances"] = np.mean(list(languages.values()))
    info["language/min_#instances"] = np.min(list(languages.values()))
    info["language/max_#nstances"] = np.max(list(languages.values()))
    info["language/std_#instances"] = np.std(list(languages.values()))
    info["language/total_#status"] = len(status)
    info["status/mean_#instances"] = np.mean(list(status.values()))
    info["status/min_#instances"] = np.min(list(status.values()))
    info["status/max_#nstances"] = np.max(list(status.values()))
    info["status/std_#instances"] = np.std(list(status.values()))
    info["cpu_time/mean"] = np.mean(cpu_times)
    info["cpu_time/min"] = np.min(cpu_times)
    info["cpu_time/max"] = np.max(cpu_times)
    info["cpu_time/std"] = np.std(cpu_times)
    info["memory/mean"] = np.mean(memory)
    info["memory/min"] = np.min(memory)
    info["memory/max"] = np.max(memory)
    info["memory/std"] = np.std(memory)
    info["code_size/mean"] = np.mean(code_size)
    info["code_size/min"] = np.min(code_size)
    info["code_size/max"] = np.max(code_size)
    info["code_size/std"] = np.std(code_size)
    return info


def _write_csv(filepath, metadict):
    with open(filepath, "w") as fd:
        writer = csv.DictWriter(
            fd, fieldnames=["submission_id"] + [x[0] for x in _problem_metadata_fields] + ['cluster']
        )
        writer.writeheader()
        for dataid, value in metadict.items():
            row = value.copy()
            row["submission_id"] = dataid
            writer.writerow(row)

def _dump_dataset(id2image, dataset_dict, output_dir, name):
    train_dict = {k: v for k, v in dataset_dict.items() if k in id2image}
    _write_csv(os.path.join(output_dir, f"{name}.csv"), train_dict)
    with open(os.path.join(output_dir, f"{name}.info"), "w") as fd:
        for k, v in get_field_info(train_dict).items():
            ss = f"{k} : {v}"
            logger.debug(ss)
            print(ss, file=fd)
    logger.info(f"{name} size : {len(train_dict)}")

def dump_datasets(output_dir, image2id, train_dict, val_dict=None, test_dict=None):
    os.makedirs(output_dir, exist_ok=True)
    # potential loss of augmented versions, but it's ok for this use here
    id2image = {v: k for k, v in image2id.items()}
    with open(os.path.join(output_dir, "image2id"), "w") as fd:
        for image, id in image2id.items():
            print(f"{image} {id}", file=fd)
    _dump_dataset(id2image, train_dict, output_dir, 'train')
    if val_dict is not None:
        _dump_dataset(id2image, val_dict, output_dir, 'val')
    if test_dict is not None:
        _dump_dataset(id2image, test_dict, output_dir, 'test')

def collect_images(image_dir, metadict):
    image2id = dict()
    n_missing_labels = 0
    for root, _, files in os.walk(image_dir):
        for f in files:
            submission_id = ".".join(os.path.basename(f).split(".")[:-1])
            if submission_id in metadict:
                image2id[os.path.abspath(os.path.join(root, f))] = submission_id
            else:
                n_missing_labels += 1
    logger.warning(f"{n_missing_labels} images do not have labels")
    return image2id

def update_problem2submissions(image2id, metadict, problem2submissions):
    id2image = {y: x for x, y in image2id.items()}
    deleted = 0
    for submissionid in list(metadict.keys()):
        if submissionid not in id2image:
            if problem2submissions[metadict[submissionid]['problem_id']] > 0:
                problem2submissions[metadict[submissionid]['problem_id']] -= 1
            del metadict[submissionid]
            deleted += 1
    logger.info(f'removed {deleted} submissions with no images')
        
def filter_minimum_submissions(dataset_dict, prob2submissions, image2id, min_n):
    available_ids = set(image2id.values())
    filtered_datset = dict()
    n_removed = 0
    for subid in list(dataset_dict.keys()):
        if subid not in available_ids:
            prob2submissions[dataset_dict[subid]['problem_id']] -= 1
    for subid in list(dataset_dict.keys()):
        if prob2submissions[dataset_dict[subid]['problem_id']] >= min_n:
            filtered_datset[subid] = dataset_dict.pop(subid)
        else:
            prob2submissions.pop(dataset_dict[subid]['problem_id'])
            n_removed += 1
    logger.info(f'filtering problems with minimum submission {min_n}, removed {n_removed}')
    return filtered_datset

def limit_num_problems(dataset_dict, prob2submissions, max_problems, max_submission_per_problem=-1):
    filtered_datset = dict()
    problemid_sorted_by_submissions = [
        x[0] for x in sorted(
            prob2submissions.items(), key=lambda x : x[1], reverse=True)
            ]
    if max_problems > 0:
        problemid_sorted_by_submissions = problemid_sorted_by_submissions[:max_problems]
    problem2taken = {pid: 0 for pid in problemid_sorted_by_submissions}
    for subid in list(dataset_dict.keys()):
        if dataset_dict[subid]['problem_id'] in problemid_sorted_by_submissions:
            if max_submission_per_problem > 0:
                if problem2taken[dataset_dict[subid]['problem_id']] >= max_submission_per_problem:
                    continue
            problem2taken[dataset_dict[subid]['problem_id']] += 1
            filtered_datset[subid] = dataset_dict.pop(subid)
    logger.info(f'filtering {len(problemid_sorted_by_submissions)}/{len(prob2submissions)} problems')
    return filtered_datset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadir", type=str, help="meatadata directory")
    parser.add_argument("imgdir", type=str, help="directory with all the images")
    parser.add_argument(
        "outdir", type=str, help="output directory to hold the processed data"
    )
    parser.add_argument(
        "--duplicates", type=str, default="", help="duplicate directories"
    )
    parser.add_argument(
        "--val",
        type=int,
        default=0,
        help="number of data points to split for validation",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="number of data points to split for validation",
    )
    parser.add_argument(
        "--norm", action="store_true", default=False, help="key raw numerical labels"
    )
    parser.add_argument(
        "--keep-dup", action="store_true", default=False, help="allow duplciated submissions in validation"
    )
    parser.add_argument(
        "--minimum-submission", type=int, default=1, help="keep problemid with at least this number of submissions"
    )
    parser.add_argument(
        "--problemid-exclude", type=str, default='', help="the list of problems to exclude"
    )
    parser.add_argument(
        "--problemid-include", type=str, default='', help="the list of problems to include"
    )
    parser.add_argument(
        "--max-problems", type=int, default=-1, help="the number of problems to keep"
    )
    parser.add_argument(
        "--problem-submissions", type=int, default=-1, help="the number of submissions of each problem"
    )
    parser.add_argument(
        "--parent-csv", type=str, default='', help="read from an existing set"
    )
    parser.add_argument(
        '--lang', type=str, default='', help='limit the language in considerable if non empty'
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    logging_handle()

    excluded_problem_list = []
    included_problem_list = []
    if args.problemid_exclude != '':
        with open(args.problemid_exclude, 'r') as fd:
            for line in fd:
                excluded_problem_list.append(line.strip())
    if args.problemid_include != '':
        with open(args.problemid_include, 'r') as fd:
            for line in fd:
                included_problem_list.append(line.strip())

    langs = [] if not args.lang else args.lang
    if langs:
        langs = langs.split(',') if ',' in langs else [args.lang]
    if args.parent_csv != '':
        logger.info(f'parse from within an existing dataset at {args.parent_csv}')
        train, problemlist, problem2submission = read_existing_set(args.parent_csv, langs)
    else:
        train, problemlist, problem2submission = read_metadata(args.metadir, args.duplicates, excluded_problem_list, included_problem_list, lang=set(langs))
    image2id = collect_images(args.imgdir, train)
    # update_problem2submissions(image2id, train, problem2submission)
    available_submission_ids = set(image2id.values())
    train_filtered = dict()
    for subid, val in train.items():
        if subid in available_submission_ids:
            train_filtered[subid] = val
    train = train_filtered
    problem2submission = defaultdict(int)
    for _, datapoint in train.items():
        problem2submission[datapoint['problem_id']] += 1

    if args.minimum_submission > 0:
        train = filter_minimum_submissions(
            train, 
            problem2submission, 
            image2id,
            args.minimum_submission
        )
    if args.max_problems > 0 or args.problem_submissions > 0:
        train = limit_num_problems(train, problem2submission, args.max_problems, args.problem_submissions)
    
    with open(os.path.join(args.outdir, 'problem_id.lbl'), 'w') as fd:
        uniq_problem_ids = set()
        for v in train.values():
            uniq_problem_ids.add(v['problem_id'])
        for problem_id in uniq_problem_ids:
            print(problem_id, file=fd)

    if args.norm:
        for dataid, values in train.items():
            for point, problem in zip(
                ["cpu_time", "memory"], ["time_limit", "memory_limit"]
            ):
                values[point] = (
                    values[point] / problemlist[values["problem_id"]][problem]
                )

    val = None
    test = None
    n_heldout = args.val + args.test
    if n_heldout > 0:
        heldout = get_subset(train, n_heldout, problem2submission, not args.keep_dup)
        if args.test > 0:
            test = get_subset(heldout, args.test, problem2submission, not args.keep_dup)
        val = heldout

    dump_datasets(args.outdir, image2id, train, val, test)
