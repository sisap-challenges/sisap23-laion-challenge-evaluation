import argparse
import h5py
import numpy as np
import os
import csv
import glob
from pathlib import Path
from urllib.request import urlretrieve

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def get_groundtruth(size="100K", private=False):
    if private:
        # hard-coded at the moment
        out_fn = os.path.join("sisap23-mirror", "SISAP23-Challenge", f"laion2B-en-private-gold-standard-v2-{size}-F64-IEEE754.h5")
    else:
        #url = f"http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge/laion2B-en-public-gold-standard-v2-{size}.h5"
        url = f"https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-public-gold-standard-v2-{size}.h5"

        out_fn = os.path.join("data", f"groundtruth-{size}.h5")
        download(url, out_fn)
    gt_f = h5py.File(out_fn, "r")
    true_I = np.array(gt_f['knns'])
    gt_f.close()
    return true_I

def get_all_results(dirname):
    mask = [dirname + "/*/*/*.h5",
        dirname + "/*/result/*/*/*.h5",
        dirname + "/*/result/*/*/*/*.h5"]
    print("search for results matching:")
    print("\n".join(mask))
    for m in mask:
        for fn in glob.iglob(m):
            print(fn)
            f = h5py.File(fn, "r")
            if "knns" not in f or not ("size" in f or "size" in f.attrs):
                print("Ignoring " + fn)
                f.close()
                continue
            yield f
            f.close()

def get_recall(I, gt, k):
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)

def return_h5_str(f, param):
    if param not in f:
        return 0
    x = f[param][()]
    if type(x) == np.bytes_:
        return x.decode()
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        help='directory in which results are stored',
        default="results"
    )
    parser.add_argument(
        '--private',
        help="private queries held out for evaluation",
        action='store_true',
        default=False
    )
    parser.add_argument("csvfile")
    args = parser.parse_args()
    true_I_cache = {}
    test_sizes = ["10M", "30M", "100M"]


    columns = ["data", "size", "algo", "buildtime", "querytime", "params", "recall"]

    with open(args.csvfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for res in get_all_results(args.results):
            try:
                size = res.attrs["size"]
                d = dict(res.attrs)
            except:
                size = res["size"][()].decode()
                d = {k: return_h5_str(res, k) for k in columns}
            if size not in test_sizes:
                continue
            if size not in true_I_cache:
                true_I_cache[size] = get_groundtruth(size, args.private)
            recall = get_recall(np.array(res["knns"]), true_I_cache[size], 10)
            d['recall'] = recall
            print(d["data"], d["algo"], d["params"], "=>", recall)
            writer.writerow(d)
