import h5py
import numpy as np
import os
import csv
from pathlib import Path
from urllib.request import urlretrieve

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def get_groundtruth(size="100K"):
    url = f"http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge/public-queries/en-gold-standard-public/small-laion2B-en-public-gold-standard-{size}.h5"
    out_fn = os.path.join("data", f"groundtruth-{size}.h5")
    download(url, out_fn)
    return out_fn

def get_all_results(dirname):
    for root, _, files in os.walk(dirname):
        for fn in files:
            if os.path.splitext(fn)[-1] != ".h5":
                continue
            try:
                f = h5py.File(os.path.join(root, fn), "r")
                yield f
                f.close()
            except:
                print("Unable to read", fn)

def get_recall(I, gt, k):
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i]) & set(gt[i, :k]))
    return recall / (n * k)

if __name__ == "__main__":
    gt_fn = get_groundtruth()
    gt_f = h5py.File(gt_fn, "r")
    true_I = np.array(gt_f['knns'])
    gt_f.close()

    columns = ["data", "size", "algo", "buildtime", "querytime", "params", "recall"]
    
    with open('res.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for res in get_all_results("result"):
            recall = get_recall(np.array(res["knns"]), true_I, 30)
            d = dict(res.attrs)
            d['recall'] = recall
            print(d["data"], d["algo"], d["params"], "=>", recall)
            writer.writerow(d)