# This is based on https://github.com/matsui528/annbench/blob/main/plot.py
import argparse
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
from itertools import cycle


marker = cycle(('p', '^', 'h', 'x', 'o', 's', '*', '+', 'D', '1', 'X')) 
linestyle = cycle((':', '-', '--'))

def draw(lines, xlabel, ylabel, title, filename, with_ctrl, width, height):
    """
    Visualize search results and save them as an image
    Args:
        lines (list): search results. list of dict.
        xlabel (str): label of x-axis, usually "recall"
        ylabel (str): label of y-axis, usually "query per sec"
        title (str): title of the result_img
        filename (str): output file name of image
        with_ctrl (bool): show control parameters or not
        width (int): width of the figure
        height (int): height of the figure
    """
    plt.figure(figsize=(width, height))

    for line in lines:
        for key in ["xs", "ys", "label", "ctrls"]:
            assert key in line

    for line in lines:
        plt.plot(line["xs"], line["ys"], label=line["label"], marker=next(marker), linestyle=next(linestyle))
        if with_ctrl:
            for x, y, ctrl in zip(line["xs"], line["ys"], line["ctrls"]):
                plt.annotate(text=str(ctrl), xy=(x, y),
                             xytext=(x, y+50))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which="both")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.cla()

def get_pareto_frontier(line):
    data = sorted(zip(line["ys"], line["xs"], line["ctrls"]),reverse=True)
    line["xs"] = []
    line["ys"] = []
    line["ctrls"] = []

    cur = 0
    for y, x, label in data:
        if x > cur:
            cur = x
            line["xs"].append(x)
            line["ys"].append(y)
            line["ctrls"].append(label)

    return line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        default="100K"
    )
    parser.add_argument("csvfile")
    args = parser.parse_args()
    
    with open(args.csvfile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    
    lines = {}
    for res in data:
        if res["size"] != args.size:
            continue
        dataset = res["data"]
        algo = res["algo"]
        label = dataset + algo
        if label not in lines:
            lines[label] =  {
                "xs": [],
                "ys": [],
                "ctrls": [],
                "label": label,
            }
        lines[label]["xs"].append(float(res["recall"]))
        lines[label]["ys"].append(10000/float(res["querytime"])) # FIX query size hardcoded
        try:
            run_identifier = res["params"].split("query=")[1]
        except:
            run_identifier = res["params"]
        lines[label]["ctrls"].append(run_identifier)
    
    draw([get_pareto_frontier(line) for line in lines.values()], 
            "Recall", "QPS (1/s)", "Result", f"result_{args.size}.png", True, 10, 8)
