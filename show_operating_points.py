import argparse
import pandas as pd

# show best performing parameters exceeding threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm',)
    parser.add_argument(
        '--threshold',
        default=0.9,
        help='minimum recall',
        type=int)
    parser.add_argument(
        'csv',
        metavar='CSV',
        help='input csv')

    args = parser.parse_args()
    df = pd.read_csv(args.csv)

    if args.algorithm:
        algorithms = [args.algorithm]
    else:
        algorithms = set(df.algo.values)
    for algo in algorithms:
        print(f'show {algo}')
        if (len(df[(df.recall > args.threshold) & (df.algo == algo)].groupby(['algo', 'size']).min()[['querytime']])) == 0:
            print("didn't exceed recall, print highest recall:")
            print(df[(df.algo == algo)].groupby(['algo', 'size']).max()[['recall', 'querytime']])
    
        else:
            print(df[(df.recall > args.threshold) & (df.algo == algo)].groupby(['algo', 'size']).min()[['querytime']])

    print("Task A: Overview passing threshold")

    print(df[(df.recall > args.threshold)].groupby(['algo', 'size']).min()[['querytime']])

    print("Task C: Indexing binary vectors") 

    print(df[(df.data == "hammingv2")].sort_values(by=["size", "querytime"]).values)


