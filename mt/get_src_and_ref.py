import argparse
import numpy as np
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasetpath", type=str)
    parser.add_argument("srcpath", type=str)
    parser.add_argument("tgtpath", type=str)
    args = parser.parse_args()

    # load sources and references with pandas
    dataset_df = pd.read_csv(args.datasetpath, sep="\t")
    refs = dataset_df["targetString"].to_list()
    srcs = dataset_df["sourceString"].to_list()

    # save as txt file
    output_file = open(args.tgtpath, "w", encoding="utf-8")
    for line in refs:
        output_file.write(line.replace("\n","\\n"))
        output_file.write('\n')

    # save as txt file
    output_file = open(args.srcpath, "w", encoding="utf-8")
    for line in srcs:
        output_file.write(line.replace("\n","\\n"))
        output_file.write('\n')


if __name__ == "__main__":
    main()