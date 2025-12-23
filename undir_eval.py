import os
import pandas as pd
import numpy as np
import csv
import argparse


datasets = ['icews18_min', 'icews18_mid', 'icews18_max', 'cora', 'citeseer', 'pubmed', 'internet', 'ro', 'kegg', 'ogbl_collab', 'ogbl_ppa']

turns = [1, 2, 3, 4, 5]
split_ratios = {
    'cora': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'citeseer': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'pubmed': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'icews18_min': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'icews18_mid': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'icews18_max': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'internet': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'ro': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'ogbl_ppa': [0.02],
    'ogbl_collab': [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}


for dataset_name in datasets:
    print(f"Processing dataset: {dataset_name}")
    test_results = []
    roc_aucs = np.zeros((len(split_ratios[dataset_name]), len(turns)))
    pr_aucs = np.zeros((len(split_ratios[dataset_name]), len(turns)))
    f1s = np.zeros((len(split_ratios[dataset_name]), len(turns)))
    hits20s = np.zeros((len(split_ratios[dataset_name]), len(turns)))
    hits50s = np.zeros((len(split_ratios[dataset_name]), len(turns)))
    hits100s = np.zeros((len(split_ratios[dataset_name]), len(turns)))

    for j, ratio in enumerate(split_ratios[dataset_name]):
        for k, turn in enumerate(turns):
            result_filename = f"{dataset_name}-r{ratio}-t{turn}-speedup.csv"
            result_path = os.path.join('./results', result_filename)

            if not os.path.exists(result_path):
                print(f"File {result_path} not found, skipping.")
                continue

            try:
                df = pd.read_csv(result_path)

                roc_auc = df['roc_auc'].iloc[0]
                pr_auc = df['pr_auc'].iloc[0]
                f1 = df['f1'].iloc[0]
                hits20 = df['hits20'].iloc[0]
                hits50 = df['hits50'].iloc[0]
                hits100 = df['hits100'].iloc[0]

                roc_aucs[j][k] = roc_auc
                pr_aucs[j][k] = pr_auc
                f1s[j][k] = f1
                hits20s[j][k] = hits20
                hits50s[j][k] = hits50
                hits100s[j][k] = hits100
                
                print(f"Finished: Dataset: {dataset_name}, Ratio: {ratio}, Turn: {turn}")

            except Exception as e:
                print(f"Error processing file {result_path}: {e}")

    roc_aucs_mean, roc_aucs_std = roc_aucs.mean(axis=1), roc_aucs.std(axis=1)
    pr_aucs_mean, pr_aucs_std = pr_aucs.mean(axis=1), pr_aucs.std(axis=1)
    f1s_mean, f1s_std = f1s.mean(axis=1), f1s.std(axis=1)
    hits20s_mean, hits20s_std = hits20s.mean(axis=1), hits20s.std(axis=1)
    hits50s_mean, hits50s_std = hits50s.mean(axis=1), hits50s.std(axis=1)
    hits100s_mean, hits100s_std = hits100s.mean(axis=1), hits100s.std(axis=1)

    for i, ratio in enumerate(split_ratios[dataset_name]):
        test_results.append({
            "split_ratio": ratio,
            "roc_auc": f"{roc_aucs_mean[i] * 100:.2f}±{roc_aucs_std[i] * 100:.2f}",
            "pr_auc": f"{pr_aucs_mean[i] * 100:.2f}±{pr_aucs_std[i] * 100:.2f}",
            "f1": f"{f1s_mean[i] * 100:.2f}±{f1s_std[i] * 100:.2f}",
            "hits20": f"{hits20s_mean[i] * 100:.2f}±{hits20s_std[i] * 100:.2f}",
            "hits50": f"{hits50s_mean[i] * 100:.2f}±{hits50s_std[i] * 100:.2f}",
            "hits100": f"{hits100s_mean[i] * 100:.2f}±{hits100s_std[i] * 100:.2f}",
        })

    output_csv = f"./logs/results/test_results_{dataset_name}.csv"
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split_ratio", "roc_auc", "pr_auc", "f1",
                                                      "hits20", "hits50", "hits100"])
        writer.writeheader()
        writer.writerows(test_results)

    print(f"Results saved to {output_csv}.")
