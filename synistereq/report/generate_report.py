import csv
import os
import numpy as np
import subprocess
from shutil import copyfile

from synistereq.utils import read_prediction_csv, write_predictions

def generate_report(prediction_csv, output_file):
    processed_predictions, n_skids = process_predictions(prediction_csv)
    latex_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latex")
    write_predictions(processed_predictions, os.path.join(latex_dir, "l1.csv") )

    # Read in the file
    bar_template = os.path.join(latex_dir, 'bar_template.tikz.tex')
    bar_dest = os.path.join(latex_dir, 'bar.tikz.tex')
 
    with open(bar_template, 'r') as f:
      filedata = f.read()

    # Replace the target string
    filedata = filedata.replace('replace', '\\foreach \\skididx in {{1,...,{}}}{{%'.format(n_skids))

    # Write the file out again
    with open(bar_dest, 'w') as f:
      f.write(filedata)

    build_ret = subprocess.check_call(f"cd {latex_dir}; pdflatex main.tex",
                                      shell = True)

    copyfile(os.path.join(latex_dir, "main.pdf"), output_file)


def process_predictions(prediction_csv):
    position_ids, skids, x, y, z, nts = read_prediction_csv(prediction_csv)
    nt_list = list(nts.keys())
    unique_skids = set(skids)
    skids_to_predictions = {skid: [0 for i in range(len(nt_list))] for skid in unique_skids}
    for i in range(len(position_ids)):
        skid = skids[i]
        nt_predictions = [p[i] for p in nts.values()]
        winner_idx = np.argmax(nt_predictions)
        skids_to_predictions[skid][winner_idx] = skids_to_predictions[skid][winner_idx] + 1

    # table
    # skid, nt1, ..., ntN
    table = []
    header = ["skid"] + nt_list
    table.append(header)
    for skid, pred in skids_to_predictions.items():
        row = [skid] + [v for v in pred]
        table.append(row)

    return table, len(unique_skids)
