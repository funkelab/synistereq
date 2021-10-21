import csv
import os
import numpy as np
import subprocess
import shutil
from shutil import copyfile

from synistereq.utils import read_prediction_csv, write_predictions

def generate_report(prediction_csv, output_file):

    # where to store the temporary latex files
    latex_dir = "./latex"
    os.makedirs(latex_dir, exist_ok=True)

    # copy latex templates into the temporary directory
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latex")
    file_names = os.listdir(source_dir)
    for file_name in file_names:                #copy .tex files into local directory
        shutil.copy(os.path.join(source_dir, file_name), latex_dir)

    # aggregate NT counts per neuron, write them to "l1.csv"
    processed_predictions, n_skids = process_predictions(prediction_csv)
    write_predictions(processed_predictions, os.path.join(latex_dir, "l1.csv"))

    # modify the latex templates
    bar_template = os.path.join(latex_dir, 'bar_template.tikz.tex')
    bar_dest = os.path.join(latex_dir, 'bar.tikz.tex')
    with open(bar_template, 'r') as f:
      filedata = f.read()
    filedata = filedata.replace('replace', '\\foreach \\skididx in {{1,...,{}}}{{%'.format(n_skids))
    with open(bar_dest, 'w') as f:
      f.write(filedata)

    # call latex to create the report PDF file
    build_ret = subprocess.check_call(f"cd {latex_dir}; pdflatex main.tex",
                                      shell = True)

    # copy the report to output_file
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
