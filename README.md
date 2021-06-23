# SYNISTEREQ
A library to request predictions and reports from a list of skids or locations from NEUPRINT, CATMAID and FLYWIRE.

## Installation
```console
git clone https://github.com/funkelab/synistereq.git
cd synistereq
conda create -n synistereq python=3.8 numpy scipy cython
pip install -r requirements.txt
pip install .
```

Replace the two placeholder checkpoints at synistereq/checkpoints with production network checkpoints (nils_data/synister_data/production_checkpoints):
```
fafb_checkpoint.placeholder -> fafb_checkpoint
hemi_checkpoint.placeholder -> hemi_checkpoint
```

Replace the two placeholder credentials at synistereq (nils_data/synister_data/credentials)
```
neuprint_credentials_placeholder.ini -> neuprint_credentials.ini
catmaid_credentials_placeholder.ini -> catmaid_credentials.ini
```

Update the file paths in `synistereq/checkpoint_paths.ini`

## Usage
In order to predict from a csv file containing skids or body ids, run request from the root directory via:
```console
python synistereq/request.py --skids <skids_file> --dataset <FAFB/HEMI> --service <CATMAID/NEUPRINT/FLYWIRE> --batch_size <8> --out <output_predictions_path> --report <output_report_path> --log <output_log_path>
```

Similarly for prediction from locations:
```console
python synistereq/request.py --positions <positions_file> --dataset <FAFB/HEMI> --service <CATMAID/NEUPRINT/FLYWIRE> --batch_size <8> --out <output_predictions_path> --report <output_report_path> --log <output_log_path>
```

A `<skids_file>` should contain only one column with header `skid` and integer ids for all skids to run in this one column. A position file should contain `id,x,y,z` columns for a unique synaptic integer id and the synaptic locations in the coordinate system of the chosen service. If no service is passed positions are assumed to be in container coordinates as stored on the Janelia servers for each dataset.

## TODO
See Issues
