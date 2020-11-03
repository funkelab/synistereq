from synistereq.predict_neurotransmitters import predict_neurotransmitters
from synistereq.utils import read_skids_csv, read_positions_csv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--skids", 
		    help="File with skid column, pre synapses will be grabbed from service", 
		    default=None, required=False)
parser.add_argument("--positions", help="File with id, x,y,z columns representing synaptic locations", 
		    type=str, default=None, required=False)
parser.add_argument("--dataset", help="FAFB or HEMI", 
		    type=str, default=None, required=True)
parser.add_argument("--service", help="CATMAID, NEUPRINT, FLYWIRE", 
		    type=str, default=None, required=True)
parser.add_argument("--batch_size", help="FAFB or HEMI", 
		    type=int, default=24, required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    skids = None
    if args.skids is not None:
        skids = read_skids_csv(args.skids)

    positions = None
    position_ids = None
    if args.positions is not None:
        positions, position_ids = read_positions_csv(args.positions)

    nt_probabilities, positions, position_ids, ids_to_skids = predict_neurotransmitters(args.dataset,
                                                                                        args.service,
                                                                                        skids,
                                                                                        positions,
                                                                                        position_ids,
                                                                                        args.batch_size)

    print(nt_probabilities, positions, position_ids, ids_to_skids)


