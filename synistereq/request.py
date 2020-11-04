from synistereq.predict_neurotransmitters import predict_neurotransmitters
from synistereq.utils import read_skids_csv, read_positions_csv, format_predictions, write_predictions
from synistereq.report import generate_report
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
parser.add_argument("--out", help="Output file path", 
		    type=str, default="./predictions.csv", required=False)
parser.add_argument("--report", help="Report file path", 
		    type=str, default="./report.pdf", required=False)


if __name__ == "__main__":
    args = parser.parse_args()
    skids = None
    if args.skids is not None:
        skids = read_skids_csv(args.skids)

    positions = None
    position_ids = None
    if args.positions is not None:
        positions, position_ids, position_ids_to_skids = read_positions_csv(args.positions)


    nt_probabilities, positions, position_ids, position_ids_to_skids = predict_neurotransmitters(args.dataset,
                                                                                                args.service,
                                                                                                skids,
                                                                                                positions,
                                                                                                position_ids,
                                                                                                position_ids_to_skids,
                                                                                                args.batch_size)
    print(nt_probabilities)
    print(positions)
    print(position_ids)
    print(position_ids_to_skids)
    predictions = format_predictions(nt_probabilities,
                                     positions,
                                     position_ids,
                                     position_ids_to_skids)

    write_predictions(predictions, args.out)
    generate_report(args.out, args.report)
