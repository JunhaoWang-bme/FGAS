import os
import sys
import subprocess
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), 'nnunet'))
from postprocessing.largest_connected_components import process_all_masks


def run_nnunet_prediction(input_dir, output_dir, task_id, model, trainer, folds):
    command = [
        "nnUNet_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-t", str(task_id),
        "-m", model,
        "-tr", trainer,
        "-f", str(folds)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode == 0:
        print("nnUNet prediction completed")
        return True
    else:
        print(f"nnUNet prediction failed: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='nnUNet prediction and post-processing script')

    parser.add_argument('-i', '--input', required=True, help='Input data directory')
    parser.add_argument('-o', '--output', required=True, help='Prediction result directory')
    parser.add_argument('-t', '--task_id', required=True, type=int, help='Task ID')
    parser.add_argument('-m', '--model', default='3d_fullres', help='Model type (default: 3d_fullres)')
    parser.add_argument('-tr', '--trainer', default='UMDConsistencyTrainer',
                        help='Trainer (default: UMDConsistencyTrainer)')
    parser.add_argument('-f', '--folds', required=True, type=int, help='Number of folds')
    parser.add_argument('-c', '--connectivity', type=int, default=26, help='Connectivity (default: 26)')
    parser.add_argument('--no-postprocess', action='store_true', help='Skip post-processing step')

    args = parser.parse_args()

    prediction_success = run_nnunet_prediction(
        input_dir=args.input,
        output_dir=args.output,
        task_id=args.task_id,
        model=args.model,
        trainer=args.trainer,
        folds=args.folds
    )

    if prediction_success and not args.no_postprocess:
        print("Starting post-processing on prediction results...")
        process_all_masks(
            input_dir=args.output,
            output_dir=args.output,
            connectivity=args.connectivity
        )
        print("All processing completed!")
    elif args.no_postprocess:
        print("Skipping post-processing step")


if __name__ == "__main__":
    main()