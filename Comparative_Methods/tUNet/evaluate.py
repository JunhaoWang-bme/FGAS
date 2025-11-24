import nibabel as nib
import numpy as np
import os
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def calculate_metrics(pred, target, n_classes):
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    accuracy = accuracy_score(target_flat, pred_flat)

    precision, recall, f1, support = precision_recall_fscore_support(
        target_flat, pred_flat, average=None, labels=range(n_classes)
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average='macro'
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average='weighted'
    )

    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }


def evaluate_predictions(config):
    pred_dir = config['paths']['prediction_dir']
    label_dir = config['data']['data_dir']
    n_classes = config['model']['n_classes']

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('_pred.nii.gz')]
    all_metrics = []

    for pred_file in tqdm(pred_files):
        pred_path = os.path.join(pred_dir, pred_file)
        pred_nib = nib.load(pred_path)
        pred_data = pred_nib.get_fdata().astype(np.uint8)

        label_name = pred_file.replace('_pred.nii.gz', config['data']['label_suffix'])
        label_path = os.path.join(label_dir, label_name)

        if os.path.exists(label_path):
            label_nib = nib.load(label_path)
            label_data = label_nib.get_fdata().astype(np.uint8)

            label_data = np.clip(label_data, 0, n_classes - 1)

            metrics = calculate_metrics(pred_data, label_data, n_classes)
            all_metrics.append(metrics)

            print(f"{pred_file}: Precision={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")

    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if isinstance(all_metrics[0][key], np.ndarray):
                avg_metrics[key] = np.mean([m[key] for m in all_metrics], axis=0)
            else:
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics
    return None


def plot_results(metrics_history, save_path="evaluation_results.png"):
    epochs = list(range(1, len(metrics_history) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(epochs, [m['accuracy'] for m in metrics_history])
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True)
    axes[0, 1].plot(epochs, [m['macro_f1'] for m in metrics_history])
    axes[0, 1].set_title('Macro F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].grid(True)
    axes[1, 0].plot(epochs, [m['weighted_f1'] for m in metrics_history])
    axes[1, 0].set_title('Weighted F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True)

    if len(metrics_history) > 0:
        n_classes = len(metrics_history[0]['f1_per_class'])
        for i in range(n_classes):
            axes[1, 1].plot(epochs, [m['f1_per_class'][i] for m in metrics_history],
                            label=f'Class {i}')
        axes[1, 1].set_title('F1 Score per Class')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    config = load_config("config.yaml")
    metrics = evaluate_predictions(config)
    if metrics:
        print(f"\nOK")
        print(f"Precision: {metrics['accuracy']:.4f}")
        print(f"F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()