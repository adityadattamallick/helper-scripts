import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import cv2

def get_yolov8seg_predictions(model_path, data_yaml, conf_threshold=0.25, iou_threshold=0.45, split='val'):
    """
    Get YOLOv8 segmentation predictions for evaluation
    
    Args:
        model_path: Path to trained YOLOv8 segmentation model (.pt file)
        data_yaml: Path to dataset yaml file
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
        split: Dataset split to evaluate ('val' or 'test')
    
    Returns:
        conf_matrix: Confusion matrix
        class_names: Dictionary of class names
    """
    print(f"Loading YOLOv8 Segmentation model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running validation on {split} set...")
    # Run validation to get predictions
    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=True,
        plots=True,  # This will generate confusion matrix plot
        save_json=False
    )
    
    # Extract confusion matrix from results
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        conf_matrix = results.confusion_matrix.matrix
        class_names = results.names
        print(f"\nConfusion matrix shape: {conf_matrix.shape}")
        print(f"Number of class names: {len(class_names)}")
        
        # Handle mismatch: YOLOv8 often adds background class (last row/col)
        if conf_matrix.shape[0] > len(class_names):
            print(f"Matrix has extra row/column (likely background class)")
            print(f"Trimming matrix from {conf_matrix.shape[0]}x{conf_matrix.shape[1]} to {len(class_names)}x{len(class_names)}")
            # Remove the last row and column (background class)
            conf_matrix = conf_matrix[:len(class_names), :len(class_names)]
        
        return conf_matrix, class_names
    else:
        print("Warning: Could not extract confusion matrix from validation results")
        return None, None


def extract_predictions_manually(model_path, data_yaml, split='val', conf_threshold=0.25):
    """
    Manually extract predictions for segmentation model by running inference
    and matching with ground truth labels
    
    Args:
        model_path: Path to trained model
        data_yaml: Path to dataset yaml
        split: 'val' or 'test'
        conf_threshold: Confidence threshold
    
    Returns:
        y_true: List of ground truth class labels
        y_pred: List of predicted class labels
    """
    import yaml
    
    model = YOLO(model_path)
    
    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get image directory based on split
    if split == 'val':
        img_dir = Path(data_config['val'])
    elif split == 'test':
        img_dir = Path(data_config['test'])
    else:
        img_dir = Path(data_config['train'])
    
    # Get label directory (assumes labels are in parallel directory)
    label_dir = img_dir.parent.parent / 'labels' / img_dir.parent.name
    if not label_dir.exists():
        label_dir = img_dir.parent / 'labels'
    
    print(f"Image directory: {img_dir}")
    print(f"Label directory: {label_dir}")
    
    y_true = []
    y_pred = []
    
    # Get all images
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    print(f"Found {len(image_files)} images")
    
    for img_path in image_files:
        # Get corresponding label file
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            continue
        
        # Read ground truth labels
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:  # class_id + at least one coordinate pair
                    true_class = int(parts[0])
                    y_true.append(true_class)
        
        # Run inference
        results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        
        # Extract predicted classes
        if results and results[0].masks is not None:
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)
            for pred_class in pred_classes:
                y_pred.append(pred_class)
        
    print(f"Collected {len(y_true)} ground truth labels and {len(y_pred)} predictions")
    
    return np.array(y_true), np.array(y_pred)


def calculate_cohens_kappa_from_matrix(conf_matrix):
    """
    Calculate Cohen's Kappa directly from confusion matrix
    
    Args:
        conf_matrix: NxN confusion matrix
    
    Returns:
        kappa_score: Cohen's Kappa score
    """
    n = conf_matrix.sum()
    
    if n == 0:
        return 0.0
    
    # Observed agreement (accuracy)
    p_o = np.trace(conf_matrix) / n
    
    # Expected agreement
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)
    p_e = np.sum(row_sums * col_sums) / (n * n)
    
    # Cohen's Kappa
    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def calculate_cohens_kappa_from_predictions(y_true, y_pred, num_classes=44):
    """
    Calculate Cohen's Kappa from prediction arrays
    
    Args:
        y_true: Array of ground truth labels
        y_pred: Array of predicted labels
        num_classes: Number of classes
    
    Returns:
        kappa_score: Cohen's Kappa score
        conf_matrix: Confusion matrix
    """
    # Handle length mismatch
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate confusion matrix
    labels = list(range(num_classes))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate Cohen's Kappa
    kappa_score = cohen_kappa_score(y_true, y_pred, labels=labels)
    
    return kappa_score, conf_matrix


def print_kappa_table(conf_matrix, class_names, kappa_score, top_n=10):
    """
    Print formatted Cohen's Kappa confusion matrix table
    For large matrices (44 classes), show summary statistics
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        kappa_score: Cohen's Kappa score
        top_n: Number of top confused pairs to show
    """
    print("\n" + "="*80)
    print("COHEN'S KAPPA CONFUSION MATRIX - YOLOv8 SEGMENTATION")
    print("="*80)
    
    # Basic statistics
    n_classes = len(class_names)
    total_predictions = conf_matrix.sum()
    correct_predictions = np.trace(conf_matrix)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"\nDataset Statistics:")
    print(f"  Number of classes: {n_classes}")
    print(f"  Total predictions: {int(total_predictions)}")
    print(f"  Correct predictions: {int(correct_predictions)}")
    print(f"  Overall accuracy: {accuracy:.4f}")
    
    print("\n" + "-"*80)
    print(f"Cohen's Kappa Score: {kappa_score:.4f}")
    print("-"*80)
    
    # Interpretation
    print("\nInterpretation:")
    if kappa_score < 0:
        print("Poor agreement (worse than random)")
    elif kappa_score < 0.20:
        print("Slight agreement")
    elif kappa_score < 0.40:
        print("Fair agreement")
    elif kappa_score < 0.60:
        print("Moderate agreement")
    elif kappa_score < 0.80:
        print("Substantial agreement")
    else:
        print("Almost perfect agreement")
    
    # Per-class accuracy
    print("\n" + "-"*80)
    print("Per-Class Performance (Top 10 and Bottom 10):")
    print("-"*80)
    
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        total_true = conf_matrix[i, :].sum()
        if total_true > 0:
            class_acc = conf_matrix[i, i] / total_true
            class_accuracies.append((class_name, class_acc, int(total_true), int(conf_matrix[i, i])))
    
    # Sort by accuracy
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Best Performing Classes:")
    print(f"{'Class':<40} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
    print("-"*70)
    for class_name, acc, total, correct in class_accuracies[:top_n]:
        print(f"{class_name:<40} {acc:.4f}    {correct:<10} {total:<10}")
    
    print("\nBottom 10 Worst Performing Classes:")
    print(f"{'Class':<40} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
    print("-"*70)
    for class_name, acc, total, correct in class_accuracies[-top_n:]:
        print(f"{class_name:<40} {acc:.4f}    {correct:<10} {total:<10}")
    
    # Most confused pairs
    print("\n" + "-"*80)
    print(f"Top {top_n} Most Confused Class Pairs:")
    print("-"*80)
    
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and conf_matrix[i, j] > 0:
                confused_pairs.append((
                    class_names[i],
                    class_names[j],
                    int(conf_matrix[i, j])
                ))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'True Class':<30} {'Predicted As':<30} {'Count':<10}")
    print("-"*70)
    for true_class, pred_class, count in confused_pairs[:top_n]:
        print(f"{true_class:<30} {pred_class:<30} {count:<10}")
    
    print("\n" + "="*80 + "\n")


def save_kappa_results(conf_matrix, class_names, kappa_score, output_path="kappa_results.txt"):
    """
    Save Cohen's Kappa results to file
    """
    # Ensure matrix and class names match
    if conf_matrix.shape[0] != len(class_names):
        print(f"Warning: Matrix size ({conf_matrix.shape[0]}) doesn't match class names ({len(class_names)})")
        # Trim to match
        n = min(conf_matrix.shape[0], len(class_names))
        conf_matrix = conf_matrix[:n, :n]
        class_names = class_names[:n]
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COHEN'S KAPPA CONFUSION MATRIX - YOLOv8 SEGMENTATION\n")
        f.write("="*80 + "\n\n")
        
        # Full confusion matrix
        f.write("Full Confusion Matrix:\n")
        f.write("-"*80 + "\n")
        df = pd.DataFrame(
            conf_matrix,
            index=[f"True_{i}_{name}" for i, name in enumerate(class_names)],
            columns=[f"Pred_{i}_{name}" for i, name in enumerate(class_names)]
        )
        f.write(df.to_string())
        
        f.write("\n\n" + "-"*80 + "\n")
        f.write(f"Cohen's Kappa Score: {kappa_score:.4f}\n")
        f.write("-"*80 + "\n")
        
        # Per-class statistics
        f.write("\n\nPer-Class Statistics:\n")
        f.write("-"*80 + "\n")
        for i, class_name in enumerate(class_names):
            if i < conf_matrix.shape[0]:
                total_true = conf_matrix[i, :].sum()
                if total_true > 0:
                    class_acc = conf_matrix[i, i] / total_true
                    f.write(f"{class_name}: Accuracy={class_acc:.4f}, Total={int(total_true)}\n")
    
    print(f"✓ Results saved to {output_path}")


def save_confusion_matrix_csv(conf_matrix, class_names, output_path="confusion_matrix.csv"):
    """
    Save confusion matrix as CSV for easy import into Excel/Sheets
    """
    # Ensure matrix and class names match
    if conf_matrix.shape[0] != len(class_names):
        print(f"Warning: Matrix size ({conf_matrix.shape[0]}) doesn't match class names ({len(class_names)})")
        n = min(conf_matrix.shape[0], len(class_names))
        conf_matrix = conf_matrix[:n, :n]
        class_names = class_names[:n]
    
    df = pd.DataFrame(
        conf_matrix,
        index=class_names,
        columns=class_names
    )
    df.to_csv(output_path)
    print(f"Confusion matrix saved to {output_path}")


# Main execution
if __name__ == "__main__":
    # Configuration
    model_path = "/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/best.pt"  # Update with your model path
    data_yaml = "/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/dataset/data.yaml"
    
    print("="*80)
    print("YOLOv8 SEGMENTATION MODEL - COHEN'S KAPPA EVALUATION")
    print("="*80)
    print(f"\nModel: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Number of classes: 44")
    print("\nThis may take a few minutes...\n")
    
    # Method 1: Try to get confusion matrix from validation (fastest)
    conf_matrix, class_names = get_yolov8seg_predictions(
        model_path=model_path,
        data_yaml=data_yaml,
        conf_threshold=0.25,
        iou_threshold=0.45,
        split='val'  # Change to 'test' if you want to evaluate on test set
    )
    
    if conf_matrix is not None and conf_matrix.sum() > 0:
        # Calculate Cohen's Kappa
        kappa_score = calculate_cohens_kappa_from_matrix(conf_matrix)
        
        # Print results
        print_kappa_table(conf_matrix, list(class_names.values()), kappa_score)
        
        # Save results
        save_kappa_results(conf_matrix, list(class_names.values()), kappa_score)
        save_confusion_matrix_csv(conf_matrix, list(class_names.values()))
        
    else:
        print("\nValidation method failed. Trying manual extraction.")
        print("This will take longer as it processes each image individually.\n")
        
        # Method 2: Manual extraction (slower but more reliable)
        y_true, y_pred = extract_predictions_manually(
            model_path=model_path,
            data_yaml=data_yaml,
            split='val',
            conf_threshold=0.25
        )
        
        if len(y_true) > 0 and len(y_pred) > 0:
            # Load class names from yaml
            import yaml
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            class_names = data_config['names']
            
            kappa_score, conf_matrix = calculate_cohens_kappa_from_predictions(
                y_true, y_pred, num_classes=44
            )
            
            print_kappa_table(conf_matrix, class_names, kappa_score)
            save_kappa_results(conf_matrix, class_names, kappa_score)
            save_confusion_matrix_csv(conf_matrix, class_names)
        else:
            print("Error: Could not extract predictions. Please check your paths.")