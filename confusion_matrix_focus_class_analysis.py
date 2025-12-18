"""
YOLOv8 Segmentation Model Confusion Matrix Generator
Generates detailed confusion matrices at multiple confidence thresholds
For YOLOv8-seg models trained on segmentation datasets
"""

from ultralytics import YOLO
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json

# Configuration
MODEL_PATH = '/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/best.pt'
TEST_IMAGES = '/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/dataset/valid/images'
TEST_LABELS = '/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/dataset/valid/labels'

# Target classes mapping (based on your data.yaml)
CLASS_MAPPING = {
    'Background': 'BACKGROUND',  # Special marker for images with no objects
    'Staining': 38,  # 'staining or visible changes without cavitation'
    'Calculus': 33,  # 'calculus'
    'Micro cavitation': 43,  # 'visible changes with microcavitation'
    'Cavitation': 42,  # 'visible changes with cavitation'
    'Non-carious lesion': 36  # 'non-carious lesion'
}

# Full class names from data.yaml
ALL_CLASSES = ['11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23', '24', 
               '25', '26', '27', '28', '31', '32', '33', '34', '35', '36', '37', '38', 
               '41', '42', '43', '44', '45', '46', '47', '48', 'amalgam filling', 
               'calculus', 'fixed prosthesis', 'incisive papilla', 'non-carious lesion', 
               'palatine raphe', 'staining or visible changes without cavitation', 
               'temporary restoration', 'tongue', 'tooth coloured filling', 
               'visible changes with cavitation', 'visible changes with microcavitation']

CONFIDENCE_THRESHOLDS = [0.25, 0.5, 0.75]


def polygon_to_bbox(polygon_points):
    """Convert polygon points to bounding box [x_center, y_center, width, height]"""
    x_coords = [polygon_points[i] for i in range(0, len(polygon_points), 2)]
    y_coords = [polygon_points[i] for i in range(1, len(polygon_points), 2)]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_center, y_center, width, height]


def load_ground_truth_labels(label_path):
    """Load ground truth labels from YOLO segmentation format txt file"""
    labels = []
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:  # class + at least 3 points (6 coordinates)
                    class_id = int(parts[0])
                    polygon_points = list(map(float, parts[1:]))
                    
                    # Convert polygon to bounding box
                    bbox = polygon_to_bbox(polygon_points)
                    
                    labels.append({
                        'class_id': class_id,
                        'bbox': bbox,
                        'polygon': polygon_points
                    })
    return labels


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x_center, y_center, width, height] format"""
    # Convert to [x1, y1, x2, y2]
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def match_predictions_to_ground_truth(predictions, ground_truths, iou_threshold=0.5):
    """Match predictions to ground truth boxes using IoU"""
    matched_pairs = []
    unmatched_predictions = []
    unmatched_ground_truths = list(range(len(ground_truths)))
    
    # Sort predictions by confidence (highest first)
    pred_indices = sorted(range(len(predictions)), 
                         key=lambda i: predictions[i]['confidence'], 
                         reverse=True)
    
    for pred_idx in pred_indices:
        pred = predictions[pred_idx]
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in unmatched_ground_truths:
            gt = ground_truths[gt_idx]
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            matched_pairs.append({
                'pred_class': pred['class_id'],
                'gt_class': ground_truths[best_gt_idx]['class_id'],
                'iou': best_iou
            })
            unmatched_ground_truths.remove(best_gt_idx)
        else:
            unmatched_predictions.append(pred)
    
    # Unmatched ground truths are false negatives
    false_negatives = [ground_truths[i] for i in unmatched_ground_truths]
    
    return matched_pairs, unmatched_predictions, false_negatives


def analyze_confusion_matrix(conf_threshold):
    """Generate confusion matrix for specific classes at given confidence threshold"""
    print(f"\n{'='*80}")
    print(f"Analyzing at Confidence Threshold: {conf_threshold}")
    print(f"{'='*80}")
    
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Initialize counters for target classes
    target_class_ids = [v for v in CLASS_MAPPING.values() if v != 'BACKGROUND']
    correct_counts = defaultdict(int)
    misclassified_counts = defaultdict(int)
    
    # Special counters for background images
    background_correct = 0  # True Negatives: no GT objects, no predictions
    background_misclassified = 0  # False Positives: no GT objects, but model predicted something
    
    # Get all test images
    test_image_dir = Path(TEST_IMAGES)
    test_label_dir = Path(TEST_LABELS)
    image_files = list(test_image_dir.glob('*.jpg')) + list(test_image_dir.glob('*.png'))
    
    print(f"Found {len(image_files)} test images")
    
    # Process each image
    for img_idx, img_path in enumerate(image_files):
        if (img_idx + 1) % 50 == 0:
            print(f"Processing image {img_idx + 1}/{len(image_files)}...")
        
        # Load ground truth
        label_path = test_label_dir / (img_path.stem + '.txt')
        ground_truths = load_ground_truth_labels(label_path)
        
        # Check if this is a background image (no ground truth objects)
        is_background_image = len(ground_truths) == 0
        
        # Run prediction (segmentation model)
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        
        # Extract predictions from segmentation model
        predictions = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            masks = results[0].masks
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                # Get normalized bbox coordinates
                xywhn = boxes.xywhn[i].cpu().numpy()
                
                # Store prediction with bbox
                pred = {
                    'class_id': cls_id,
                    'confidence': conf,
                    'bbox': xywhn.tolist()
                }
                
                # Optionally store mask polygon if available
                if masks is not None and hasattr(masks, 'xy'):
                    try:
                        mask_polygon = masks.xy[i]
                        if len(mask_polygon) > 0:
                            pred['polygon'] = mask_polygon
                    except:
                        pass
                
                predictions.append(pred)
        
        # Handle background images
        if is_background_image:
            if len(predictions) == 0:
                # True Negative: no objects in GT, no predictions
                background_correct += 1
            else:
                # False Positive: no objects in GT, but model predicted something
                background_misclassified += 1
            continue  # Skip to next image
        
        # For non-background images, match predictions to ground truth
        matched, unmatched_preds, false_negs = match_predictions_to_ground_truth(
            predictions, ground_truths
        )
        
        # Count correct and misclassified for target classes
        for match in matched:
            pred_cls = match['pred_class']
            gt_cls = match['gt_class']
            
            # Only count if ground truth is one of our target classes
            if gt_cls in target_class_ids:
                if pred_cls == gt_cls:
                    correct_counts[gt_cls] += 1
                else:
                    misclassified_counts[gt_cls] += 1
        
        # False negatives (missed detections) count as misclassified
        for fn in false_negs:
            if fn['class_id'] in target_class_ids:
                misclassified_counts[fn['class_id']] += 1
    
    # Add background to results
    correct_counts['BACKGROUND'] = background_correct
    misclassified_counts['BACKGROUND'] = background_misclassified
    
    return correct_counts, misclassified_counts


def generate_report():
    """Generate complete confusion matrix report"""
    print("\n" + "="*80)
    print("YOLOv8 CONFUSION MATRIX ANALYSIS")
    print("="*80)
    
    results = {}
    
    for conf_threshold in CONFIDENCE_THRESHOLDS:
        correct_counts, misclassified_counts = analyze_confusion_matrix(conf_threshold)
        results[conf_threshold] = {
            'correct': correct_counts,
            'misclassified': misclassified_counts
        }
    
    # Print formatted tables
    print("\n\n" + "="*80)
    print("FINAL CONFUSION MATRIX RESULTS")
    print("="*80)
    
    # Create reverse mapping for display
    class_id_to_name = {v: k for k, v in CLASS_MAPPING.items() if v is not None}
    
    # Prepare data for CSV export
    csv_data = []
    
    for conf_threshold in CONFIDENCE_THRESHOLDS:
        print(f"\n{'─'*80}")
        print(f"Confidence Threshold: {conf_threshold}")
        print(f"{'─'*80}")
        
        # Create table
        table_data = []
        
        # Correct row
        correct_row = ['Correct']
        for class_name, class_id in CLASS_MAPPING.items():
            if class_name == 'Background':
                count = results[conf_threshold]['correct'].get('BACKGROUND', 0)
            else:
                count = results[conf_threshold]['correct'].get(class_id, 0)
            correct_row.append(count)
        table_data.append(correct_row)
        
        # Misclassified row
        misclass_row = ['Misclassified']
        for class_name, class_id in CLASS_MAPPING.items():
            if class_name == 'Background':
                count = results[conf_threshold]['misclassified'].get('BACKGROUND', 0)
            else:
                count = results[conf_threshold]['misclassified'].get(class_id, 0)
            misclass_row.append(count)
        table_data.append(misclass_row)
        
        # Print as DataFrame for better formatting
        columns = [''] + list(CLASS_MAPPING.keys())
        df = pd.DataFrame(table_data, columns=columns)
        print(df.to_string(index=False))
        
        # Calculate and print accuracy for each class
        print(f"\nPer-class Accuracy:")
        for class_name, class_id in CLASS_MAPPING.items():
            if class_name == 'Background':
                correct = results[conf_threshold]['correct'].get('BACKGROUND', 0)
                misclass = results[conf_threshold]['misclassified'].get('BACKGROUND', 0)
            else:
                correct = results[conf_threshold]['correct'].get(class_id, 0)
                misclass = results[conf_threshold]['misclassified'].get(class_id, 0)
            total = correct + misclass
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"  {class_name:20s}: {correct:4d}/{total:4d} = {accuracy:6.2f}%")
            
            # Add to CSV data
            csv_data.append({
                'Confidence_Threshold': conf_threshold,
                'Class': class_name,
                'Correct': correct,
                'Misclassified': misclass,
                'Total': total,
                'Accuracy_%': round(accuracy, 2)
            })
    
    # Save results to CSV
    csv_df = pd.DataFrame(csv_data)
    csv_filename = 'confusion_matrix_results.csv'
    csv_df.to_csv(csv_filename, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_filename}")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    # Run the analysis
    results = generate_report()
    
    print("\n" + "="*80)
    print("REPRODUCTION INSTRUCTIONS")
    print("="*80)
    print("""
To reproduce these results:

1. Install required packages:
   pip install ultralytics pandas numpy

2. Ensure your dataset structure matches:
   dataset/
   ├── test/
   │   ├── images/  (test images)
   │   └── labels/  (YOLO segmentation format .txt files)

3. Verify your model:
   - Must be a YOLOv8 segmentation model (best.pt from yolov8s-seg training)
   - Trained on segmentation data with polygon annotations

4. Update paths in the script:
   - MODEL_PATH: path to your best.pt model
   - TEST_IMAGES: path to test/images directory
   - TEST_LABELS: path to test/labels directory

5. Run the script:
   python confusion_matrix_generator.py

6. The script will:
   - Load your trained YOLOv8-seg model
   - Run segmentation inference on test images at each confidence threshold
   - Extract bounding boxes from both predictions and ground truth polygons
   - Match predictions to ground truth using IoU ≥ 0.5
   - Count correct and misclassified predictions for each target class
   - Generate formatted confusion matrix tables

Notes:
- Works with YOLOv8 segmentation models (yolov8n-seg, yolov8s-seg, etc.)
- Ground truth format: class x1 y1 x2 y2 ... xn yn (polygon coordinates)
- Predictions use bounding boxes extracted from segmentation masks
- Correct = predicted class matches ground truth (TP)
- Misclassified = predicted wrong class (FP) or missed detection (FN)
- Only evaluates the 6 specified target classes
- Uses IoU threshold of 0.5 for matching predictions to ground truth
- Both predictions and ground truth polygons are converted to bounding boxes for matching
""")