import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Define the focus classes mapping (using indices from data.yaml)
FOCUS_CLASSES = {
    33: 'calculus',
    36: 'non-carious lesion',
    38: 'staining or visible changes without cavitation',
    42: 'visible changes with cavitation',
    43: 'visible changes with microcavitation'
}

def count_class_instances(labels_dir, focus_class_indices):
    """Count instances of focus classes in YOLO label files."""
    class_counts = defaultdict(int)

    if not os.path.exists(labels_dir):
        print(f"Warning: Directory not found: {labels_dir}")
        return class_counts

    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(labels_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in focus_class_indices:
                            class_counts[class_id] += 1

    return class_counts

def plot_class_distribution(train_counts, val_counts, test_counts, focus_classes):
    """Create a grouped bar chart for class distribution across splits."""
    class_indices = sorted(focus_classes.keys())
    class_names = [focus_classes[idx] for idx in class_indices]

    train_values = [train_counts.get(idx, 0) for idx in class_indices]
    val_values = [val_counts.get(idx, 0) for idx in class_indices]
    test_values = [test_counts.get(idx, 0) for idx in class_indices]
    total_values = [t + v + te for t, v, te in zip(train_values, val_values, test_values)]

    # Larger figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))

    # Thicker bars
    y = range(len(class_names))
    height = 0.35

    bars1 = ax1.barh([i - height for i in y], train_values, height,
                     label='Train', alpha=0.85, color='#2E86AB')
    bars2 = ax1.barh(y, val_values, height,
                     label='Validation', alpha=0.85, color='#A23B72')
    bars3 = ax1.barh([i + height for i in y], test_values, height,
                     label='Test', alpha=0.85, color='#F18F01')

    # Value labels (larger font)
    max_value = max(train_values + val_values + test_values) if (train_values + val_values + test_values) else 1
    offset = max_value * 0.01

    for bars, values in zip([bars1, bars2, bars3],
                            [train_values, val_values, test_values]):
        for bar, value in zip(bars, values):
            if value > 0:
                ax1.text(
                    value + offset,
                    bar.get_y() + bar.get_height() / 2,
                    str(value),
                    va='center',
                    ha='left',
                    fontsize=13,
                    fontweight='bold'
                )

    ax1.set_xlabel('Number of Instances', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Class', fontsize=16, fontweight='bold')
    ax1.set_title('Focus Classes Distribution by Dataset Split',
                  fontsize=18, fontweight='bold')

    ax1.set_yticks(y)
    ax1.set_yticklabels(class_names, fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.legend(fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Subplot 2: Total distribution
    ax2.barh(class_names, total_values, alpha=0.85, color='#06A77D')

    ax2.set_xlabel('Total Number of Instances', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Class', fontsize=16, fontweight='bold')
    ax2.set_title('Total Focus Classes Distribution',
                  fontsize=18, fontweight='bold')

    ax2.tick_params(axis='both', labelsize=14)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()

    # Value labels for subplot 2
    max_total = max(total_values) if total_values else 1
    offset_total = max_total * 0.01

    for i, v in enumerate(total_values):
        ax2.text(
            v + offset_total,
            i,
            str(v),
            va='center',
            ha='left',
            fontsize=13,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig('focus_classes_distribution.png',
                dpi=1500,
                bbox_inches='tight')
    print("\nChart saved as 'focus_classes_distribution.png' at 1500 DPI")

    plt.show()

def main():
    base_path = "/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/dataset"

    train_labels = os.path.join(base_path, "train", "labels")
    val_labels = os.path.join(base_path, "valid", "labels")
    test_labels = os.path.join(base_path.replace("dataset", "datasettest"), "labels")

    focus_class_indices = set(FOCUS_CLASSES.keys())

    print("Counting class instances...\n")
    print(f"Train:      {train_labels}")
    print(f"Validation: {val_labels}")
    print(f"Test:       {test_labels}\n")

    train_counts = count_class_instances(train_labels, focus_class_indices)
    val_counts = count_class_instances(val_labels, focus_class_indices)
    test_counts = count_class_instances(test_labels, focus_class_indices)

    print("=" * 60)
    print("FOCUS CLASSES DISTRIBUTION SUMMARY")
    print("=" * 60)

    for idx in sorted(FOCUS_CLASSES.keys()):
        name = FOCUS_CLASSES[idx]
        t = train_counts.get(idx, 0)
        v = val_counts.get(idx, 0)
        te = test_counts.get(idx, 0)

        print(f"\n{name} (Class {idx})")
        print(f"  Train:      {t}")
        print(f"  Validation: {v}")
        print(f"  Test:       {te}")
        print(f"  Total:      {t + v + te}")

    print("\n" + "=" * 60)

    plot_class_distribution(train_counts, val_counts, test_counts, FOCUS_CLASSES)

if __name__ == "__main__":
    main()
