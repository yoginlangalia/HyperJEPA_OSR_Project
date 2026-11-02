#!/usr/bin/env python3
"""
Prediction Visualization Script for Houston Dataset
====================================================
This script generates annotated images where each pixel is colored
according to its predicted class label.

Usage:
    python visualize_predictions.py <csv_file> [--output <output_path>]

Example:
    python visualize_predictions.py GGCL/inference/Houston_all_pixel_predictions.csv
    python visualize_predictions.py DMVL/Inference/Houston_all_pixel_predictions.csv --output DMVL_prediction_map.png

Color Mapping for Houston Dataset (15 classes):
================================================
Class  0: Healthy Grass    -> #32CD32 (Bright Lime Green)
Class  1: Stressed Grass   -> #228B22 (Medium Green)
Class  2: Synthetic Grass  -> #00CED1 (Cyan / Turquoise)
Class  3: Trees            -> #006400 (Dark Green)
Class  4: Soil             -> #D2B48C (Tan / Beige)
Class  5: Water            -> #87CEEB (Light Blue / Sky Blue)
Class  6: Residential      -> #FFA500 (Orange)
Class  7: Commercial       -> #FF0000 (Red)
Class  8: Road             -> #800080 (Purple / Magenta)
Class  9: Highway          -> #800000 (Dark Red / Maroon)
Class 10: Railway          -> #808069 (Gray-Brown)
Class 11: Parking Lot 1    -> #FFFF00 (Yellow)
Class 12: Parking Lot 2    -> #808000 (Olive / Dark Yellowish-Green)
Class 13: Tennis Court     -> #008080 (Teal / Dark Cyan)
Class 14: Running Track    -> #4A0000 (Dark Maroon / Brown-Red)

Notes:
- The script automatically detects the image dimensions from the data
- Supports both CSV and NPZ file formats
- Output is saved as PNG with the same name as input (or custom output path)
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
import sys

# ============================================================================
# COLOR MAPPING CONFIGURATION
# ============================================================================
# Class Index -> RGB Color Tuple (R, G, B)
# Modify these colors as needed for different datasets/preferences

CLASS_COLORS = {
    0:  (50, 205, 50),     # Healthy Grass    -> Bright Lime Green
    1:  (34, 139, 34),     # Stressed Grass   -> Medium Green
    2:  (0, 206, 209),     # Synthetic Grass  -> Cyan / Turquoise
    3:  (0, 100, 0),       # Trees            -> Dark Green
    4:  (210, 180, 140),   # Soil             -> Tan / Beige
    5:  (135, 206, 235),   # Water            -> Light Blue / Sky Blue
    6:  (255, 165, 0),     # Residential      -> Orange
    7:  (255, 0, 0),       # Commercial       -> Red
    8:  (128, 0, 128),     # Road             -> Purple / Magenta
    9:  (128, 0, 0),       # Highway          -> Dark Red / Maroon
    10: (128, 128, 105),   # Railway          -> Gray-Brown
    11: (255, 255, 0),     # Parking Lot 1    -> Yellow
    12: (128, 128, 0),     # Parking Lot 2    -> Olive / Dark Yellowish-Green
    13: (0, 128, 128),     # Tennis Court     -> Teal / Dark Cyan
    14: (74, 0, 0),        # Running Track    -> Dark Maroon / Brown-Red
}

CLASS_NAMES = {
    0:  "Healthy Grass",
    1:  "Stressed Grass",
    2:  "Synthetic Grass",
    3:  "Trees",
    4:  "Soil",
    5:  "Water",
    6:  "Residential",
    7:  "Commercial",
    8:  "Road",
    9:  "Highway",
    10: "Railway",
    11: "Parking Lot 1",
    12: "Parking Lot 2",
    13: "Tennis Court",
    14: "Running Track",
}
# ============================================================================


def load_predictions_csv(csv_path):
    """
    Load predictions from CSV file.
    Expected format: row, col, predicted_class_index, predicted_class_name
    
    Returns:
        numpy array of shape (height, width) containing class indices
    """
    print(f"Loading predictions from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Determine image dimensions from data
    max_row = df['row'].max()
    max_col = df['col'].max()
    height = max_row + 1
    width = max_col + 1
    
    print(f"  Detected image dimensions: {height} x {width} ({height * width} pixels)")
    
    # Create prediction map
    prediction_map = np.zeros((height, width), dtype=np.int32)
    
    # Fill in predictions
    rows = df['row'].values
    cols = df['col'].values
    predictions = df['predicted_class_index'].values
    
    prediction_map[rows, cols] = predictions
    
    # Get class distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\n  Class distribution:")
    for cls, count in zip(unique, counts):
        name = CLASS_NAMES.get(cls, f"Unknown({cls})")
        pct = 100.0 * count / len(predictions)
        print(f"    Class {cls:2d} ({name:15s}): {count:7d} pixels ({pct:5.2f}%)")
    
    return prediction_map


def load_predictions_npz(npz_path):
    """
    Load predictions from NPZ file.
    Expected to contain 'predictions' or similar array.
    
    Returns:
        numpy array of shape (height, width) containing class indices
    """
    print(f"Loading predictions from NPZ: {npz_path}")
    data = np.load(npz_path)
    
    # Try to find the predictions array
    print(f"  Available keys: {list(data.keys())}")
    
    if 'predictions' in data:
        predictions = data['predictions']
    elif 'predicted_labels' in data:
        predictions = data['predicted_labels']
    else:
        # Use the first array found
        key = list(data.keys())[0]
        predictions = data[key]
        print(f"  Using key '{key}' for predictions")
    
    # If predictions is 1D, we need row/col info
    if predictions.ndim == 1:
        if 'rows' in data and 'cols' in data:
            rows = data['rows']
            cols = data['cols']
            height = rows.max() + 1
            width = cols.max() + 1
            prediction_map = np.zeros((height, width), dtype=np.int32)
            prediction_map[rows, cols] = predictions
        else:
            raise ValueError("1D predictions found but no row/col arrays available")
    elif predictions.ndim == 2:
        prediction_map = predictions.astype(np.int32)
    else:
        raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
    
    print(f"  Prediction map shape: {prediction_map.shape}")
    
    # Get class distribution
    unique, counts = np.unique(prediction_map, return_counts=True)
    print(f"\n  Class distribution:")
    for cls, count in zip(unique, counts):
        name = CLASS_NAMES.get(cls, f"Unknown({cls})")
        pct = 100.0 * count / prediction_map.size
        print(f"    Class {cls:2d} ({name:15s}): {count:7d} pixels ({pct:5.2f}%)")
    
    return prediction_map


def create_colored_image(prediction_map, colors=None):
    """
    Create a colored RGB image from the prediction map.
    
    Args:
        prediction_map: 2D numpy array of class indices
        colors: Optional dict mapping class index to RGB tuple
    
    Returns:
        PIL Image with colored pixels
    """
    if colors is None:
        colors = CLASS_COLORS
    
    height, width = prediction_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Apply colors for each class
    for class_idx, color in colors.items():
        mask = (prediction_map == class_idx)
        rgb_image[mask] = color
    
    # Handle any unknown classes (use black)
    known_classes = set(colors.keys())
    unique_classes = set(np.unique(prediction_map))
    unknown_classes = unique_classes - known_classes
    
    if unknown_classes:
        print(f"  Warning: Unknown classes found: {unknown_classes} (colored black)")
        for cls in unknown_classes:
            mask = (prediction_map == cls)
            rgb_image[mask] = (0, 0, 0)
    
    return Image.fromarray(rgb_image)


def create_legend_image(colors=None, class_names=None):
    """
    Create a legend image showing all class colors and names.
    
    Args:
        colors: Dict mapping class index to RGB tuple
        class_names: Dict mapping class index to class name
    
    Returns:
        PIL Image with the legend
    """
    if colors is None:
        colors = CLASS_COLORS
    if class_names is None:
        class_names = CLASS_NAMES
    
    # Simple legend: colored squares with class names
    from PIL import ImageDraw, ImageFont
    
    num_classes = len(colors)
    box_size = 20
    padding = 5
    text_width = 150
    row_height = box_size + padding
    
    legend_width = box_size + padding + text_width + padding
    legend_height = num_classes * row_height + padding
    
    legend = Image.new('RGB', (legend_width, legend_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(legend)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, (class_idx, color) in enumerate(sorted(colors.items())):
        y = padding + i * row_height
        # Draw color box
        draw.rectangle([padding, y, padding + box_size, y + box_size], fill=color, outline=(0, 0, 0))
        # Draw class name
        name = class_names.get(class_idx, f"Class {class_idx}")
        draw.text((padding + box_size + padding, y + 2), f"{class_idx}: {name}", fill=(0, 0, 0), font=font)
    
    return legend


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated prediction images for Houston dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_predictions.py GGCL/inference/Houston_all_pixel_predictions.csv
  python visualize_predictions.py DMVL/Inference/Houston_all_pixel_predictions.csv --output custom_output.png
  python visualize_predictions.py predictions.npz --with-legend
        """
    )
    
    parser.add_argument('input_file', type=str,
                        help='Path to input file (CSV or NPZ format)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for the generated image (default: <input_name>_prediction_map.png)')
    parser.add_argument('--with-legend', action='store_true',
                        help='Also generate a legend image showing class colors')
    parser.add_argument('--combined', action='store_true',
                        help='Create combined image with legend on the side')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Determine file type and load predictions
    ext = os.path.splitext(args.input_file)[1].lower()
    
    if ext == '.csv':
        prediction_map = load_predictions_csv(args.input_file)
    elif ext == '.npz':
        prediction_map = load_predictions_npz(args.input_file)
    else:
        print(f"Error: Unsupported file format: {ext}")
        print("Supported formats: .csv, .npz")
        sys.exit(1)
    
    # Create colored image
    print(f"\nGenerating colored prediction map...")
    colored_image = create_colored_image(prediction_map)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = os.path.dirname(args.input_file) or '.'
        output_path = os.path.join(output_dir, f"{base_name}_prediction_map.png")
    
    # Handle combined image with legend
    if args.combined:
        legend = create_legend_image()
        
        # Create combined image
        padding = 20
        combined_width = colored_image.width + padding + legend.width
        combined_height = max(colored_image.height, legend.height)
        
        combined = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
        combined.paste(colored_image, (0, 0))
        combined.paste(legend, (colored_image.width + padding, 0))
        
        combined.save(output_path)
        print(f"\nSaved combined image with legend: {output_path}")
    else:
        colored_image.save(output_path)
        print(f"\nSaved prediction map: {output_path}")
    
    # Save legend if requested
    if args.with_legend:
        legend = create_legend_image()
        legend_path = output_path.replace('.png', '_legend.png')
        legend.save(legend_path)
        print(f"Saved legend: {legend_path}")
    
    print(f"\nImage size: {colored_image.width} x {colored_image.height} pixels")
    print("Done!")


if __name__ == "__main__":
    main()
