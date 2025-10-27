import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def save_improved_bbox_image(frame_image, boxes, labels, save_path, confidence_scores=None):
    """
    Improved bounding box visualization for GPT-4o
    - Cleaner boxes with better colors
    - Clear labels with confidence scores
    - Less visual clutter
    """
    if len(boxes) == 0:
        # Save original image if no boxes
        cv2.imwrite(save_path, cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR))
        return
    
    # Create figure with higher DPI for clarity
    fig, ax = plt.subplots(1, figsize=(10, 8), dpi=100)
    ax.imshow(frame_image)
    ax.axis('off')
    
    # Use distinct, high-contrast colors
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if isinstance(box, list):
            x1, y1, x2, y2 = box
        else:
            x1, y1, x2, y2 = box.cpu().numpy() if hasattr(box, 'cpu') else box
            
        # Use distinct color for each box
        color = colors[i % len(colors)]
        
        # Draw clean rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add clear label with background
        if confidence_scores and i < len(confidence_scores):
            label_text = f"{i+1}: {label} ({confidence_scores[i]:.2f})"
        else:
            label_text = f"{i+1}: {label}"
            
        # Add text with contrasting background
        ax.text(x1, y1-5, label_text, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
               fontsize=10, color='black', weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()

def extract_confidence_from_labels(prob_labels):
    """Extract confidence scores from LASER probability labels"""
    confidences = []
    clean_labels = []
    
    for label in prob_labels:
        if "::" in label:
            conf_str, clean_label = label.split("::", 1)
            try:
                confidence = float(conf_str)
                confidences.append(confidence)
                clean_labels.append(clean_label)
            except ValueError:
                confidences.append(0.0)
                clean_labels.append(label)
        else:
            confidences.append(0.0)
            clean_labels.append(label)
    
    return clean_labels, confidences
