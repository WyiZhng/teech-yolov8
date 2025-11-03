import os
import pandas as pd
from PIL import Image

def yolo_to_absolute(yolo_coords, img_width, img_height):
    """
    Converts YOLO format coordinates (normalized) to absolute pixel coordinates.
    YOLO format: [x_center, y_center, width, height]
    Absolute format: [x_min, y_min, width, height]
    """
    x_center, y_center, width, height = yolo_coords
    abs_width = width * img_width
    abs_height = height * img_height
    x_min = (x_center * img_width) - (abs_width / 2)
    y_min = (y_center * img_height) - (abs_height / 2)
    return [int(x_min), int(y_min), int(abs_width), int(abs_height)]

def process_label_files(root_dir, output_csv):
    """
    Processes all label files in the dataset and converts them to the specified CSV format.
    """
    # Define the columns for the output DataFrame
    columns = ['image_id', 'tooth_id', 'surface', 'icdas', 'gx', 'gy', 'gw', 'gh']
    all_data = []

    # Iterate through train, val, test sets
    for subset in ['train', 'val', 'test']:
        labels_dir = os.path.join(root_dir, subset, 'labels')
        images_dir = os.path.join(root_dir, subset, 'images')

        if not os.path.isdir(labels_dir):
            print(f"Directory not found: {labels_dir}")
            continue

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue

            image_id_base = os.path.splitext(label_file)[0]
            # Find corresponding image file to get dimensions
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_img_path = os.path.join(images_dir, image_id_base + ext)
                if os.path.exists(potential_img_path):
                    img_path = potential_img_path
                    break
            
            if not img_path:
                print(f"Image for label {label_file} not found. Skipping.")
                continue

            with Image.open(img_path) as img:
                img_width, img_height = img.size

            image_id = os.path.basename(img_path)

            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    icdas = int(parts[0])
                    yolo_coords = [float(p) for p in parts[1:]]
                    
                    abs_coords = yolo_to_absolute(yolo_coords, img_width, img_height)
                    gx, gy, gw, gh = abs_coords

                    # tooth_id and surface are not in the source files
                    tooth_id = ''  # Placeholder
                    surface = ''   # Placeholder

                    all_data.append([image_id, tooth_id, surface, icdas, gx, gy, gw, gh])

    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Successfully created {output_csv}")

if __name__ == '__main__':
    # The root directory of your dataset
    dataset_root = 'VOCdevkit'
    # The name for the output CSV file
    output_file = 'icdas_strong_labels.csv'
    process_label_files(dataset_root, output_file)
