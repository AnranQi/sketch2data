import os
from pathlib import Path
from tqdm import tqdm
import yaml

def data2yolo_format(dir, sizex, sizey):
    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in file.readlines():  
                row = row.strip().split(" ")
                cls = int(row[1])
                cls = 0
                box = [(float(row[2])+float(row[3]))/2.0/sizex , (float(row[4]) + float(row[5]))/2.0/sizey,  (float(row[3]) - float(row[2]))/sizex, (float(row[5]) - float(row[4]))/sizey]
                lines.append(f"{cls} {' '.join(f'{x:.4f}' for x in box)}\n")
                with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                    fl.writelines(lines)  # write label.txt



def split_data(image_dir, annotation_dir, train_split=0.7, val_split=0.1, test_split=0.2):
    import random,shutil
    images = sorted(os.listdir(image_dir))
    annotations = sorted(os.listdir(annotation_dir))
    data = list(zip(images, annotations))
    random.shuffle(data)
    total_count = len(data)
    train_count = int(total_count * train_split)
    val_count = int(total_count * val_split)

    train_data = data[:train_count]
    val_data = data[train_count:train_count + val_count]
    test_data = data[train_count + val_count:]

    for dataset, dataset_name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
        os.makedirs(os.path.join(image_dir, dataset_name), exist_ok=True)
        os.makedirs(os.path.join(annotation_dir, dataset_name), exist_ok=True)
        for image_file, annotation_file in dataset:
            shutil.move(os.path.join(image_dir, image_file), os.path.join(image_dir, dataset_name, image_file))
            shutil.move(os.path.join(annotation_dir, annotation_file), os.path.join(annotation_dir, dataset_name,  annotation_file))

def write_yaml(detection_dir, yaml_name):
   # detection_dir: the dir which store the detection dataset
   # yaml_name: the yaml name only
    data = {
        "path": os.path.abspath(detection_dir), #need full path here
        "train": "images\\train",  # train images (relative to 'path')
        "val": "images\\val",      # val images (relative to 'path')
        "test": "images\\test",    # test images (optional)
        "names": {
            0: 0  # Classes
        }
    }

    # Write the data to a YAML file
    file_path = os.path.join(detection_dir, yaml_name)
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print(f"YAML file has been created and saved as {file_path}")

# Download
detection_dir = '.\\thoughts_detection_dataset'
yaml_name = 'thoughts.yaml'
sizex = 1000
sizey = 700

dir = Path(detection_dir)  # dataset root dir
image_dir = os.path.join(detection_dir, 'images')
annotation_dir = os.path.join(detection_dir, 'labels')

data2yolo_format(dir, sizex, sizey)
split_data(image_dir, annotation_dir, train_split=0.7, val_split=0.1, test_split=0.2)
write_yaml(detection_dir, yaml_name)
