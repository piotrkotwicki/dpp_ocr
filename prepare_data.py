import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml

data_dir = 'data'
xml_file = 'annotations.xml'
output_dir = 'dataset_yolo'


def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def main():
    xml_path = os.path.join(data_dir, xml_file)
    if not os.path.exists(xml_path):
        print("Nie znaleziono pliku XML!")
        return

    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    files_data = {}

    for image in root.findall('image'):
        filename = image.get('name')
        width = float(image.get('width'))
        height = float(image.get('height'))

        bboxes = []
        for box in image.findall('box'):
            if box.get('label') == 'plate':
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                yolo_bbox = convert_bbox((width, height), (xtl, xbr, ytl, ybr))
                bboxes.append(yolo_bbox)

        if bboxes:
            files_data[filename] = bboxes

    all_files = list(files_data.keys())
    random.shuffle(all_files)

    split_idx = int(len(all_files) * 0.7)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Razem plików: {len(all_files)}")
    print(f"Trening (70%): {len(train_files)}")
    print(f"Walidacja (30%): {len(val_files)}")

    def process_files(file_list, split_name):
        for fname in file_list:
            src_img = os.path.join(data_dir, fname)
            dst_img = os.path.join(output_dir, 'images', split_name, fname)

            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)

                txt_name = os.path.splitext(fname)[0] + ".txt"
                dst_txt = os.path.join(output_dir, 'labels', split_name, txt_name)

                with open(dst_txt, 'w') as f:
                    for bbox in files_data[fname]:
                        f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            else:
                print(f"Pominięto (brak pliku): {fname}")

    process_files(train_files, 'train')
    process_files(val_files, 'val')

    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'plate'}
    }

    with open('dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print("\nGotowe! Dane przygotowane w folderze 'dataset_yolo' oraz plik 'dataset.yaml'.")


if __name__ == '__main__':
    main()