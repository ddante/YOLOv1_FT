import os

# Directory containing the images and the annotations file
DATABASE_DIR = "VOC2007"

# Name od the classes of the object in the database
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# Size use to reshpae the imported image
IMG_SHAPE_W = 448
IMG_SHAPE_H = 448

# Number of grid cells used in YOLO
IMG_CELLS_W = 7
IMG_CELLS_H = 7

# Number of bbox per cell
NUM_BBOX = 2

def create_labels(img_id, f):
    # For each image, read the annotations from the xml file and
    # append them the dataset list of files.
    # Labels for each object in the image:
    #    bounding box coordinates [x_min, y_min, x_max, y_max], class id [ID]
    import xml.etree.ElementTree as ET

    in_file = DATABASE_DIR + '/Annotations/%s.xml' % img_id

    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text

        cls = obj.find('name').text

        # Skip objects that are difficult to detect or
        # that are not in the class list
        if(int(difficult) == 1 or cls not in CLASSES):
            continue

        # Convert from class name to class ID
        class_id = CLASSES.index(cls)

        bbox = obj.find('bndbox')
        x_min = bbox.find('xmin').text
        y_min = bbox.find('ymin').text
        x_max = bbox.find('xmax').text
        y_max = bbox.find('ymax').text

        bb_text = x_min + ', ' + y_min + ', ' + x_max + ', ' + y_max

        # Append the labels to the image path
        f.write(', ' + bb_text + ', ' + str(class_id))


def read_dataset(set_type):
    # For the training and validation sets, create a txt file
    # that contains the list of all the images in the dataset.
    # Then, for each image, add its labels.

    list_dir = DATABASE_DIR + '/ImageSets/Main/%s.txt' % set_type
    with open(list_dir, 'r') as f:
        img_ids = f.read().strip().split()

    img_dir =  DATABASE_DIR + '/%s.txt' % set_type
    with open(img_dir, 'w') as f:
        for img_id in img_ids:
            f.write(DATABASE_DIR+'/JPEGImages/%s.jpg' % img_id)
            create_labels(img_id, f)
            f.write('\n')


read_dataset('train')
read_dataset('val')

def read_img(img_path, labels):
    import cv2 as cv
    import numpy as np

    img = cv.imread(img_path)
    #cv.imshow('img', img)
    #cv.waitKey(1000)

    img_w, img_h = img.shape[0:2]
    img = cv.resize(img, (IMG_SHAPE_W, IMG_SHAPE_H))

    # Normalize the image in [0,1]
    img = img / 255.0

    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #cv.imshow('img', img)
    #cv.waitKey(1000)

    # Lenght of the labels = (B*bb + C)
    #   B: number of bbox
    #   bb: center + shape + cofindene of the box
    #   C: one-hot vector for the class
    label_len = 5*NUM_BBOX + len(CLASSES)

    # For each grid cells label_matrix stores the following information:
        # For each box:
        #   pc: wheter or not the center of the bbox is in the cell [1 int]
        #   bb: center and size of the bbox [4 float]
        # class: one-hot verctor for the class of the object [20 int]
    label_matrix = np.zeros([IMG_SHAPE_W, IMG_SHAPE_H, label_len])

    for i in range(0, len(labels), 5):
        x_min  = int(labels[i])
        y_min  = int(labels[i+1])
        x_max  = int(labels[i+2])
        y_max  = int(labels[i+3])
        cls_id = int(labels[i+4])

        # Center of the bbox normalized with the image size
        xc_bb = (x_min + x_max) / 2.0 / img_w
        yc_bb = (y_min + y_max) / 2.0 / img_h

        # Dimensions of the bbox normalized with the image size
        w_bb  = (x_max - x_min) / img_w
        h_bb  = (y_max - y_min) / img_h

        loc_x = IMG_CELLS_W * xc_bb
        loc_y = IMG_CELLS_H * yc_bb

        # Cell (row, col) containing the center of the bbox
        cell_x = int(loc_x)
        cell_y = int(loc_y)

        # Center of the bbox w.r.t the top left corner of the cell
        # and normilez with the cell size
        x = loc_x - cell_x
        y = loc_y - cell_y

        idx_oh = NUM_BBOX*5 + cls_id

        if(label_matrix[cell_y, cell_x, 0] == 0):
            label_matrix[cell_y, cell_x, 0]      = 1
            label_matrix[cell_y, cell_x, 1:5]    = [x, y, w_bb, h_bb]
            label_matrix[cell_y, cell_x, idx_oh] = 1

    return img, label_matrix

with open(DATABASE_DIR + '/val.txt') as f:
    lines = f.readlines()

for item in lines:
    item = item.replace("\n","").split(", ")
    read_img(item[0], item[1:])
