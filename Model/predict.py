import cv2
import numpy as np
import pandas as pd
import model as Model
import argparse
from tqdm import tqdm
from parameters import *
from utils import *
from keras import backend as K

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='Name of the pretrained model', required=True)
parser.add_argument('--dir', type=str, help='Path to a folder containing the data need to be predicted', default='data/test')
parser.add_argument('--num', type=int, help='Number of images need to be predicted', default=10)
parser.add_argument('--use_images', action='store_true', default=False)
parser.add_argument('--only_indicator', action='store_true', default = False, help='Hide the label each image')
args = parser.parse_args()

data_path = args.dir
num_imgs = args.num

# Load the best model
print('Loading model...')
# model = load_model(CHECKPOINT_DIR)
model = Model.get_model(train=False)
model.load_weights(CHECKPOINT_DIR+f'/best_train_{args.name}.h5')
print('Model has been loaded!')

with open(f'{data_path}/{LABEL_FILE_NAME}', 'r', encoding='utf-8') as labels_file:
    true_labels = labels_file.readlines()

print('--- START ---')

correct_char = 0
total_char = 0

if args.use_images:
    print('Use images!')
    # Predict
    for i in tqdm(range(num_imgs)):
        image = cv2.imread(f'{data_path}/images/{i}.jpg', cv2.IMREAD_GRAYSCALE)
        image = np.asarray(image)
        image = image.transpose()
        image = np.flip(image, axis=1)
        image = image/255.0 # Normalize
        pred = model.predict(image.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
        decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                        greedy=True)[0][0])

        pred_label = num_to_label(decoded[0])
        true_label = true_labels[i]
        total_char += len(true_label)

        for _ in range(min(len(pred_label), len(true_label))):
            if(pred_label[_] == true_label[_]):
                correct_char += 1
        
        if not args.only_indicator:
            print(f'Image [{i}]: {pred_label} - Ground truth: {true_label}') 
else:
    print('Use csv file!')
    df = pd.read_csv(f'{data_path}/{DATASET_FILE_NAME}', header=None)
    test_x = df[:num_imgs] / 255
    test_x = np.array(test_x).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)

    preds = model.predict(test_x)
    preds = preds[:, 2:, :]
    decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])
    for i in tqdm(range(num_imgs)):
        pred_label = num_to_label(decoded[i])
        true_label = true_labels[i]
        total_char += len(true_label)

        for _ in range(min(len(pred_label), len(true_label))):
            if(pred_label[_] == true_label[_]):
                correct_char += 1
        
        if not args.only_indicator:
            print(f'Image [{i}]: {pred_label} - Ground truth: {true_label}')
print('---------------------------------------')
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('--- END ---')