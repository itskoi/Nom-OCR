from parameters import *
from utils import *
import model as Model
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from tensorflow.keras.optimizers import Adam

# ---- Parse the args -----
parser = argparse.ArgumentParser()
parser.add_argument('--finetune', action='store_true', help='Load the previous work')
parser.add_argument('--name', type=str, help='Version name', required=True)
args = parser.parse_args()

# ----- Prepare the Data ----- 
print('Preparing the data...')
## ---- Images ----
print('Load images...')
train_x = np.empty(shape=(0, IMG_WIDTH*IMG_HEIGHT))
valid_x = np.empty(shape=(0, IMG_WIDTH*IMG_HEIGHT))

print('Train dataset...')
if TRAIN_SIZE > CHUNK_SIZE:
    train_df = pd.read_csv(TRAIN_DATASET_FILE, header=None, chunksize=CHUNK_SIZE)
    for idx, df in enumerate(train_df):
        if idx >= int(np.ceil(TRAIN_SIZE/CHUNK_SIZE)):
            break
        train_x = np.concatenate([train_x, df[:] / 255], axis=0)
        print(f'Read Train dataset chunk [{idx}]')
        
else:
    train_df = pd.read_csv(TRAIN_DATASET_FILE, header=None)
    train_x = train_df[:TRAIN_SIZE] / 255 # normalize

train_x = np.array(train_x).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1).astype(np.float32)
print(f'Train dataset shape: {train_x.shape}')
print('Valid dataset...')
if VALID_SIZE > CHUNK_SIZE:
    valid_df = pd.read_csv(VALID_DATASET_FILE, header=None, chunksize=CHUNK_SIZE)
    for idx, df in enumerate(valid_df):
        if idx >= int(np.ceil(VALID_SIZE/CHUNK_SIZE)):
            break
        valid_x = np.concatenate([valid_x, df[:] / 255], axis=0)
        print(f'Read Valid dataset chunk [{idx}]')
else:
    valid_df = pd.read_csv(VALID_DATASET_FILE, header=None)
    valid_x = valid_df[:VALID_SIZE] / 255 # normalize

valid_x = np.array(valid_x).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1).astype(np.float32)
print(f'Valid dataset shape: {valid_x.shape}')
print('Load labels...')
## ---- Labels ----
with open(TRAIN_LABEL_FILE, 'r', encoding='utf-8') as train_label_file:
    train_labels = train_label_file.readlines()
    train_labels = [label.strip() for label in train_labels]
with open(VALID_LABEL_FILE, 'r', encoding='utf-8') as valid_label_file:
    valid_labels = valid_label_file.readlines()
    valid_labels = [label.strip() for label in valid_labels]
### ---------------------------------------------------------
### train_y: true labels converted to numbers and padded with -1.
###          the length of each label is equal to MAX_STR_LEN
### train_label_len: contains the length of each true label (without padding)
### train_input_len: length of predicted label = NUM_OF_TIMESTEPS - 2
### train_output: is a dummy output for CTC loss
### ---------------------------------------------------------
train_y = np.ones([TRAIN_SIZE, MAX_STR_LEN]) * -1
train_label_len = np.zeros([TRAIN_SIZE, 1])
train_input_len = np.ones([TRAIN_SIZE, 1]) * (NUM_OF_TIMESTEPS-2)
train_output = np.zeros([TRAIN_SIZE])
for i in range(TRAIN_SIZE):
    train_label_len[i] = len(train_labels[i])
    train_y[i, 0:len(train_labels[i])] = label_to_num(train_labels[i])

valid_y = np.ones([VALID_SIZE, MAX_STR_LEN]) * -1
valid_label_len = np.zeros([VALID_SIZE, 1])
valid_input_len = np.ones([VALID_SIZE, 1]) * (NUM_OF_TIMESTEPS-2)
valid_output = np.zeros([VALID_SIZE])
for i in range(VALID_SIZE):
    valid_label_len[i] = len(valid_labels[i])
    valid_y[i, 0:len(valid_labels[i])] = label_to_num(valid_labels[i])

print('Data has been prepared!')

# ----- Define the model -----

print('Builing the CRNN model...')
model, pred_model = Model.get_model(train=True)
# the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate = 0.0001))

# Load previous work
if args.finetune:
    print('Load pretrained model...')
    model.load_weights(CHECKPOINT_DIR + f'/best_train_{args.name}.h5')    

# Create the checkpoint callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=CHECKPOINT_DIR + f'/best_train_{args.name}.h5',
                                save_weights_only=True,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                verbose=1
                            )
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0, verbose=0, mode='min')
print('The model has been built!')

print('Training...')
model.fit(x=[train_x, train_y, train_input_len, train_label_len], 
          y=train_output, 
          validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
          epochs=60,
          batch_size=128,
          callbacks=[model_checkpoint_callback, model_early_stopping_callback]
          )

print(f'Save weights at: {CHECKPOINT_DIR}/best_pred_{args.name}.h5 ...')
pred_model.save(CHECKPOINT_DIR + f'/best_pred_{args.name}.h5')

print('Complete!')