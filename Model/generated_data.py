from PIL import Image, ImageDraw, ImageFont
from parameters import *
from tqdm import tqdm
import numpy as np
import os
import argparse
import cv2

class DataGenerator:
    def __init__(self, data_folder, data_size = 100, save_image=False):
        self.log = []
        self.errors = []
        self.data_folder = data_folder
        self.image_folder = data_folder + '/images'
        self.label_file = data_folder + '/labels.txt'
        self.font_list = FONT_LIST
        self.data_set_csv = data_folder + f'/{DATASET_FILE_NAME}'
        self.characters = []
        self.dataset_size = data_size
        self.save_image = save_image

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def get_list_characters(self):
        if len(self.characters) != 0:
            return self.characters
        else:
            characters = [char for char in ALPHABET]
            self.characters = characters
            return characters
    
    def create_text_image(self, text, font_ttf, idx, font_size):
        try:
            image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_ttf, font_size)
            w, h = draw.textsize(text, font=font)
            draw.text(((IMG_WIDTH - w) / 2, (IMG_HEIGHT - h) / 2), text, (0, 0, 0), font=font)

            if self.save_image:
                image.save(f'{self.image_folder}/{idx}.jpg')

            self.log.append({'font': font_ttf, 'image':f'{idx}.jpg'})
            return image
        except Exception as e:
            self.errors.append({'font': font_ttf, 'errors': str(e)})
            return None

    def generate_data_set(self, text, idx):
        images = []
        with open(self.font_list, 'r') as fonts:
            font_list = [font for font in fonts]
            font = np.random.choice(font_list)
            font_size = np.random.randint(FONT_SIZE_MIN, FONT_SIZE_MAX+1)
            image = self.create_text_image(text, font.replace('\n', ''), idx, font_size)
            if image != None:
                images.append(image)
        return images

    def generate_images_from_corpus(self, corpus):
        with open(corpus, 'r', encoding='utf-8') as file:
            texts = file.readlines()
            for idx, text in enumerate(texts):
                self.generate_data_set(text, idx)
                
    
    def generate_dataset(self):
        characters = self.get_list_characters()
        labels = []
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        if self.save_image and not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        for i in tqdm(range(self.dataset_size)):
            # Generate text
            text_size = np.random.randint(MIN_STR_LEN, MAX_STR_LEN + 1)
            text = ''
            for _ in range(text_size):
                text += np.random.choice(characters)
            labels.append(text)
            # Create image from the text
            c_images = self.generate_data_set(text, i)
            image = np.asarray(c_images[0])
            image = self.rgb2gray(image)
            image = image.transpose()
            image = np.flip(image, axis=1)
            image = image.reshape(1, IMG_WIDTH * IMG_HEIGHT)
            with open(self.data_set_csv, 'ab') as df:
                # image = np.concatenate((image, np.array([[int(idx)]])), axis=1)
                np.savetxt(df, image, delimiter=",", fmt="%d")
                df.close()
        # Save labels
        with open(self.label_file, 'w', encoding='utf-8') as label_file:
            for label in labels:
                label_file.write(label+'\n')

def generate_alphabet():
    with open(NOMCORPUS_FILE, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
        alphabets = set()
        for line in corpus:
            for char in line.strip():
                if char != '':
                    alphabets.add(char)
        with open(ALPHABETS_FILE, 'w', encoding='utf-8') as afile:
            for char in alphabets:
                afile.write(char)
            # Add a blank space
            afile.write(' ')
       

        
        
# -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--alphabet', action='store_true', help='Created word dictionary from corpus!')
parser.add_argument('--dir', type=str, help='Path to folder containing the data')
parser.add_argument('--corpus', type=str, help='Path to corpus')
parser.add_argument('--data_size', type=int, help='Number of generated images', default=1000)
parser.add_argument('--save_image', action='store_true', help='Save images', default=False)
parser.add_argument('--only_images', action='store_true', help='Generate images from corpus', default=False)
args = parser.parse_args()

if args.alphabet:
    print('Generating alphabet...')
    generate_alphabet()
    print('Alphabet is generated at: ', ALPHABETS_FILE)
elif args.only_images:
    print('Generating images from corpus')
    generator = DataGenerator(data_folder=args.dir, data_size=args.data_size, save_image=args.only_images)
    generator.generate_images_from_corpus(args.corpus)
    print('Image Dataset is generated at: ', args.dir)
else:
    print('Generating dataset...')
    generator = DataGenerator(data_folder=args.dir, data_size=args.data_size, save_image=args.save_image)
    generator.generate_dataset()
    print('Text Image Dataset is generated at: ', args.dir)