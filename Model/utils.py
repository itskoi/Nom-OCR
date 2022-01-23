from parameters import *
import numpy as np

def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(ALPHABET.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=ALPHABET[ch]
    return ret
