import os
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from google_scraper import scrape

# size = 224, 224
images_per_class = 850
class_list = [# 'camellias',
    'western red cedar tree',
    'redwood cedar tree',
    'rhododendrons',
    'rose hips',
    'cherry trees',
    'shrub',
    'Daphne flower']

path = 'images/'

for img_class in class_list:
    try:
        scrape(str(img_class), int(images_per_class))
    except Exception as e:
        print('Error. Skipping! {} class: {}'.format(e, img_class))
        continue
    
    folder = str(img_class).replace(' ','_')
    
    ##Resize image
    picture = os.listdir(path + folder)

    for j, pic in enumerate(picture):
        try:
            im = Image.open('{}{}/{}'.format(path, folder, pic)).convert('RGB')
            os.remove('{}{}/{}'.format(path, folder, pic))
            im.thumbnail(im.size, Image.ANTIALIAS)
            im.save('{}{}/{}.jpg'.format(path, folder, str(j)), "JPEG", optimize=True)
        except Exception as e:
            print('Error. Skipping! {} image: {}'.format(e, j))
            continue
