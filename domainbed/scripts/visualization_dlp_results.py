from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_image(image_name):
    dataset = image_name.split('-')[0]
    domain = image_name.split('-')[1]
    class_name = image_name.split('-')[2]
    image_path = f'~/datasets/{dataset}/{domain}/{class_name}/*'
    images = glob.glob(image_path)[:4]
    fig=plt.figure()
    for image in images:
        axes.append( fig.add_subplot(rows, cols, i+1) )
        axes[-1].set_title(subplot_title)  
        image = cv2.imread(imag)
        plt.axis('off')
        plt.imshow(image)
        plt.show()

def get_image_paths(image_name, num=5):
    dataset = image_name.split('-')[0]
    domain = image_name.split('-')[1]
    class_name = image_name.split('-')[2]
    image_path = f'~/datasets/{dataset}/{domain}/{class_name}/*'
    images = glob.glob(image_path)[:num]
    # print(image_name, len(images))
    return images

clip = [76.6, 95.8, 79.9, 36.4, 72.2]
template = [82.3, 96.1, 82.3, 34.1, 73.7]
domain = [78.8, 96.6, 81.9, 28.4, 71.4]
ap = [84.3, 97.3, 84.2, 52.6, 79.6]

erm = [82.7, 92.9, 78.1, 50.2, 75.9]
coral = [82.0, 93.2, 78.9, 53.5, 76.9]
dann = [83.2, 93.8, 78.8, 52.2, 77.0]


RESULTS = [clip, erm, coral, dann, template ,  ap]
X = ['ERM', 'COR', 'DANN', '+temp' , '+DPL']
X = ['ERM', 'CORAL', 'DANN', '+T' , '+DPL']
color = ['powderblue', 'paleturquoise', 'skyblue', 'pink', 'crimson']
DATASETS = ['VLCS', 'PACS', 'OfficeHome', 'Terra', 'Average']

def show_average():
    dataset=4
    result = [RESULT[dataset] for RESULT in RESULTS]
    Y = [num - result[0] for num in result][1:]

    plt.bar(X, Y, color=color)
    plt.yticks(fontsize=18, rotation=90)
    plt.xticks(X, fontsize=18)
    plt.title(DATASETS[dataset], {'fontsize': 30})
    plt.ylim(-10,20)
    plt.ylabel('Accuracy', fontsize=28)
    plt.show()



def show_all():
    fig, ax = plt.subplots(2, 2, 
                        gridspec_kw={
                        'width_ratios': [1, 1],
                        'height_ratios': [1, 1]})
    
    dataset=0
    result = [RESULT[dataset] for RESULT in RESULTS]
    Y = [num - result[0] for num in result][1:]

    ax[0][0].bar(X, Y, color=color)
    ax[0][0].set_xticklabels(X, fontsize=18)
    ax[0][0].set_title(DATASETS[dataset], {'fontsize': 30})

    ax[0][0].set_ylim(-5,20)

    ax[0][0].label_outer()
    ax[0][0].tick_params(axis='y', labelsize=18)

    dataset=1
    result = [RESULT[dataset] for RESULT in RESULTS]
    Y = [num - result[0] for num in result][1:]

    ax[0][1].bar(X, Y, color=color)
    ax[0][1].set_xticklabels(X, fontsize=18)
    ax[0][1].set_title(DATASETS[dataset], {'fontsize': 30})

    ax[0][1].set_ylim(-5,20)

    ax[0][1].label_outer()
    ax[0][1].tick_params(axis='y', labelsize=18)
    
    dataset=2
    result = [RESULT[dataset] for RESULT in RESULTS]
    Y = [num - result[0] for num in result][1:]

    ax[1][0].bar(X, Y, color=color)
    ax[1][0].set_xticklabels(X, fontsize=18)
    ax[1][0].set_title(DATASETS[dataset], {'fontsize': 30})

    ax[1][0].set_ylim(-5,20)

    ax[1][0].label_outer()
    ax[1][0].tick_params(axis='y', labelsize=18)
    
    dataset=3
    result = [RESULT[dataset] for RESULT in RESULTS]
    Y = [num - result[0] for num in result][1:]

    ax[1][1].bar(X, Y, color=color)
    ax[1][1].set_xticklabels(X, fontsize=18)
    ax[1][1].set_title(DATASETS[dataset], {'fontsize': 30})

    ax[1][1].set_ylim(-5,20)

    ax[1][1].label_outer()
    ax[1][1].tick_params(axis='y', labelsize=18)

    fig.supylabel('Accuracy compared with CLIP', fontsize=28)
    plt.show()

show_all()