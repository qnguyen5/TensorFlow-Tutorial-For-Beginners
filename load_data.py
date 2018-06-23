import numpy as np
import os
from skimage import data as skimagedata
from skimage import transform as skimagetransform
from skimage import color as skimagecolor
import matplotlib.pyplot as plt

def load_data(data_directory):
    """
    data_directory is of class 'str'
    """
    directories = [d for d in os.listdir(data_directory)
    if os.path.isdir(os.path.join(data_directory,d)) == True]

    labels = []
    images = []

    for d in directories:
        label_directory = os.path.join(data_directory,d)
        file_names = [os.path.join(label_directory,f)
        for f in os.listdir(label_directory) if f.endswith(".ppm")]

        for f in file_names:
            images.append(skimagedata.imread(f))
            labels.append(int(d))

    return [images,labels]


ROOT_PATH = "C:/Users/quoc_/AppData/Local/Programs/Python/Python36/pycode3/datacamp-community-tutorials/TensorFlow Tutorial For Beginners"
train_data_directory = os.path.join(ROOT_PATH,"TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH,"TrafficSigns/Testing")

images,labels = load_data(train_data_directory)
images,labels = [np.asarray(images),np.asarray(labels)]
unique_labels = np.asarray(list(set(labels)))


#PRELIMINARY DATA VISUALIZATION
"""
#>See Distribution of Labels
#fig,ax = plt.subplots(nrows = 1, ncols = 1)
#n,bins,patches = ax.hist(labels,len(set(labels)),facecolor = 'blue')
#ax.grid(True)
#plt.show()

#>Make a list of 4 random images to see
#np.random.seed(12361234) #reproducible experiment
#traffic_signs = np.sort([np.random.randint(0,len(labels)) for x in range(4)])
#for i in range(len(traffic_signs)):
    #plt.subplot(1,4,i+1)
    #plt.axis('off')
    #plt.imshow(images[traffic_signs[i]])
    #plt.subplots_adjust(wspace = 0.5)
#plt.show()

#>Print out all images to see
fig,ax = plt.subplots(nrows = 8, ncols = 8, figsize = (15,15))

for i in range(len(unique_labels)):
    label = unique_labels[i]
    where_are_the_labels = np.argwhere(labels == label)
    labelcount =  where_are_the_labels.size
    label_first_index = where_are_the_labels[np.random.randint(0,labelcount)][0]
    image = images[label_first_index]
    ax_ax1 = i//8
    ax_ax2 = i % 8
    ax[ax_ax1][ax_ax2].imshow(image)
    ax[ax_ax1][ax_ax2].axis('off')
    ax[ax_ax1][ax_ax2].set_title("Label {0} ({1})".format(label,labelcount))

plt.show()
"""

images28 = np.asarray([skimagetransform.resize(image, (28,28)) for image in images])
images28 = skimagecolor.rgb2gray(images28)


#TEST DATA

# Load the test data
test_images, test_labels = load_data(test_data_directory)

# Transform the images to 28 by 28 pixels
test_images28 = [skimagetransform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
test_images28 = skimagecolor.rgb2gray(np.array(test_images28))
