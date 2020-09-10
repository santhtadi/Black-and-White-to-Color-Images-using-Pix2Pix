# Black-and-White-to-Color-Images-using-Pix2Pix
Black and White to Color Images using Pix2Pix (GAN - Generative Adversarial Networks)

# Description

This notebook is executed in colab and is a fun experiment on using GAN for converting images of people from Gray Scale to RGB

# Algorithm

I used the Pix2Pix Network published at https://arxiv.org/abs/1611.07004 for this experiment

Most of the notebook is based heavily on the Pix2Pix tutorial and the model architecture provided by Tensorflow.

More details can be found at https://www.tensorflow.org/tutorials/generative/pix2pix

# Dataset

The dataset I have used is the Labeled Faces in the wild dataset at http://vis-www.cs.umass.edu/lfw/


Citation: 

**Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller.
Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.
University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.**

**[[pdf](http://vis-www.cs.umass.edu/lfw/lfw.pdf)]**

## Steps for making the dataset

I took 500 images as the dataset for experimenting with pix2pix

The input and output images should be places side by side

```
# iterate over the first 500 images
for i, fp in tqdm(enumerate(all_images[:500])):
    
    # read the image in BGR format using opencv*
    img = cv2.imread(fp)
    
    # resize the image to 256x256
    img = cv2.resize(img, (256, 256))
    
    # convert the image to gray scale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # convert the gra scale to a 3 channel gray image, note that the image is not converted to a color image
    grey_3channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    
    # stitch the input and output images side by side
    save_img = np.concatenate((img, grey_3channel), axis=1)
    
    # use opencv imwrite to save the dataset image
    cv2.imwrite("./test/"+str(i+)+".jpg", save_img)
```


# Results
Some of the results from the generated model:

Note: This image is not in the dataset

![Result Image](https://github.com/santhtadi/Black-and-White-to-Color-Images-using-Pix2Pix/blob/master/new_download.png?raw=true)

