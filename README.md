# Objects recognition with SIFT descriptors

## How to set up

The necessary files in the model directory are available on this link : 
https://drive.google.com/file/d/1PFndo48MaLw3B37CIXSoIkzgs4_LTW71

The datasets used are available on these links:
1. https://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php
1. http://www.vision.caltech.edu/Image_Datasets/Caltech101/

#### Make a detection
Run `$ python detect.py − i path_to_image −t threshold −K neighbors −d dataset`

the parameters are:

1. -i: path to the image containing the object to be recognized
1. -t (optional) : Threshold value for matching. Default 0.6
1. -K (optional) : Number of best scores to consider. Default 5
1. -d (optional) : Dataset whose model we want to consider (1 = Dataset1, 2 = Dataset2). Default 1

Exemple: 

`$ python detect.py − i test_dataset1/obj15__355.png −t 0.6 −K 5`
