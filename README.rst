Pixel labelling for layout analysis
===================================

This is a fine grained document pixel labelling tool for layout analysis
purposes. It uses a FCN-style deep network with conditional random field
postprocessing to assign each pixel of an input image to a particular class
(background, main text, decoration, annotation). 

Everything is highly experimental, subject to changes without notice, and will
break frequently.

Installation
------------

Run:

::
        $ pip3 install .

to install the dependencies and the command line tool. For development purposes
use:

::
        $ pip3 install --editable .

Training
--------

Training requires a directory with input images in JPG and their corresponding
labelled ground truth in PNG format. The labels should correspond to the hisDB
standard, i.e. 1-bit per class in the lowest 4bits of the red color channel.

There are half a dozen options that don't really improve training results,
notably per-class loss weights, encoder refinement, and augmentation.

Inference
---------

Run:

::
        $ seg pred -m $model_file $img_1 $img_2 ... $img_n

Outputs are the original file name plus a `class_n` suffix and an opaque
overlay image per input file.
