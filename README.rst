2-stage neural layout analysis
==============================

This is a fully neural layout analysis tool capable of extracting arbitrarily
shaped lines from documents. It operates in two steps, first labelling all
baselines in the document using a FCN-style deep network and then expanding
individual baselines with a small convolutional dilation network.

Everything is highly experimental, subject to changes without notice, and will
break frequently.

Installation
------------

- **Installation using Conda:**

``conda env create -f environment.yml``


- **Installation using pip:**

``git clone --single-branch -b baseline https://github.com/mittagessen/seg.git``

``pip3 install .``

to install the dependencies and the command line tool. For development purposes
use:

``pip3 install --editable .``

Training
--------

Training requires a directory with triplets of input images
$prefix.{plain,seeds,lines}.png. `plain` are RGB inputs, `seeds` are 8bpp
baselines annotations with each non-zero value representing a single baseline,
and `lines` are expansions of baselines to all pixels belonging to the line in
the same format as the seed files. A sample dataset based on the UW3 corpus can
be found here_.

Each step has to be trained separately but uses the same training data. Run:

::

   $ seg train --validation val train

to train the baseline detector. The dilation tool can be trained with:

::

   $ seg train_dilation --validation val train


Inference
---------

Not implemented yet.

.. _here: http://homer.dh.uni-leipzig.de/uw3.tar.xz
