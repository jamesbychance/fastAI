#0. Google collab only
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

#1. Import libraries
from fastbook import *
from fastai.vision.widgets import *



