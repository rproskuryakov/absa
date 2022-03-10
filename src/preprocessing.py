import os
import string
import xml
import xml.etree.ElementTree as ET

import pymorphy2
import spacy
import numpy as np
import pandas as pd

spacy_model = spacy.load("ru_core_news_sm")
