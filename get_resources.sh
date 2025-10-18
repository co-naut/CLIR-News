#!/bin/bash

echo -e "\nINFO: installing gdown through pip\n"

pip install gdown

echo -e "\nINFO: gdown installed\nINFO: downloading word2vec embeddings\n"

gdown --folder --id 1RDeOuoRniCe1AN9jFSC6TPYEJ3N7JAmr

echo -e "\nINFO: exiting..."
