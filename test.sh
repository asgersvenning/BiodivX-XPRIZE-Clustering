#!/bin/bash
# rm -rf /home/george/codes/BiodivX-XPRIZE-Clustering/test/data/20230730_tiny_fb/crops_clustered

python /home/george/codes/BiodivX-XPRIZE-Clustering/cluster.py\
 --in_folder /home/george/codes/BiodivX-XPRIZE-Clustering/data/20230730_tiny_fb/crops\
 --meta_folder /home/george/codes/BiodivX-XPRIZE-Clustering/data/20230730_tiny_fb/metadata\
 --device cuda:0