This repo contains a series of AD models I trained during the summer of 2023 in an existing public dataset: https://arxiv.org/abs/2107.02157
I followed closely the work of https://www.nature.com/articles/s42256-022-00441-3#data-availability, which outlined the use of two AEs and two VAEs.

The data used can be found in the Data folder, and all of the python ML code is found in the Notebooks folder, which contains many versions of similar codes,
since I'm paranoid about losing work. However, the background data h5 file is not included here, as it is very large.
Accordingly, it may be easier to interface with the dataset directly instead of through this repo. All of the trained models can be found in Trained Models.

Everything was copied directly from Goolge Colab, so none of the directories in the codes will work; accessing data, as well as saving and loading models will have
to be slightly modified before use. Furthermore, there is no good system to determine which trained models are the most updated.

I will hopefully be fixing these issues soon such that all of the codes can be run with no issue, and it is clear which trained models are the most updated.