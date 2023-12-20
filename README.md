# hpc-project
Evaluation of performance enhancement methods

The recommended method to view the code is to open hpc_project.ipynb in a Google Colab environment and follow the outputs for the code cells. Detailed documentation is given along with multiple tests performed before downloading the Jupyter Notebook for setting up on GitHub.

#
# Instructions to run
#

First download the 160px version of ImageNette2 and extract in the same folder as this code to access the datasets.

https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz

The link above will provide the tarball for the download
Make sure to extract the data into this directory too.

Later, run the main.py to perform the wandb sweeps. If wandb asks for any login information, setup everything on the Weights and Biases website before confirming the login

#
# Links/resources
#

All model tarballs from the final checkpoint for each Wand sweep can be found in the link below
https://drive.google.com/drive/folders/1zEqhhyuhpojxKNrvMdi4twvbGR9Ql93O?usp=drive_link

The link to the project presentation can be found here
https://docs.google.com/presentation/d/1C45F1IpgqUI2G3GSXJNHZVCEamXbE1KB_sxTS9oXdpQ/edit?usp=sharing

All Weights and Biases sweeps are to be found below (publicly accessible)
https://wandb.ai/impossibile/hpc-proj/sweeps
The link below has the sweep for the reduced precision bfloat16 format
https://wandb.ai/impossibile/hpc-proj/sweeps/x5cp0llc
The link below has the sweep for the float16 format runs
https://wandb.ai/impossibile/hpc-proj/sweeps/s9n31tfj