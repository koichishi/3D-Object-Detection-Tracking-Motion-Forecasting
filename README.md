# Self-driving Model Using LiDAR 

This repository contains the project by Gongyi Shi and Anny Dai, under supervision of 
Prof. Raquel Urtasun.

## Getting started

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

   ```bash
   curl 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh' > Miniconda.sh
   bash Miniconda.sh
   rm Miniconda.sh
   ```
   
2. Create a new conda environment:

   ```bash
   conda env create --file environment.yml
   ```

3. Activate environment:

   ```bash
   conda activate csc490
   ```

4. Download [PandaSet](https://scale.com/resources/download/pandaset).
   After submitting your request to download the dataset, you will receive an
   email from Scale AI with instructions to download PandaSet in three parts.
   Download Part 1 only. After you have downloaded `pandaset_0.zip`,
   unzip the dataset as follows:

   ```bash
   unzip pandaset_0.zip -d <your_path_to_dataset>
   ```
