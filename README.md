# MACS MSc Thesis

## Introduction

Welcome! 

Here you can find the scripts that I wrote for my MSc thesis in Applied Sciences & Engineering: Applied Computer Science at Vrije Universiteit Brussel. I performed this MSc thesis under supervision of Prof. Dr. Wim Thiery and with help of my daily supervisor Inne Vanderkelen. The files have references to files/folders stored on Hydra, the HPC cluster of VUB and ULB. The manuscript of my MSc thesis can be found [here](MA_ACS_Lampe_Seppe_S2_August_2021_links_hidden.pdf).

In this project, we predict nightly thunderstorms over Lake Victoria from a satellite-derived product using machine and deep learning. As a proxy for thunderstorm activity we obtained the 'Overshooting Top' (OT) dataset designed by NASA's Langley Research Center, which is derived from the SEVIRI satellite product.

![alt text](https://github.com/SeppeLampe/MACS-MSc-Thesis-/blob/a06ba13b1c615e33e4507e0329d507e3315c4773/Figures/OTs.jpg "Overshooting Tops (Bedka et al., 2010)")

We predict the 1% most storm-intense nights with a hit rate of 85% and a false alarm rate of 15%. Further research will (i) expand on the deep learning application for these predictions, (ii) implement spatial predictions and (iii) use the SEVIRI images as input for our model (while keeping the OT dataset as prediction label).

<img src="https://github.com/SeppeLampe/MACS-MSc-Thesis-/blob/a06ba13b1c615e33e4507e0329d507e3315c4773/Figures/log%20temp.png" width=66% height=66% alt="Results from MSc Thesis">

## Table of Contents

<details>
  <summary>CreateTuples.py</summary>
    <a href="https://github.com/SeppeLampe/MSc-Thesis/blob/main/CreateTuples.py">This script</a> loops through the OT dataset and generates day-night tuples, which will be used for this project.
</details>

<details>
  <summary>CreateShards.py</summary>
    <a href="https://github.com/SeppeLampe/MSc-Thesis/blob/main/CreateShards.py">This script</a> loops through the day-night tuples and combines them into shards, which we can efficiently load into RAM via Webdataset.
</details>

<details>
  <summary>GetVariable.py</summary>
    <a href="https://github.com/SeppeLampe/MSc-Thesis/blob/main/GetVariable.py">This script</a> loops through the day-night tuples and extracts only one variable from them, and then stores them in netCDF4 files.
</details>

<details>
  <summary>DataLoader.py</summary>
    <a href="https://github.com/SeppeLampe/MSc-Thesis/blob/main/DataLoader.py">This script</a> divides the data into training, validation and test sets and creates Dataloader objects, which can efficiently loop through these subsets.
</details>

<details>
  <summary>Classifier.py</summary>
    <a href="https://github.com/SeppeLampe/MSc-Thesis/blob/main/Classifier.py">This script</a> specifies how training, validation and testing should be performed, when weights need to be updated (only during training) and determines the logging.
</details>

<details>
  <summary>CNN.py</summary>
    <a href="https://github.com/SeppeLampe/MSc-Thesis/blob/main/CNN.py">This script</a> provides an easy way to generate Convolutional Neural Networks (CNNs) in an object-oriented manner.
</details>

<details>
  <summary>Trainer.py</summary>
    <a href="https://github.com/SeppeLampe/MSc-Thesis/blob/main/Trainer.py">This script</a> determines the Trainer, which handles communication between the Dataloader, CNN and Classifier objects.
</details>

