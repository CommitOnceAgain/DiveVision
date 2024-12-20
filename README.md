# DiveVision

This project aims at exploring solutions regarding **Underwater Image Restoration**.

At this time, the goal is to use off-the-shelves models and test their performance on the [LSUI dataset](https://bianlab.github.io/data.html).

The project will be enhanced, and here are some perspectives:
- build and train a model (probably a ViT)
- monitor the model performances (MLFlow?)
- make a FastAPI server to serve the model
- build a web app
- build a mobile app

## Installation

### Prerequisites

Python 3.12<br>
[Poetry](https://python-poetry.org/)<br>
Clone the repository

### Steps

1. ```poetry install```
2. Try and run the notebook `test_model.ipynb` to see if everything is working, using the poetry environment

## Resources

- **U-Shape Transformer for Underwater Image Enhancement. Peng L., Zhu C., Bian L., 2021**
    - [Github](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement)
    - [Paper](https://arxiv.org/abs/2111.11843)
