# AutoGluon for Kaggle Competitions and Machine Learning Tasks

Welcome to the repository for **AutoGluon** examples! This repository contains a collection of Colab notebooks demonstrating the use of AutoGluon for various machine learning tasks, ranging from tabular classification and regression to multimodal tasks and time series forecasting.

Each notebook is structured to show how to set up and execute AutoGluon models, covering a wide variety of tasks. You will also find video tutorials explaining the process, as well as example outputs for reproducibility.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Notebooks Overview](#notebooks-overview)
   - [Tabular Classification/Regression](#tabular-classificationregression)
   - [Text Classification](#text-classification)
   - [Image Classification and Detection](#image-classification-and-detection)
   - [Image Segmentation](#image-segmentation)
   - [Semantic Matching](#semantic-matching)
   - [Multimodal Use Cases](#multimodal-use-cases)
   - [Time Series Forecasting](#time-series-forecasting)
   - [Object Detection](#object-detection)
3. [Running the Colabs](#running-the-colabs)
4. [Video Tutorials](#video-tutorials)
5. [Contributing](#contributing)
6. [License](#license)

## Getting Started

Before running the notebooks, ensure you have the following dependencies installed:

- Python 3.7 or above
- [AutoGluon](https://auto.gluon.ai/stable/index.html)

To install AutoGluon, run:

```bash
pip install autogluon
```
You can either clone the repository and run these notebooks in a local Jupyter environment or execute them directly on Colab. Each notebook includes a link to run in Colab, simplifying the process.

## Notebooks Overview

1. **Tabular Classification/Regression**
   - **Tabular Quick Start**: A quick introduction to tabular data classification and regression with AutoGluon.
   - **In-Depth Tabular**: A deeper dive into tabular models with advanced configurations.
   - **Multimodal Tabular**: Combining text, images, and tabular data for classification.
   - **Automatic Feature Engineering**: Automatically generate features from tabular data.
   - **Multi-Label Classification**: Handle datasets with multiple labels per instance.
   - **GPU Acceleration**: Leverage GPUs for faster training on large datasets.

2. **Text Classification**
   - **Sentiment Analysis & Sentence Similarity**: Classifying text data.
   - **Finetune Foundation Models**: Fine-tuning pretrained models.
   - **Named Entity Recognition (NER)**: Identify entities in text.

3. **Image Classification and Detection**
   - **Beginner Image Classification**: A beginner's guide to image classification.
   - **Zero Shot Classification**: Classify images without labeled data.
   - **Object Detection**: Detect objects in images using the COCO dataset.

4. **Image Segmentation**
   - **Beginner Semantic Segmentation**: Learn how to segment images into meaningful parts.

5. **Semantic Matching**
   - **Image-to-Image Matching**: Match similar images.
   - **Text-to-Text Matching**: Match similar text.
   - **Image-Text Matching**: Match images with text descriptions.

6. **Multimodal Use Cases**
   - **Multimodal Mixed Types**: Combine text and tabular data in machine learning models.
   - **Image, Text, and Tabular Prediction**: A comprehensive multimodal prediction.

7. **Time Series Forecasting**
   - **Forecasting with Chronos**: Forecast time series data with AutoGluon.

8. **Object Detection**
   - **Object Detection Colab**: A practical guide to object detection using the COCO dataset.

## Running the Colabs

To run the notebooks:

1. Open the notebook links provided above.
2. Ensure that your runtime is set to GPU for notebooks that require it (especially for tasks like object detection).
3. Run each cell sequentially and review the outputs. The outputs are included in the notebooks for reference.

## Video Tutorials

For each notebook, you can find a 1-minute video walkthrough explaining the code and its purpose. The video tutorials are split into parts for ease of navigation.

- [Link to Video Tutorials (Part 1)](URL to video)
- [Link to Video Tutorials (Part 2)](URL to video)

## Contributing

Feel free to fork this repository and contribute by opening a pull request. Ensure all contributions follow the structure and are well documented.

## License

This project is licensed under the MIT License.

  
