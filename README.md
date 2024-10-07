# AutoGluon for Kaggle Competitions and Machine Learning Tasks

Welcome to the repository for **AutoGluon** examples! This repository contains a collection of Colab notebooks demonstrating the use of AutoGluon for various machine learning tasks, ranging from tabular classification and regression to multimodal tasks and time series forecasting. I have added links to each Colabs in the **Notebooks Overview** section.

Each notebook is structured to show how to set up and execute AutoGluon models, covering a wide variety of tasks. You will also find video tutorials explaining the process, as well as example outputs for reproducibility.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Projects Overview](#projects-overview)
   - [House Prices - Advanced Regression Techniques](#house-prices-advanced-regression-techniques)
   - [IEEE Fraud Detection](#ieee-fraud-detection)
3. [Notebooks Overview](#notebooks-overview)
   - [Tabular Classification/Regression](#tabular-classificationregression)
   - [Text Classification](#text-classification)
   - [Image Classification and Detection](#image-classification-and-detection)
   - [Image Segmentation](#image-segmentation)
   - [Semantic Matching](#semantic-matching)
   - [Multimodal Use Cases](#multimodal-use-cases)
   - [Time Series Forecasting](#time-series-forecasting)
   - [Object Detection](#object-detection)
4. [Running the Colabs](#running-the-colabs)
5. [Video Tutorials](#video-tutorials)
6. [Contributing](#contributing)
7. [License](#license)

## Getting Started

To run the notebooks, you can clone the repository and run these notebooks in a local Jupyter environment or execute them directly on Colab.

- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [GitHub Repository](https://github.com/autogluon/autogluon)

## Projects Overview

### House Prices Advanced Regression Techniques
- **[House Prices Advanced Regression Techniques](https://colab.research.google.com/drive/1jcSkPAwMDLnP285LIAi33BNEYpt7jDZg?usp=share_link)**:  A project for predicting house prices and submitting using AutoGluon.

### IEEE Fraud Detection
- **[IEEE Fraud Detection](https://colab.research.google.com/drive/1XWAb2Ug2ke8b7UrtcFSMU_bJ4unhMz0W?usp=share_link)**: A project for detecting fraudulent transactions in the IEEE dataset using AutoGluon.

## Notebooks Overview

1. **Tabular Classification/Regression**
   - **[Tabular Quick Start](https://colab.research.google.com/drive/1cufmzfci6qgjTSIrjs1tUxkuGFdJxm5J?usp=share_link)**: A quick introduction to tabular data classification and regression with AutoGluon.
   - **[In-Depth Tabular](https://colab.research.google.com/drive/1iwmAxO-zxAq5fGEKvI60Y0JuJt5dF6Q9?usp=share_link)**: A deeper dive into tabular models with advanced configurations.
   - **[Multimodal Tabular](https://colab.research.google.com/drive/1PH1RqTcpFGK1igJkF4cRqSMjgvaOPGkB?usp=share_link)**: Combining text, images, and tabular data for classification.
   - **[Automatic Feature Engineering](https://colab.research.google.com/drive/1hiFgys32B0R_g6Ft4Uojq4efFT2iuBA8?usp=share_link)**: Automatically generate features from tabular data.
   - **[Multi-Label Classification](https://colab.research.google.com/drive/1W06OzQazySQ5JUFRboXiC-cOGk0igcsK?usp=sharing)**: Handle datasets with multiple labels per instance.
   - **[GPU Acceleration](https://colab.research.google.com/drive/14Er4rBw_1DoTLrLFylndcGCBXCj_YiN7?usp=share_link)**: Leverage GPUs for faster training on large datasets.

2. **Text Classification**
   - **[Sentiment Analysis & Sentence Similarity](https://colab.research.google.com/drive/1wyxb--gO63CGcpeexVZJQX9w-3gQf5qf?usp=share_link)**: Classifying text data.
   - **[Finetune Foundation Models](https://colab.research.google.com/drive/1b8xI4FoBnDRUtF5HqQoZeCUQy6rQ0YNu?usp=share_link)**: Fine-tuning pretrained models.
   - **[Named Entity Recognition (NER)](https://colab.research.google.com/drive/1c12hhZFGaJjlT0kLRzVFW5sUY9u0HhmH?usp=share_link)**: Identify entities in text.

3. **Image Classification and Detection**
   - **[Beginner Image Classification](https://colab.research.google.com/drive/1L3LqGtNw-NCdiRM58BkbRT6wdpszdTJR?usp=sharing)**: A beginner's guide to image classification.
   - **[Zero Shot Classification](https://colab.research.google.com/drive/18slXTxweIktUfBpGyxwU7IbhX0htMS2x?usp=sharing)**: Classify images without labeled data.
   - **[Object Detection](https://colab.research.google.com/drive/1qR-Dv5X5zXBacrNImJ15CJyBjTahVFpQ?usp=share_link)**: Detect objects in images using the COCO dataset.

4. **Image Segmentation**
   - **[Beginner Semantic Segmentation](https://colab.research.google.com/drive/1ioaf3bGBvbSEAjBTe8YP_v0efkibt8uY?usp=sharing)**: Learn how to segment images into meaningful parts.
   - **[Document Classification](https://colab.research.google.com/drive/1UVreZldKX7RpKFMNGomfEOVGuY-KNI1e?usp=share_link)**: Classify document images.
   - **[PDF Classification](https://colab.research.google.com/drive/1tqDQIo5WiGfKq6E57aR8b3oA4iDmIKaE?usp=sharing)**: Classify PDF files.

5. **Semantic Matching**
   - **[Image-to-Image Matching](https://colab.research.google.com/drive/1rH_Z6t7LfxZnhc5WzKAO2lsKe3koaQ1k?usp=share_link)**: Match similar images.
   - **[Text-to-Text Matching](https://colab.research.google.com/drive/1Nee1-thRna2n0IFbqcL5ntIknPGsoI7t?usp=sharing)**: Match similar text.
   - **[Image-Text Matching](https://colab.research.google.com/drive/16p0Hb9gF6yysNsJMar7q_Q_GXPU0xOjd?usp=share_link)**: Match images with text descriptions.
   - **[Zero Shot Image-Text Matching](https://colab.research.google.com/drive/1v25DIseBdgD6mR_ZglHh_lXD4Z5pxmdm?usp=share_link)**: Match images with text descriptions without labeled data.
   - **[Text Semantic Search](https://colab.research.google.com/drive/1yseHILZKmmsEmt2iWl8RBrWtOtl_L4cv?usp=sharing)**: Perform semantic search on text data.

6. **Multimodal Use Cases**
   - **[Multimodal Mixed Types](https://colab.research.google.com/drive/1aNO8yDrpvS9HbwUtQnpa4V22f1jQXrwl?usp=share_link)**: Combine text and tabular data in machine learning models.
   - **[Image, Text, and Tabular Prediction](https://colab.research.google.com/drive/1YCKNO-DbshS49v8A6MpA2VFtx-crBw0v?usp=share_link)**: A comprehensive multimodal prediction.
   - **[Multimodal NER](https://colab.research.google.com/drive/11yYI5y4boY3l_ykf1vrnff8sPRtMHYMc?usp=share_link)**: Named Entity Recognition on multimodal data.

7. **Time Series Forecasting**
   - **[Forecasting In-Depth](https://colab.research.google.com/drive/1bowUgZ1-Qbmw3nREuAQUdMGUZK0tVmvh?usp=sharing)**: An in-depth guide to time series forecasting.
   - **[Forecasting with Chronos](https://colab.research.google.com/drive/1HiObqyF1FXbUclRJ1XV2jw3VCp2nSWo5?usp=sharing)**: Forecast time series data with AutoGluon and Chronos.

8. **Object Detection**
   - **[Object Detection Colab](https://colab.research.google.com/drive/1qR-Dv5X5zXBacrNImJ15CJyBjTahVFpQ?usp=share_link)**: A practical guide to object detection using the COCO dataset.

## Running the Colabs

To run the notebooks:

1. Open the notebook links provided above.
2. Ensure that your runtime is set to GPU for notebooks that require it (especially for tasks like object detection).
3. Run each cell sequentially and review the outputs. The outputs are included in the notebooks for reference.

## Video Tutorials

For each notebook, you can find a 1-minute video walkthrough explaining the code and its purpose. The video tutorials are split into parts for ease of navigation.

- [Link to Video Tutorials (Part 1)](URL to video)
- [Link to Video Tutorials (Part 2)](URL to video)

## Contribution

This project is part of an academic assignment. While it's not open for direct contributions, feedback and suggestions are welcome through the issue tracker.
