# AutoGluon for Kaggle Competitions and Machine Learning Tasks

Welcome to the repository for **AutoGluon** examples! This repository contains a collection of Colab notebooks demonstrating the use of AutoGluon for various machine learning tasks, ranging from tabular classification and regression to multimodal tasks and time series forecasting.

Each notebook is structured to show how to set up and execute AutoGluon models, covering a wide variety of tasks. You will also find video tutorials explaining the process, as well as example outputs for reproducibility.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Projects Overview](#projects-overview)
   - [California House Price Prediction](#california-house-price-prediction)
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

### California House Price Prediction
- **[California House Price Prediction](./CaliforniaHousePred.ipynb)**: A project for predicting house prices in California using AutoGluon.

### IEEE Fraud Detection
- **[IEEE Fraud Detection](./IEEEFraud.ipynb)**: A project for detecting fraudulent transactions in the IEEE dataset using AutoGluon.

## Notebooks Overview

1. **Tabular Classification/Regression**
   - **[Tabular Quick Start](./a_2_AutoGluonTabular_QuickStart.ipynb)**: A quick introduction to tabular data classification and regression with AutoGluon.
   - **[In-Depth Tabular](./a_1_AutoGluonTabular_InDepth.ipynb)**: A deeper dive into tabular models with advanced configurations.
   - **[Multimodal Tabular](./a_3_AutoGluonTabular-Multimodal.ipynb)**: Combining text, images, and tabular data for classification.
   - **[Automatic Feature Engineering](./a_4_AutoGluonTabular-featureEngineering.ipynb)**: Automatically generate features from tabular data.
   - **[Multi-Label Classification](./a_5_AutoGluonTabular-Multilabel.ipynb)**: Handle datasets with multiple labels per instance.
   - **[GPU Acceleration](./a_6_AutoGluonTabular-GPU.ipynb)**: Leverage GPUs for faster training on large datasets.

2. **Text Classification**
   - **[Sentiment Analysis & Sentence Similarity](./b_1_Beginner-text.ipynb)**: Classifying text data.
   - **[Finetune Foundation Models](./b_2_Multilingual-text.ipynb)**: Fine-tuning pretrained models.
   - **[Named Entity Recognition (NER)](./b_3_NER.ipynb)**: Identify entities in text.

3. **Image Classification and Detection**
   - **[Beginner Image Classification](./d_1_BeginnerImage-cls.ipynb)**: A beginner's guide to image classification.
   - **[Zero Shot Classification](./d_2_Clip-zeroshot.ipynb)**: Classify images without labeled data.
   - **[Object Detection](./d_3_QuickStart_coco.ipynb)**: Detect objects in images using the COCO dataset.

4. **Image Segmentation**
   - **[Beginner Semantic Segmentation](./e_1_BeginnerSemantic_Seg.ipynb)**: Learn how to segment images into meaningful parts.
   - **[Document Classification](./e_2_DocumentClassification.ipynb)**: Classify document images.
   - **[PDF Classification](./e_3_PDFClassification.ipynb)**: Classify PDF files.

5. **Semantic Matching**
   - **[Image-to-Image Matching](./f_1_Image2Image-Matching.ipynb)**: Match similar images.
   - **[Text-to-Text Matching](./f_2_Text2Text-Matching.ipynb)**: Match similar text.
   - **[Image-Text Matching](./f_3_Image_Text-Matching.ipynb)**: Match images with text descriptions.
   - **[Zero Shot Image-Text Matching](./f_4_ZeroShot_Img_Text-Matching.ipynb)**: Match images with text descriptions without labeled data.
   - **[Text Semantic Search](./f_5_Text_Semantic_Search.ipynb)**: Perform semantic search on text data.

6. **Multimodal Use Cases**
   - **[Multimodal Mixed Types](./g_1_Multimodal_Text_Tabular.ipynb)**: Combine text and tabular data in machine learning models.
   - **[Image, Text, and Tabular Prediction](./g_2_Beginner_Multimodal.ipynb)**: A comprehensive multimodal prediction.
   - **[Multimodal NER](./g_3_Multimodal_NER.ipynb)**: Named Entity Recognition on multimodal data.

7. **Time Series Forecasting**
   - **[Forecasting In-Depth](./h_1_Forecasting-Indepth.ipynb)**: An in-depth guide to time series forecasting.
   - **[Forecasting with Chronos](./h_2_Forecasting-Chronos.ipynb)**: Forecast time series data with AutoGluon and Chronos.

8. **Object Detection**
   - **[Object Detection Colab](./d_3_QuickStart_coco.ipynb)**: A practical guide to object detection using the COCO dataset.

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
