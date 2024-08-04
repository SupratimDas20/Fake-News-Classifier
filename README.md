# Fake News Classifier

## Project Overview

This project aims to develop a robust fake news classifier using a dataset comprising 18,285 instances and 4 features. The classifier is designed to distinguish between genuine and fake news articles by leveraging machine learning and natural language processing (NLP) techniques.

## Key Features

- **Dataset**: The dataset consists of 18,285 instances with 4 key features, providing a foundation for building and evaluating the classifier.
- **Baseline Model**: Implemented logistic regression as a baseline model to establish a benchmark for performance evaluation.
- **Advanced Techniques**: Explored more sophisticated approaches, including Long Short-Term Memory (LSTM) neural networks, which are well-suited for sequence data analysis.
- **NLP Methods**: Leveraged NLP techniques such as tokenization, word embedding, and sentiment analysis to extract meaningful features from textual data, enhancing the model's ability to accurately classify news articles.

## Project Structure

- `data/`: Contains the dataset used for training and evaluation.
- `models/`: Includes the implementation of the logistic regression model and the LSTM neural network.
- `notebooks/`: Jupyter notebooks detailing the data exploration, feature extraction, and model development processes.
- `src/`: Source code for data preprocessing, feature extraction, and model training.
- `results/`: Stores the results of model evaluation and comparisons.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-classifier.git
   cd fake-news-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the data preprocessing and feature extraction:
   ```bash
   python src/preprocess.py
   ```

4. Train the baseline logistic regression model:
   ```bash
   python src/train_logistic_regression.py
   ```

5. Train the LSTM neural network:
   ```bash
   python src/train_lstm.py
   ```

6. Evaluate the models and compare performance:
   ```bash
   python src/evaluate_models.py
   ```

## Results

The project successfully demonstrated the ability to classify fake news using both traditional machine learning methods and advanced neural network techniques. The LSTM model, in particular, showed promising results, outperforming the baseline logistic regression model in terms of accuracy and other evaluation metrics.

## Conclusion

This project highlights the effectiveness of combining machine learning with NLP techniques to tackle the problem of fake news classification. The use of LSTM networks for sequence data analysis proved to be a valuable approach, offering significant improvements over simpler models.

## Future Work

- **Hyperparameter Tuning**: Further optimization of model hyperparameters to enhance performance.
- **Data Augmentation**: Exploring data augmentation techniques to improve model robustness.
- **Model Deployment**: Deploying the classifier as a web service for real-time fake news detection.

## Contact

For any questions or collaboration opportunities, feel free to reach out:

- Name: Supratim Das
- Email: supratimdaswork@gmail.com

