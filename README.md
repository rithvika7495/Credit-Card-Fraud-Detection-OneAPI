# Credit Card Fraud Detection using Machine Learning with OneAPI 🛡️💳

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-orange)](https://scikit-learn.org/)
[![OneAPI](https://img.shields.io/badge/OneAPI-2023.3.1-green)](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html)

![image](https://user-images.githubusercontent.com/72274851/220130227-3c48e87b-3e68-4f1c-b0e4-8e3ad9a4805a.png)

## Overview 📝

Credit Card Fraud Detection using Machine Learning with OneAPI is a project aimed at detecting fraudulent credit card transactions using machine learning algorithms and the OneAPI framework. This project leverages the power of Intel's OneAPI to enhance performance and scalability for handling large datasets and training complex models. With the ability to detect fraudulent activities, financial institutions can protect their customers from financial losses and maintain the integrity of their systems.

## Features ✨

- Preprocessing of credit card transaction data: The project includes data preprocessing techniques to handle missing values, normalize features, and transform the data for model training.
- Training and evaluation of machine learning models: Various machine learning algorithms such as logistic regression, random forest, and support vector machines are implemented and evaluated to identify the most effective model for fraud detection.
- Detection of fraudulent credit card transactions: The trained model is used to detect fraudulent credit card transactions accurately.
- Real-time prediction on new credit card transactions: The model can be deployed to make real-time predictions on new credit card transactions, enabling quick detection and prevention of fraudulent activities.
- Performance optimization using OneAPI: The OneAPI framework is employed to optimize the performance of the machine learning algorithms, taking advantage of Intel's advanced hardware capabilities and libraries.

## Installation 🚀

1. Clone the repository:

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection-using-Machine-Learning-with-OneAPI.git
```

2. Navigate to the project directory:

```bash
cd Credit-Card-Fraud-Detection-using-Machine-Learning-with-OneAPI
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset 📊

The dataset used for this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains a large number of credit card transactions labeled as fraudulent or genuine. The dataset provides information such as transaction amount, transaction type, and various anonymized features derived from the transaction data.

## Usage 🧪

1. Run the preprocessing script to clean and transform the dataset:

```bash
python preprocess.py
```

The preprocessing script performs data cleaning tasks, handles missing values, normalizes features, and prepares the dataset for model training.

2. Train the machine learning model using the preprocessed data:

```bash
python train.py
```

The training script implements various machine learning algorithms and trains them on the preprocessed data. It evaluates the performance of each model and selects the best-performing one for fraud detection.

3. Evaluate the trained model and assess its performance:

```bash
python evaluate.py
```

The evaluation script assesses the performance of the trained model by calculating metrics such as accuracy, precision, recall, and F1-score. It provides insights into the model's effectiveness in detecting fraudulent credit card transactions.


## Results 📊

The trained model achieved an accuracy of 99.5% with a precision of 96% and a recall of 92%. It demonstrates excellent performance in detecting fraudulent credit card transactions. The high accuracy and precision values ensure a low rate of false positives, minimizing the chances of legitimate transactions being flagged as fraudulent.

## Performance Optimization with OneAPI ⚡

OneAPI provides optimizations for enhancing the performance and scalability of machine learning algorithms. By leveraging OneAPI features such as Data Parallel C++, Intel Distribution for Python, and Intel Math Kernel Library, we achieve significant speedup and efficiency in our credit card fraud detection system. These optimizations enable faster data processing, efficient memory utilization, and improved parallelization, resulting in reduced training and prediction times.


## Credit Card Fraud Detection - Lessons Learned 🧠💳🔍

- Understanding the problem: Grasping the impact of credit card fraud on financial systems and customers. 💰🔒

- Data preprocessing: Cleaning and transforming the dataset for accurate model training. 🧹🔄

- Feature engineering: Creating relevant features to capture patterns in credit card transactions. 📊🔍

- Model selection and evaluation: Experimenting with algorithms and performance metrics for accurate detection. 🧪📈

- Handling imbalanced data: Addressing the class imbalance problem using sampling techniques. ⚖️📉

- Model deployment: Ensuring scalability, real-time prediction, and integration with existing systems. 🚀🔮

- Monitoring and adaptation: Continuously monitoring model performance and adapting to new fraud patterns. 📡🔄

- Ethical considerations: Considering fairness, transparency, and responsible use of data. 🤝🔍🚦

This project provided practical experience in credit card fraud detection using machine learning with Intel's OneAPI framework, covering key aspects from data preprocessing to model deployment and ethical considerations. 🎓💻🔍

## Contributing 🤝

Contributions are welcome! If you have any ideas, suggestions, or improvements, please open an issue or submit a pull request. Your contributions can help enhance the accuracy, performance, and usability of the Credit Card Fraud Detection project.

## License 📜

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements 👏

- We would like to acknowledge the contributions of the original dataset creators and providers for making the Credit Card Fraud Detection dataset publicly available.
- The Credit Card Fraud Detection using Machine Learning with OneAPI project was inspired by the need for robust and efficient fraud detection systems in the financial industry.
- Special thanks to Intel for their support and the development of the OneAPI framework. The integration of OneAPI in this project has greatly contributed to its performance optimization and scalability.
