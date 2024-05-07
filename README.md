# EasyML: A Window into AI by Bobby Tomlinson

## Overview
EasyML is a programming language designed to provide an accessible medium for implementing simple machine learning models. In a world where AI and machine learning are major areas of interest across industries, EasyML aims to empower individuals from various backgrounds to explore predictive analytics using their data effortlessly. Whether you're a professional with limited technical skills or a seasoned programmer, EasyML offers a seamless and intuitive approach to leveraging data for predictive modeling.

## Design
### Syntax
EasyML's syntax is straightforward and tailored for ease of use. It avoids complex control structures and functions to prioritize readability and simplicity. This approach allows even novice users to write and understand machine learning code effectively.

### Semantics
EasyML supports essential machine learning-specific data types and executes instructions sequentially, making it easy to follow and interpret.

## Features
EasyML primarily focuses on supervised learning tasks, and offers a range of features aimed at simplifying the machine learning process:
- **Data Handling**: Users can work with tabular data in CSV or Excel formats.
- **Data Cleaning**: Built-in functions enable users to clean datasets effortlessly.
- **Model Training**: EasyML supports supervised learning for both numeric and categorical predictions.
- **Model Evaluation**: Users can assess model performance using metrics tailored to the model type, such as accuracy, recall, mean absolute error, and R-squared.
- **Model Selection**: EasyML trains multiple models and selects the one with the highest composite metric score for optimal performance.

## How to Run EasyML Code
1. Ensure you have Python 3 installed on your system.
2. Clone this EasyML repository.
3. Navigate to the directory containing the EasyML interpreter (`easyML.py`) and your EasyML code files.
4. Unzip trainingData.zip, and ensure contents are located in the main project directory.
6. Open a terminal or command prompt.
7. Install the dependencies using the following command: Run  in the command
  ```pip3 install -r requirements.txt```
8. Run the EasyML code using the following command:
  ```python3 easyML.py samplecodename.ezml```


## Sample Implementations

### 1. Data Cleaning - sample1.ezml

**Explanation:**
- This sample code showcases working with Excel data.
- The dataset 'running.xlsx' is loaded into memory.
- Data cleaning is performed to prepare the dataset for analysis.
- The cleaned dataset is then saved for further analysis or modeling.
- Blank entries are removed, and entries that differ from a column's average data type are removed.


### 2. Predicting Categorical Values (Classification) - sample2.ezml

**Explanation:**
- This sample code demonstrates the process of predicting categorical values using a classification model.
- The `DATAPATH` command specifies the path to the dataset file, which is 'Titanic.csv'.
- The `DATASET` command loads the dataset into memory as `userDataCat`.
- The `CLEAN` command cleans the dataset to remove any missing values or inconsistencies.
- With the `MODEL` command, a regression model (`PREDICT_CAT`) is trained using the cleaned dataset, with the target variable located in column J.
- Finally, the `DOWNLOAD` commands save the cleaned dataset and the trained classification model for future use.


### 3. Predicting Numeric Values (Regression) - sample3.ezml

**Explanation:**
- This sample code demonstrates the process of predicting numeric values using a regression model.
- The `DATAPATH` command specifies the path to the dataset file, which is 'housing.csv'.
- The `DATASET` command loads the dataset into memory as `userDataNum`.
- The `CLEAN` command cleans the dataset to remove any missing values or inconsistencies.
- With the `MODEL` command, a regression model (`PREDICT_NUM`) is trained using the cleaned dataset, with the target variable located in column I.
- Finally, the `DOWNLOAD` commands save the cleaned dataset and the trained regression model for future use.
