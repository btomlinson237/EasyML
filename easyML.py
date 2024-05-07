# easyML.py (EasyML Language Interpreter)

# Author: Bobby Tomlinson
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score
import joblib

# Responsible for retrieving variables and updating program state with new variables
def varmap(targetVar, state):
    if targetVar in state:
        return state[targetVar]
    else:
        raise ValueError("Error: Var not found")
    
# Accepts a datapath from user program as a string, cleans the string for use, and stores it in program state
def createDatapath(parts, state):
    var = parts[1]
    val = parts[3]
    for i in parts[4:]:
        val = ' '.join([val,i])

    file_path = val
    file_path = file_path.strip("'")
    
    # State update
    state[var] = file_path


# Loads tabular data as a variable for program use based on user-defined datapath variable (can accept csv or Excel file)
def createDataset(parts, state):
    utilizedPath = state[parts[4]]
    pathType = -1 # 0 for csv, 1 for xlsx, -1 by default

    # Checks for filetype of datapath
    if ".csv" in utilizedPath:
        pathType = 1

    if ".xlsx" in utilizedPath:
        pathType = 0


    # Dataframe is loaded based on file type
    if pathType == 0:
        userDf = pd.read_excel(utilizedPath)

    if pathType == 1:
        userDf = pd.read_csv(utilizedPath)
    
    # Represents user-defined variable name
    var = parts[1] 

    # State update
    state[var] = userDf


# Performs basic automatic cleaning of user-defined dataset 
def cleanDataset(parts, state):
    df_name = parts[1] # Variable name
    current_df = state[df_name] # Dataframe retrieved from program state
    new_df = current_df.copy() # New dataframe that will be cleaned

    # Cleans each column of dataset
    for col in current_df.columns:
        new_df = new_df.dropna(subset=[col]) # All rows with blank entries removed
        col_data_type = new_df[col].apply(type).mode().iloc[0] # Dataset determines average data type of column
        new_df = new_df[new_df[col].apply(type) == col_data_type] # Dataset removes all rows with erroneous data types (don't match column)

    # State is updated
    state[df_name] = new_df


# Converts a column letter (as given by Excel) into usable column index for target separation
def determineColumn(columnString):
    columnNum = -1
    columnChar = columnString[0]
    if columnChar.isalpha():
        columnChar = columnChar.lower()
        columnNum = ord(columnChar) - 97 # A starts at 0, B at 1, and so on...

    # If a column is actually provided as a number index, this is just casted as an int
    else:
        columnNum = int(columnString)

    return columnNum


# Creates and cross-validates classification models, informing the user of the metrics of the better model
# Models currently supported: Logistic Regression, Decision Tree Classifier
def catModel(modelParams, state):
    # Feature Engineering
    datasetName = modelParams[1] # Var Name
    targetColumnNum = modelParams[2] # Column Index

    userData = state[datasetName] # Dataframe pulled from state
    
    # Any column with text data is iteratively found and numerically encoded
    textColumns = userData.select_dtypes(include=['object']).columns.tolist()
    labelEncoders = {}
    for col in textColumns:
        labelEncoders[col] = LabelEncoder()
        userData[col] = labelEncoders[col].fit_transform(userData[col])

    # Feature and target separation
    y = userData.iloc[:, targetColumnNum]
    X = userData.drop(columns=[userData.columns[targetColumnNum]])

    # Train-test split of features and targets
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    # Logistic Regression Model and Decision Tree Classifier both instantiated
    modelLog = LogisticRegression(max_iter=1000) # prevents error related to lack of convergence
    modelDt = DecisionTreeClassifier()

    # Both models are fitted with the same training data
    modelLog.fit(X_train, y_train)
    modelDt.fit(X_train, y_train)

    # Both models generate classification predictions on the same test data
    yLogPred = modelLog.predict(X_test)
    yDtPred = modelDt.predict(X_test)

    # Accuracy, precision, and recall of each model are computed in order to cross-validate
    LogAccuracy = accuracy_score(y_test, yLogPred)
    DtAccuracy = accuracy_score(y_test, yDtPred)

    LogPrecision = precision_score(y_test, yLogPred)
    DtPrecision = precision_score(y_test, yDtPred)

    LogRecall = recall_score(y_test, yLogPred)
    DtRecall = recall_score(y_test, yDtPred)

    # Overall validation metrics are computed for each model to evaluate model effectiveness
    # Metric weights: 20% recall, 20% precision, 60% accuracy
    LogCompositeScore = (0.2 * LogRecall) + (0.2 * LogPrecision) + (0.6 * LogAccuracy)
    DtCompositeScore = (0.2 * DtRecall) + (0.2 * DtPrecision) + (0.6 * DtAccuracy)

    # Optimal model is determined through comparison of composite scores
    LogHigher = False
    if LogCompositeScore > DtCompositeScore:
        LogHigher = True

    # Results of optimal model are reported (in command line output)
    if (LogHigher):
        print("The optimal model type for classifying the target column is logistic regression")
        print("Accuracy: ", '{:.4f}'.format(LogAccuracy))
        print("Precision ", '{:.4f}'.format(LogPrecision))
        print("Recall: ", '{:.4f}'.format(LogRecall))
        return modelLog

    else:
        print("The optimal model type for classifying the target column is a decision tree classifier")
        print("Accuracy: ", '{:.4f}'.format(DtAccuracy))
        print("Precision ", '{:.4f}'.format(DtPrecision))
        print("Recall: ", '{:.4f}'.format(DtRecall))
        return modelDt
    
# Creates and cross-validates regression models, informing the user of the metrics of the better model
# Models currently supported: Linear Regression, Random Forest Regressor
def numModel(modelParams, state):
    # Feature Engineering
    datasetName = modelParams[1] # Var name
    targetColumnNum = modelParams[2] # Column Index

    userData = state[datasetName] # Dataframe pulled from state

    # Any column with text data is iterative found and numerically encoded
    textColumns = userData.select_dtypes(include=['object']).columns.tolist()
    labelEncoders = {}
    for col in textColumns:
        labelEncoders[col] = LabelEncoder()
        userData[col] = labelEncoders[col].fit_transform(userData[col])

    # Feature and target separation
    y = userData.iloc[:, targetColumnNum]
    X = userData.drop(columns=[userData.columns[targetColumnNum]])

    # Train-test split of features and targets
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    # Linear Regression Model and Random Forest Regressor both instantiated
    modelLin = LinearRegression()
    modelRf = RandomForestRegressor()

    # Both models are fitted with the same training data
    modelLin.fit(X_train, y_train)
    modelRf.fit(X_train, y_train)

    # Both models generate regression predictions on the same test data
    yLinPred = modelLin.predict(X_test)
    yRfPred = modelRf.predict(X_test)

    # MSE, MAE, and R-Squared of each model are computed (MSE and MAE normalized) in order to cross-validate
    LinMse = mean_squared_error(y_test, yLinPred)
    RfMse = mean_squared_error(y_test, yRfPred)
    minMse = min(LinMse,RfMse)
    rangeMse = max(LinMse,RfMse) - minMse
    finalLinMse = - ((LinMse - minMse)/rangeMse)
    finalRfMse = - ((RfMse - minMse)/rangeMse)

    LinMae = mean_absolute_error(y_test, yLinPred)
    RfMae = mean_absolute_error(y_test, yRfPred)
    minMae = min(LinMae,RfMae)
    rangeMae = max(LinMae,RfMae) - minMae
    finalLinMae = - ((LinMae - minMae)/rangeMae)
    finalRfMae = - ((RfMae - minMae)/rangeMae)
    
    LinR2 = r2_score(y_test, yLinPred)
    RfR2 = r2_score(y_test, yLinPred)

    # Overall validation metrics are computed for each model to evaluate model effectiveness
    # Metric weights: 25% Mean Squared Error, 25% Mean Absolute Error, 50% R-Squared
    LinCompositeScore = (0.25 * finalLinMse) + (0.25 * finalLinMae) + (0.5 * LinR2)
    RfCompositeScore = (0.25 * finalRfMse) + (0.25 * finalRfMae) + (0.5 * RfR2)

    # Optimal model is determined through comparison of composite scores
    linHigher = False
    if LinCompositeScore > RfCompositeScore:
        linHigher = True

    # Results of optimal model are reported (in command line output)
    if (linHigher):
        print("The optimal model type for a value in target column is a linear regressor")
        print("R Squared: ", '{:.2f}'.format(LinR2))
        print("Mean Absolute Error: ", '{:.2f}'.format(LinMae))
        return modelLin

    else:
        print("The optimal model type for predicting a value in target column is a random forest regressor")
        print("R Squared: ", '{:.2f}'.format(RfR2))
        print("Mean Absolute Error: ", '{:.2f}'.format(RfMae))
        return modelRf
    

# Parses model execution instruction and handles overall execution
def startModel(parts, state):
    # Extracts user-defined variable name for the model variable
    modelVarName = parts[1]

    modelCallString = parts[3]
    modelCallList = modelCallString.split('(', 1)
    
    # Extracts user-defined ML model type (classification or regression)
    predictionType = modelCallList[0]
    
    # Extracts user-defined dataset variable name
    datasetVariable = modelCallList[1]
    datasetList = datasetVariable.split(',', 1)
    datasetVariable = datasetList[0]
    
    # Extracts user-defined column to determine correct column index for ML target
    modelPredictionColumn = parts[5]
    modelPredictionColumnList = modelPredictionColumn.split(')', 1)
    modelPredictionColumn = modelPredictionColumnList[0]
    columnNum = determineColumn(modelPredictionColumn)

    # Numerically encodes user-model type
    modelType = -1 # 0 for categorical, 1 for numerical
    if 'CAT' in predictionType:
        modelType = 0
    
    if 'NUM' in predictionType:
        modelType = 1
    
    # Packages model type, dataset variable name, and target column index for the model creation function call
    modelParameters = [modelType,datasetVariable,columnNum]

    # Categorization model task is completed
    if modelType == 0:
        newModel = catModel(modelParameters, state)
        state[modelVarName] = newModel # state updated with new classification model

    # Regression model task is completed
    if modelType == 1:
        newModel = numModel(modelParameters, state)
        state[modelVarName] = newModel # state updated with new regression model


# Locally saves information from program state (either sklearn model or dataframe)
def downloadVariable(parts, state):
    dataType = parts[1] # Data type of variable to be saved
    varName = parts[2] # Variable name that should be saved
    data = state[varName] # data is extracted from program state

    # Saves dataframe
    if 'DATASET' in dataType:
        fileName = ''.join(['exported/',varName,'.csv']) # dataframe saved to "exported" folder as .csv
        print("Downloading a dataset with the name ", fileName)
        data.to_csv(fileName)

    if 'MODEL' in dataType:
        fileName = ''.join(['exported/',varName,'.sav']) # model saved to "exported folder as .sav
        print("Downloading a model with the name ", fileName)
        joblib.dump(data,fileName)


# Handles script execution at a high level, line-by-line based on the primary instruction type
def executeProgram(program):
    state = dict() # Program state initialized

    # Entire script string separated line-by-line
    for line in program.splitlines():
        parts = line.split()        
        # Primary instruction will always be the first word of the line
        instruction = parts[0]

        # Creates a datapath variable in program state
        if instruction == "DATAPATH":
            createDatapath(parts, state)

        # Creates a dataset (dataframe) variable in program state
        if instruction == "DATASET":
            createDataset(parts, state)

        # Cleans data within dataset inside the program state
        if instruction == "CLEAN":
            cleanDataset(parts, state)

        # Locally saves a variable from program state (model or dataframe)
        if instruction == "DOWNLOAD":
            downloadVariable(parts, state)

        # Creates, cross-validates, and reports optimal model based on desired parameters
        if instruction == "MODEL":
            startModel(parts, state)


# Command line is parsed to ensure adequate argument amount
if len(sys.argv) != 2:
    print("Usage: python easyML.py <ezml_file>")
    sys.exit(1)

# Extract the filename from the command line argument
scriptName = sys.argv[1]

try:
    # Script content is opened and read from the file
    with open(scriptName, 'r') as file:
        easyML_script = file.read()

except FileNotFoundError:
    # Handles case of non-existent script file
    print(f"Error: File '{scriptName}' not found.")
    sys.exit(1)


# Begins interpretation and execution of easyML script
executeProgram(easyML_script)



