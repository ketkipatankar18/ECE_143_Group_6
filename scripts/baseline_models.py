# Importing relevant packages
import re
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise import SVD, Reader, Dataset, accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV as GridSearchCV_from_sklearn

# Loading Data
# data = pd.read_json('/Applications/anaconda3/envs/ece143/renttherunway_final_data.json', lines=True)
data = pd.read_json('/Applications/anaconda3/envs/ece143/renttherunway_final_data.json', lines=True)
# data = pd.read_json('renttherunway_final_data.json', lines=True)

"""**1. Data Cleaning**


"""

# Rename columns containing space character
data.rename(columns={'bust size': 'bust_size', 'rented for': 'rented_for', 'body type':'body_type'}, inplace=True)

# Previewing data
data.sample(5)

data.shape

data.item_id.count()

# Overview of dataframe structure
data.info()

# Analyze the missing data for each column
data.isna().sum()

# Get the most common value in each colum
size_null = ['bust_size', 'weight', 'rented_for', 'body_type', 'height']
for i in size_null:
  i_sum = data.groupby(i).size().sort_values(ascending=False)
  print(i_sum.head(1))

"""1.1 Dealing with missing values


"""

# Use the most common value in each column to fill missing data
data['bust_size'].fillna('34b', inplace=True)
data['weight'].fillna('130lbs', inplace=True)
data['rating'].fillna(float(data['rating'].median()), inplace=True)
data['rented_for'].fillna('wedding', inplace=True)
data['body_type'].fillna('hourglass', inplace=True)
data['height'].fillna('5\' 4"', inplace=True)
data['age'].fillna(float(data['age'].median()), inplace=True)

# Check whether still exist missing columns
data.isna().sum()

"""1.2 Data Analysis before converting type"""

# Body Type Divided by Fit Categories
plt.figure(figsize=(15, 8))
purpose_fit = pd.crosstab(data['body_type'], data['fit'])
purpose_fit.plot(kind='bar', stacked=False)
plt.title('Body Type Divided by Fit Categories')
plt.xlabel('Body Type')
plt.ylabel('Count')
plt.legend(['Small', 'Fit', 'Large'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 20 Clothing Categories by Rental Volume and Fit
plt.figure(figsize=(12, 10))
category_counts = pd.crosstab(data['category'], data['fit'])
top_20_categories = category_counts.sum(axis=1).sort_values(ascending=True).tail(20).index
category_counts.loc[top_20_categories].plot(kind='barh', stacked=False)
plt.title('Top 20 Clothing Categories by Rental Volume and Fit')
plt.xlabel('Count')
plt.ylabel('Category')
plt.legend(['Small', 'Fit', 'Large'])
plt.tight_layout()
plt.show()


# Top 20 Bust sizes by Rental Volume and Fit
plt.figure(figsize=(12, 10))
category_counts = pd.crosstab(data['bust_size'], data['fit'])
top_20_categories = category_counts.sum(axis=1).sort_values(ascending=True).tail(20).index
category_counts.loc[top_20_categories].plot(kind='barh', stacked=False)
plt.title('Top 20 Bust sizes by Rental Volume and Fit')
plt.xlabel('Count')
plt.ylabel('Bust Size')
plt.legend(['Small', 'Fit', 'Large'])
plt.tight_layout()
plt.show()

# Rentals by Purpose divided by Fit Categories
plt.figure(figsize=(15, 8))
purpose_fit = pd.crosstab(data['rented_for'], data['fit'])
purpose_fit.plot(kind='bar', stacked=False)
plt.title('Rentals by Purpose Divided by Fit Categories')
plt.xlabel('Rental Purpose')
plt.ylabel('Count')
plt.legend(['Small', 'Fit', 'Large'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""##1.3 Converting data type"""

def bust_size_to_numeric(bust_size):
    """
    Function to convert bust size string to numeric, keeping NaN as NaN.

    Args:
        bust_size: The bust size string

    Returns:
        float or NaN: The numeric bust size or NaN if input is NaN or invalid.
    """

    if pd.isna(bust_size):
        return np.nan

    try:
        band_size = int(re.search(r'\d+', bust_size).group())
        cup_part = re.sub(r'\d+', '', bust_size).lower()

        cup_mapping = {
            'aa': 0.5, 'a': 1, 'b': 2, 'c': 3, 'd': 4,
            'dd': 5, 'ddd': 6, 'e': 5, 'f': 6,
            'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11
        }
        if '+' in cup_part:
            cup_part = cup_part.replace('+', '')
            cup_value = cup_mapping.get(cup_part, 0) + 0.5
        elif '/' in cup_part:
            cup_part = cup_part.split('/')[0]
            cup_value = cup_mapping.get(cup_part, 0)
        else:
            cup_value = cup_mapping.get(cup_part, 0)

        return band_size + cup_value

    except (AttributeError, ValueError):
        return np.nan

def rented_for_to_numeric(rented_for):
    """
    Function to convert rented for string to numeric, keeping NaN as NaN.

    Args:
        rented_for: The rented for string

    Returns:
        float or NaN: The numeric rented for or NaN if input is NaN or invalid.
    """
    if pd.isna(rented_for):
        return np.nan

    rented_for_mapping = {
        'vacation': 1,
        'other': 2,
        'party': 3,
        'formal affair': 4,
        'wedding': 5,
        'date': 6,
        'everyday': 7,
        'work': 8,
        'party: cocktail': 9
    }

    return rented_for_mapping.get(rented_for, np.nan)

def body_type_to_numeric(body_type):
    """
    Function to convert body type string to numeric, keeping NaN as NaN.

    Args:
        body_type: The body type string

    Returns:
        float or NaN: The numeric body type or NaN if input is NaN or invalid.
    """
    if pd.isna(body_type):
        return np.nan

    body_type_mapping = {
        'hourglass': 1,
        'straight & narrow': 2,
        'pear': 3,
        'athletic': 4,
        'full bust': 5,
        'petite': 6,
        'apple': 7
    }

    return body_type_mapping.get(body_type, np.nan)

def category_to_numeric(category):
    """
    Function to convert category string to numeric, keeping NaN as NaN.

    Args:
        category: The category string

    Returns:
        float or NaN: The numeric category or NaN if input is NaN or invalid.
    """

    if pd.isna(category):
        return np.nan

    category_mapping = {
        'romper': 1, 'gown': 2, 'sheath': 3, 'dress': 4, 'leggings': 5,
        'top': 6, 'jumpsuit': 7, 'sweater': 8, 'jacket': 9, 'shirtdress': 10,
        'maxi': 11, 'shift': 12, 'pants': 13, 'shirt': 14, 'mini': 15,
        'skirt': 16, 'pullover': 17, 'blouse': 18, 'suit': 19, 'coat': 20,
        'trench': 21, 'bomber': 22, 'cape': 23, 'blazer': 24, 'vest': 25,
        'duster': 26, 'ballgown': 27, 'tank': 28, 'poncho': 29, 'frock': 30,
        'tunic': 31, 'cardigan': 32, 'culottes': 33, 'down': 34, 'trouser': 35,
        'midi': 36, 'pant': 37, 'legging': 38, 'print': 39, 'knit': 40,
        'culotte': 41, 'sweatshirt': 42, 'peacoat': 43, 'kaftan': 44,
        'overalls': 45, 'jogger': 46, 'tee': 47, 'combo': 48, 'henley': 49,
        'cami': 50, 'blouson': 51, 'turtleneck': 52, 'trousers': 53,
        'overcoat': 54, 'hoodie': 55, 't-shirt': 56, 'caftan': 57,
        'tight': 58, 'kimono': 59, 'for': 60, 'crewneck': 61, 'skirts': 62,
        'parka': 63, 'buttondown': 64, 'skort': 65, 'sweatershirt': 66,
        'sweatpants': 67, 'jeans': 68
    }

    return category_mapping.get(category, np.nan)

# Convert all categorical features to numberical features
print('Unique values for bust_size before conversion:',data['bust_size'].unique())
print()
data['bust_size'] = data['bust_size'].apply(bust_size_to_numeric)
print('Unique values for bust_size after conversion:',data['bust_size'].unique())
print()
print('Unique values for rented_for before conversion: ',data['rented_for'].unique())
print()
data['rented_for'] = data['rented_for'].apply(rented_for_to_numeric)
print('Unique values for rented_for after conversion:',data['rented_for'].unique())
print()
print('Unique values for body_type before conversion:',data['body_type'].unique())
print()
data['body_type'] = data['body_type'].apply(body_type_to_numeric)
print('Unique values for body_type after conversion:',data['body_type'].unique())
print()
print('Unique values for category before conversion:',data['category'].unique())
print()
data['category'] = data['category'].apply(category_to_numeric)
print('Unique values for category after conversion:',data['category'].unique())

# Additional date and string specific conversions
data['review_date'] = pd.to_datetime(data['review_date'])
data['weight'] = data.weight.str.replace('lbs', '')
data['weight'] = data['weight'].astype('float64')

# Drop columns that are not used
data = data.drop(['review_text','review_summary','review_date'], axis=1)

def height_to_cm(height):
    """
    Function to convert height from feet and inch representation to centimeters, keeping NaN as NaN.

    Args:
        height: The height string

    Returns:
        float or NaN: The numeric height in centimeters or NaN if input is NaN or invalid.
    """
    # Split the height string like '5' 2" into feet and inches
    feet, inches = height.split("'")
    inches = inches.replace('"', '').strip()  # Remove the inch symbol and strip whitespace
    feet = int(feet.strip())  # Convert feet to integer
    inches = int(inches.strip())  # Convert inches to integer

    # Convert to cm
    height_cm = (feet * 30.48) + (inches * 2.54)
    return height_cm

# Example DataFrame
data['height'] = data['height'].apply(height_to_cm)

# Convert fit column data into numerical values
data_cor = data.replace({'fit': 2, 'small': 1, 'large': 3})

# Computing the Pearson's Correlation Matrix for all numerical features
cor = data_cor.corr(method ='pearson')

plt.figure(figsize=(24,20))
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

### Filter the outliers
print('Number of entries with age above 100:',np.sum([data['age'].unique()>=100]))
data = data[data.age <= 100]
print('Number of entries with size below 22:',np.sum([data['size'].unique()<=22]))
data = data[data['size'] <= 22]

# Reseting row labels after filtering out outliers
data.reset_index(drop=True,inplace= True)

"""Correlation, distribution and box plot"""

def create_eda_analysis(data):
    """
    Create comprehensive EDA visualizations and tables for the cleaned dataset.
    """

    # Dataset Basic Parameters Table
    basic_params = {
        'Number of Transactions': len(data),
        'Number of Users': data['user_id'].nunique(),
        'Number of Items': data['item_id'].nunique(),
        'Small Percentage': (data['fit'] == 0).mean() * 100,
        'Fit Percentage': (data['fit'] == 1).mean() * 100,
        'Large Percentage': (data['fit'] == 2).mean() * 100
    }

    params_df = pd.DataFrame.from_dict(basic_params, orient='index', columns=['Value'])
    print("\nDataset Basic Parameters:")
    print(params_df)

    # Standard numerical distributions
    numerical_features = ['age', 'weight', 'height', 'bust_size', 'rating', 'size']
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for idx, feature in enumerate(numerical_features):
        row = idx // 3
        col = idx % 3
        sns.histplot(data=data, x=feature, ax=axes[row, col], kde=True)
        axes[row, col].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

    # Box plots for numerical features by fit
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for idx, feature in enumerate(numerical_features):
        row = idx // 3
        col = idx % 3
        sns.boxplot(data=data, x='fit', y=feature, ax=axes[row, col])
        axes[row, col].set_title(f'{feature} vs Fit')
    plt.tight_layout()
    plt.show()

# Call the function
create_eda_analysis(data)

# Final Feature Correlation Matrix

# Define a common random state for reproducibility
RANDOM_STATE = 42

# Define features based on correlation strengths
features = [
    'size',           # strongest correlation (-0.12)
    'category',   # second strongest (+0.053)
    'rented_for', # third strongest (+0.031)
    'bust_size',      # (-0.022)
    'rating',         # (+0.02)
    'weight',         # (-0.02)
    'height',         # (-0.015)
    'age',            # (-0.015)
    'body_type'   # include for completeness
]

# Make sure all features are available
for feature in features:
    if feature not in data.columns:
        print(f"Warning: Feature '{feature}' not found in dataset")

# Convert 'fit' column from string to numeric
if data['fit'].dtype == 'object':
    # Check unique values
    print("Unique values in 'fit' column:", data['fit'].unique())

    # Convert string values to numeric
    fit_mapping = {'small': 0, 'fit': 1, 'large': 2}
    data['fit'] = data['fit'].map(fit_mapping)

    # Verify the conversion
    print("After conversion, unique values in 'fit' column:", data['fit'].unique())

# Use only available features
available_features = [f for f in features if f in data.columns]

# Now attempt to create the correlation matrix
correlation_df = data[available_features + ['fit']]

# Check datatypes before correlation
print("DataFrame dtypes:")
print(correlation_df.dtypes)

# Convert any remaining object columns to numeric
for col in correlation_df.columns:
    if correlation_df[col].dtype == 'object':
        print(f"Converting column {col} from object to numeric")
        correlation_df[col] = pd.to_numeric(correlation_df[col], errors='coerce')

# Now create the correlation matrix
corr_matrix = correlation_df.corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f")
plt.title('Feature Correlation Matrix')
plt.show()

"""## 1.5 Dataset and Model Preparation"""

# Creating input feature matrix X and output label matrix y
X = data[available_features].values
y = data['fit'].values

# Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a consistent train/test split for all models
indices = np.arange(len(data))
indices_train, indices_test, y_train, y_test = train_test_split(
    indices,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y  # Ensures proportional representation of classes
)

X_train = X_scaled[indices_train]
X_test = X_scaled[indices_test]

# Store the split data for SVD model
train_data = data.iloc[indices_train]
test_data = data.iloc[indices_test]

# Create a common function for model evaluation with model storage
def evaluate_model(model, X_test, y_test, model_name):
    # Predict and evaluate
    y_pred = model.predict(X_test)

    # Calculate basic metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    # Print results
    print(f"=== {model_name} Results ===")
    print(f"Accuracy Score: {results['accuracy']:.4f}")
    print(f"Mean Squared Error: {results['mse']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Small', 'Fit', 'Large'],
                yticklabels=['Small', 'Fit', 'Large'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

    # Calculate ROC-AUC for multi-class if model supports predict_proba
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)
            n_classes = len(np.unique(y_test))
            y_test_bin = label_binarize(y_test, classes=range(n_classes))

            auc_scores = []
            for i in range(n_classes):
                auc_scores.append(roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i]))

            results['auc_scores'] = auc_scores
            results['avg_auc'] = np.mean(auc_scores)

            print(f"AUC Scores by class: {[round(score, 4) for score in auc_scores]}")
            print(f"Average AUC: {results['avg_auc']:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate AUC scores. Error: {e}")

    # Store results for comparison
    return {'model': model, 'results': results}

"""#**2. Recommender system**

## 2.1 Latent Factor Model
"""

# # HYPERPARAMETER TUNING

# # Define the parameters for tuning
# param_grid = {
#     'n_factors': [2, 5, 10, 20],  # Number of latent factors
#     'n_epochs': [10, 20, 30, 40, 50],       # Number of iterations
#     'lr_all': [0.002, 0.005, 0.01, 0.1], # Learning rate
#     'reg_all': [0.01 ,0.02, 0.1, 0.2],    # Regularization
# }

# # Convert train_data and test_data to Surprise Dataset format
# reader = Reader(rating_scale=(0, 2))  # Adjusting range of reader to cover that of 'fit' column (0 to 2)
# trainset = Dataset.load_from_df(train_data[['user_id', 'item_id', 'fit']], reader)
# testset = [(uid, iid, r) for uid, iid, r in zip(test_data['user_id'], test_data['item_id'], test_data['fit'])]

# # Use GridSearchCV to find the best hyperparameters using MSE as the evaluation metric
# grid_search = GridSearchCV(SVD, param_grid, measures=['mse'], cv=3)
# grid_search.fit(trainset)
# print(f"Best Hyperparameters: {grid_search.best_params}")
# print(f"Best MSE: {grid_search.best_score['mse']:.4f}")

# # Train the model over obtained parameters to get the best model
# best_model = grid_search.best_estimator['mse']
# trainset_new = trainset.build_full_trainset()
# best_model.fit(trainset_new)

# # Test on the test set
# predictions = best_model.test(testset)
# y_true = [int(pred.r_ui) for pred in predictions]
# y_pred = [round(pred.est) for pred in predictions]

# # Calculate evaluation metrics
# svd_results = {
#     'accuracy': accuracy_score(y_true, y_pred),
#     'mse': mean_squared_error(y_true, y_pred),
#     'f1': f1_score(y_true, y_pred, average='weighted')
# }

# print("=== Best SVD Model Results ===")
# print(f"Accuracy Score: {svd_results['accuracy']:.4f}")
# print(f"Mean Squared Error: {svd_results['mse']:.4f}")
# print(f"F1 Score: {svd_results['f1']:.4f}")

# # Create confusion matrix
# from sklearn.metrics import confusion_matrix
# conf_matrix = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
#             xticklabels=['Small', 'Fit', 'Large'],
#             yticklabels=['Small', 'Fit', 'Large'])
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix for SVD Model")
# plt.show()

# Model based on parameters obtained after hyperparameter tuning

# Convert train_data and test_data to Surprise Dataset format
train_df = train_data[['user_id', 'item_id', 'fit']].copy()
test_df = test_data[['user_id', 'item_id', 'fit']].copy()
reader = Reader(rating_scale=(0, 2))
trainset = Dataset.load_from_df(train_df, reader).build_full_trainset()
testset = [(uid, iid, r) for uid, iid, r in zip(test_df['user_id'], test_df['item_id'], test_df['fit'])]

# Train the SVD model
model_svd = SVD(random_state=RANDOM_STATE,n_factors= 2, n_epochs= 30, lr_all= 0.005, reg_all= 0.02)
model_svd.fit(trainset)

# Test the model
predictions = model_svd.test(testset)
y_true = [int(pred.r_ui) for pred in predictions]
y_pred = [round(pred.est) for pred in predictions]

# Calculate metrics using sklearn for consistency
svd_results = {
    'accuracy': accuracy_score(y_true, y_pred),
    'mse': mean_squared_error(y_true, y_pred),
    'f1': f1_score(y_true, y_pred, average='weighted')
}

print("=== SVD Model Results ===")
print(f"Accuracy Score: {svd_results['accuracy']:.4f}")
print(f"Mean Squared Error: {svd_results['mse']:.4f}")
print(f"F1 Score: {svd_results['f1']:.4f}")

# Create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Small', 'Fit', 'Large'],
            yticklabels=['Small', 'Fit', 'Large'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for SVD Model")
plt.show()

# Alternative Implementation with improved accuracy

# For duplicate user_id item_id pairs use the most common fit as the final value
df = data.groupby(['user_id', 'item_id'], as_index=False)['fit'].agg(lambda x: x.mode()[0])

# Convert df to Surprise Dataset format
reader = Reader(rating_scale=(0, 2))
final_dataset = Dataset.load_from_df(df[['user_id', 'item_id', 'fit']], reader)


# Train the SVD model
model = SVD( n_factors=3,n_epochs= 30, lr_all= 0.005, reg_all= 0.02)
model.fit(trainset)

# Test the model
predictions = model.test(testset)
y_true = [pred.r_ui for pred in predictions]
y_pred = [pred.est for pred in predictions]
y_pred_restricted = [min(max(round(pred), 0), 2) for pred in y_pred] # Round value to closest integer


# Calculate metrics using sklearn for consistency
svd_results = {
    'accuracy': accuracy_score(y_true, y_pred_restricted),
    'mse': mean_squared_error(y_true, y_pred_restricted),
    'f1': f1_score(y_true, y_pred_restricted, average='weighted')
}

print("=== SVD Model Results ===")
print(f"Accuracy Score: {svd_results['accuracy']:.4f}")
print(f"Mean Squared Error: {svd_results['mse']:.4f}")
print(f"F1 Score: {svd_results['f1']:.4f}")

# Create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_restricted)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Small', 'Fit', 'Large'],
            yticklabels=['Small', 'Fit', 'Large'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for SVD Model")
plt.show()

"""## 2.2 K-Nearest Neighbours"""

# # HYPERPARAMETER TUNING

# param_grid = {
#     'n_neighbors': [20,30,40],  # Number of neighbors
#     'weights':['uniform'],
#     'algorithm':['auto','ball_tree'],
#     # 'weights': ['distance','uniform'],    # Weight function
#     # 'algorithm': ['auto', 'ball_tree','kd_tree','brute'],  # Algorithm to compute the nearest neighbors
#     'leaf_size': [20, 30, 40],  # Leaf size
#     'p': [1,2],  # Power parameter for the Minkowski distance metric (1=Manhattan, 2=Euclidean)
# }

# # Initialize the KNeighborsClassifier
# knn_model = KNeighborsClassifier(n_jobs=4)

# # Use GridSearchCV to find the best hyperparameters
# grid_search = GridSearchCV_from_sklearn(knn_model, param_grid, cv=2, n_jobs=4, verbose=1, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

# # Use the best estimator from the grid search
# best_knn_model = grid_search.best_estimator_

# # Make predictions with the best model
# y_pred = best_knn_model.predict(X_test)

# # Evaluate the model's accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Test Accuracy of the tuned KNN model: {:.2f}".format(accuracy))

# # Calculate metrics
# knn_results = {
#     'accuracy': accuracy_score(y_true, y_pred),
#     'mse': mean_squared_error(y_true, y_pred),
#     'f1': f1_score(y_true, y_pred, average='weighted')
# }

# print("=== KNN Model Results ===")
# print(f"Accuracy Score: {knn_results['accuracy']:.4f}")
# print(f"Mean Squared Error: {knn_results['mse']:.4f}")
# print(f"F1 Score: {knn_results['f1']:.4f}")

# # Create confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
#             xticklabels=['Small', 'Fit', 'Large'],
#             yticklabels=['Small', 'Fit', 'Large'])
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix for SVD Model")
# plt.show()

# Train the KNN model using parameters obtained from Hyperparameter tuning
knn_model = KNeighborsClassifier(n_neighbors=40, n_jobs=4,weights='uniform',p=1,algorithm='ball_tree',leaf_size=40)
knn_model.fit(X_train, y_train)

# Evaluate using our common function
evaluate_model(knn_model, X_test, y_test, "KNN Model")

"""## 2.3 RandomForest"""

# # HYPERPARAMETER TUNING

# # Define the hyperparameter grid
# param_grid = {
#     'n_estimators': [30,50,100],  # Number of trees
#     'max_depth': [10,20,30],  # Maximum depth of the tree
#     'min_samples_split': [2,5],  # Minimum samples required to split an internal node
#     'max_features': ['sqrt'],  # Number of features to consider for best split
#     'n_jobs': [4],
#     'random_state': [RANDOM_STATE]  # Random state for reproducibility
# }

# # Initialize the RandomForestClassifier
# rf_model = RandomForestClassifier()

# # Initialize GridSearchCV
# grid_search = GridSearchCV_from_sklearn(
#     estimator=rf_model,
#     param_grid=param_grid,
#     cv=3,  # Cross-validation (you can change this to 5 or 10 if necessary)
#     n_jobs=4,  # Use all available CPUs
#     verbose=1,  # Show detailed output
#     scoring='accuracy'  # Use accuracy as the metric to optimize
# )

# # Use GridSearchCV to find the best hyperparameters
# grid_search.fit(X_train, y_train)
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

# # Use the best model found by GridSearchCV
# best_rf_model = grid_search.best_estimator_

# # Make predictions with the best model
# y_pred = best_rf_model.predict(X_test)

# # Calculate metrics
# random_results = {
#     'accuracy': accuracy_score(y_test, y_pred),
#     'mse': mean_squared_error(y_test, y_pred),
#     'f1': f1_score(y_test, y_pred, average='weighted')
# }

# print("=== KNN Model Results ===")
# print(f"Accuracy Score: {random_results['accuracy']:.4f}")
# print(f"Mean Squared Error: {random_results['mse']:.4f}")
# print(f"F1 Score: {random_results['f1']:.4f}")

# # Create confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
#             xticklabels=['Small', 'Fit', 'Large'],
#             yticklabels=['Small', 'Fit', 'Large'])
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix for SVD Model")
# plt.show()

# Train the Random Forest Classifier using parameters obtained from Hyperparameter Tuning
rf_model = RandomForestClassifier(
    n_estimators=30,
    random_state=RANDOM_STATE,
    n_jobs=4,
    max_depth=10,
    max_features='sqrt',
    min_samples_split=5
)
rf_model.fit(X_train, y_train)

# Evaluate the model
model = evaluate_model(rf_model, X_test, y_test, "Random Forest Model")

# Display feature importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Random Forest Model')
plt.tight_layout()
plt.show()

"""## 2.4 XGBoost"""

# Calculate class weights for handling imbalance
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = {i: total_samples / count for i, count in enumerate(class_counts)}
print("Class Weights:", class_weights)

# Train XGBoost model
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    random_state=RANDOM_STATE,
    n_estimators=300,
    max_depth=3,
    learning_rate=0.075,
    reg_lambda=2,
    reg_alpha=0.01,
    min_child_weight=3,
    gamma=0.5,
    colsample_bytree=0.8
)
xgb_model.fit(X_train, y_train)

# Evaluate model
model = evaluate_model(xgb_model, X_test, y_test, "XGBoost Model")

# Display feature importance with actual feature names
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in XGBoost Model')
plt.tight_layout()
plt.show()

"""##2.5 Results Comparison"""

# Compare all models - using direct access to metrics
models = {
    'SVD': {
        'accuracy': svd_results['accuracy'],
        'mse': svd_results['mse'],
        'f1': svd_results['f1']
    },
    'KNN': {
        'accuracy': accuracy_score(y_test, knn_model.predict(X_test)),
        'mse': mean_squared_error(y_test, knn_model.predict(X_test)),
        'f1': f1_score(y_test, knn_model.predict(X_test), average='weighted')
    },
    'Random Forest': {
        'accuracy': accuracy_score(y_test, rf_model.predict(X_test)),
        'mse': mean_squared_error(y_test, rf_model.predict(X_test)),
        'f1': f1_score(y_test, rf_model.predict(X_test), average='weighted')
    },
    'XGBoost': {
        'accuracy': accuracy_score(y_test, xgb_model.predict(X_test)),
        'mse': mean_squared_error(y_test, xgb_model.predict(X_test)),
        'f1': f1_score(y_test, xgb_model.predict(X_test), average='weighted')
    },
    'MLP': {
        'accuracy': 0.7850,
        'mse': 0.2476,
        'f1': 0.7585
    }
}

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Accuracy': [models[m]['accuracy'] for m in models],
    'MSE': [models[m]['mse'] for m in models],
    'F1 Score': [models[m]['f1'] for m in models]
})

print("\n=== Model Comparison ===")
print(comparison_df.sort_values('Accuracy', ascending=False))

# Visualize comparison
plt.figure(figsize=(12, 6))
comparison_melted = pd.melt(comparison_df, id_vars=['Model'],
                            value_vars=['Accuracy', 'F1 Score'],
                            var_name='Metric', value_name='Score')
sns.barplot(x='Model', y='Score', hue='Metric', data=comparison_melted)
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.show()

"""##2.6 Cosine Similarity - Failed Algorithm"""

df = data[['user_id', 'item_id', 'fit']]

# Average the ratings if there are duplicate user-item pairs
df = df.groupby(['user_id', 'item_id'], as_index=False)['fit'].agg(lambda x: x.mode()[0])

# fit_mapping = {"fit": 1, "small": 0, "large": -1}
# df['fit'] = data['fit'].map({'small': 0, 'fit': 1, 'large': 2})

print(df['fit'].value_counts())

most_frequent_fit = df['fit'].mode()[0]
print(most_frequent_fit)

# Load the data into the Surprise Dataset format
reader = Reader(rating_scale=(0, 2))
data_surprise = Dataset.load_from_df(df[['user_id', 'item_id', 'fit']], reader)

user_item_matrix = df.pivot(index='user_id', columns='item_id', values='fit').fillna(most_frequent_fit) #This will cause the data to be even more biased towards 1
user_item_matrix=user_item_matrix.astype(int)

# Initialize the SVD model
svd = SVD()

# Train the model
svd.fit(trainset)

# Get the item latent factors from the trained SVD model
item_latent_factors = svd.qi  # Item latent factors matrix (n_items, n_latent)

# Compute the cosine similarity matrix between items based on their latent factors
similarity_matrix = cosine_similarity(item_latent_factors)

# You can store this similarity matrix for future predictions
np.save('similarity_matrix.npy', similarity_matrix)

# # Function to recommend popular items (used for new users)

item_popularity = user_item_matrix.mean(axis=0)

# Sort items by their popularity (descending order)
popular_items = item_popularity.sort_values(ascending=False)
top_items = popular_items.head(100)

fits = []
for item in top_items.index:
    # Get the ratings for the popular item
    item_ratings = user_item_matrix[item]
    item_fit = item_ratings.median()
    # print('contains one',(item_ratings == 1).sum())
    fits.append((item, item_fit))
most_frequent_fit_for_new_user=int(np.median([fit[1] for fit in fits]))
print(most_frequent_fit_for_new_user)
    # return most_frequent_fit  # Return the most frequent fit or the fit for the popular item

item_latent_factors

# worked
def predict_fit(user_id, item_id, user_item_matrix, similarity_matrix, top_n=10):
    """Predict the fit based on majority voting (k-NN) using precomputed similarity matrix"""
    # Item not in training set -  new item
    if item_id not in user_item_matrix.columns:
        return most_frequent_fit

    # New user
    if user_id not in user_item_matrix.index or user_item_matrix.loc[user_id].isnull().all():
        return most_frequent_fit_for_new_user

    # Get the user vector (ratings for items)
    user_vector = user_item_matrix.loc[user_id].values

    # List to store similarity scores
    similarity_scores = []

    # Iterate over other items to calculate similarity using the precomputed similarity matrix
    for other_item_id in user_item_matrix.columns:
        if other_item_id == item_id or np.isnan(user_vector[user_item_matrix.columns.get_loc(other_item_id)]):
            continue

        # Get the similarity score between the current item and the target item
        item_index = user_item_matrix.columns.get_loc(item_id)
        other_item_index = user_item_matrix.columns.get_loc(other_item_id)

        # Ensure the indices are valid for the similarity matrix
        if item_index < similarity_matrix.shape[0] and other_item_index < similarity_matrix.shape[1]:
            similarity = similarity_matrix[item_index, other_item_index]
            similarity_scores.append((other_item_id, similarity))

    # Sort the items by similarity score
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Get top N most similar items
    top_items = similarity_scores[:top_n]

    # Get the fits of these top N similar items for the user
    fits = []
    for item, _ in top_items:
        if item in user_item_matrix.columns:
            # fits.append(user_item_matrix.loc[user_id, item])
            fits.append(int(user_item_matrix.loc[user_id, item]))


    # Return the predicted fit (majority vote)
    if fits:
        return int(np.median(fits))
    else:
        return 0  # If no similar users have rated the item, return 0 (no fit)

true_ratings = []
predicted_ratings = []

# Loop through testset (no parentheses needed)
for user_id, item_id, true_fit in testset:
    true_ratings.append(true_fit)
    predicted_ratings.append(predict_fit(user_id, item_id, user_item_matrix, similarity_matrix))

# Calculate MSE (Mean Squared Error)
mse = np.square(np.subtract(true_ratings, predicted_ratings)).mean()
print(f"Mean Squared Error (MSE): {mse}")

# true_ratings

# predicted_ratings

# print(np.unique(predicted_ratings))