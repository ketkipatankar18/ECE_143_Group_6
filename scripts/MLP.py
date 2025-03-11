import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, classification_report, confusion_matrix

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

# Load data
data = pd.read_json('Data/renttherunway_final_data.json', lines=True)

# Rename columns for clarity
data.rename(columns={'bust size': 'bust_size', 'rented for': 'rented_for', 'body type':'body_type'}, inplace=True)

# Data Cleaning (using preprocessing from original code)
# Check missing values
data.isna().sum()

# Fill missing values with most common values
size_null = ['bust_size', 'weight', 'rented_for', 'body_type', 'height']
for i in size_null:
  i_sum = data.groupby(i).size().sort_values(ascending=False)

# Fill missing values
data['bust_size'].fillna('34b', inplace=True)
data['weight'].fillna('130lbs', inplace=True)
data['rating'].fillna(float(data['rating'].median()), inplace=True)
data['rented_for'].fillna('wedding', inplace=True)
data['body_type'].fillna('hourglass', inplace=True)
data['height'].fillna('5\' 4"', inplace=True)
data['age'].fillna(float(data['age'].median()), inplace=True)

data['bust_size'] = data['bust_size'].apply(bust_size_to_numeric)
data['rented_for'] = data['rented_for'].apply(rented_for_to_numeric)
data['body_type'] = data['body_type'].apply(body_type_to_numeric)
data['category'] = data['category'].apply(category_to_numeric)
data['height'] = data['height'].apply(height_to_cm)

# Additional date and string specific conversions
data['review_date'] = pd.to_datetime(data['review_date'])
data['weight'] = data.weight.str.replace('lbs', '')
data['weight'] = data['weight'].astype('float64')

# Drop columns that are not used
data = data.drop(['review_summary','review_date'], axis=1)

# Filter the outliers
data = data[data.age <= 100]
data = data[data['size'] <= 22]

# Reseting row labels after filtering out outliers
data.reset_index(drop=True,inplace= True)

features = [
    'size',
    'category',
    'rented_for',
    'bust_size',
    'rating',
    'weight',
    'height',
    'age',
    'body_type'
]


# Encode fit as target variable (0=small, 1=fit, 2=large)
data['fit_encoded'] = data['fit'].map({'small': 0, 'fit': 1, 'large': 2})
data['fit_encoded'] = data['fit_encoded'].astype(int)

# Feature Engineering with TF-IDF
# Create text features for TF-IDF from categorical columns
data['text_features'] = data['review_text'].astype(str)

# Apply TF-IDF to the text features
tfidf = TfidfVectorizer(max_features=100)
tfidf_features = tfidf.fit_transform(data['text_features'])
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
)

# Combine numerical features with TF-IDF features
X_numerical = data[features]

# Combine features
X_combined = pd.concat([X_numerical.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
y = data['fit_encoded']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(150, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=42
)

# Fit the model
mlp.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = mlp.predict(X_test_scaled)

mlp_results = {
    'accuracy': accuracy_score(y_test, y_pred),
    'mse': mean_squared_error(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='weighted')
}

print("=== MLP with TF-IDF Model Results ===")
print(f"Accuracy Score: {mlp_results['accuracy']:.4f}")
print(f"Mean Squared Error: {mlp_results['mse']:.4f}")
print(f"F1 Score: {mlp_results['f1']:.4f}")

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Small', 'Fit', 'Large'],
            yticklabels=['Small', 'Fit', 'Large'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for MLP Model")
plt.show()

models = {
    'SVD': {
        'accuracy': 0.756253,
        'mse': 0.245080,
        'f1': 0.691648
    },
    'KNN': {
        'accuracy': 0.742532,
        'mse': 0.269389,
        'f1': 0.648772
    },
    'Random Forest': {
        'accuracy': 0.742929,
        'mse': 0.263242,
        'f1': 0.643942
    },
    'XGBoost': {
        'accuracy': 0.743233,
        'mse': 0.271773,
        'f1': 0.654646
    },
    'MLP with TF-IDF': mlp_results
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