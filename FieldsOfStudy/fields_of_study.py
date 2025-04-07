import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure 'punkt', 'punkt_tab', and 'stopwords' are downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Connect to PostgreSQL
def connect_to_db():
    """
    Connects to the PostgreSQL database using environment variables.

    :return: A PostgreSQL connection object
    """
    # Connect to the PostgreSQL database using credentials from environment variables
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),  # Database name
        user=os.getenv('DB_USER'),    # Database user
        password=os.getenv('DB_PASSWORD'),  # User password
        host=os.getenv('DB_HOST'),    # Database host
        port=os.getenv('DB_PORT')     # Connection port
    )
    return conn

# Fetch university programs from PostgreSQL
def fetch_programs(conn):
    """
    Fetches university programs from the PostgreSQL database.

    :param conn: A PostgreSQL connection object
    :return: A Pandas DataFrame containing the university programs
    """
    # Fetch university programs from PostgreSQL
    query = "SELECT name_en FROM university_programs;"

    # Read the query result into a Pandas DataFrame
    df = pd.read_sql(query, conn)

    return df

# Fetch predefined fields of study
def fetch_fields_of_study():
    """
    Fetches predefined fields of study.

    This function fetches predefined fields of study that are used in the machine
    learning model to classify university programs.

    Returns:
        list: A list of strings, where each string is a field of study.
    """
    fields_of_study = [
        "Computer Science",  # Computer science, IT, and related fields
        "Engineering",  # Engineering, technology, and related fields
        "Mathematics",  # Mathematics, statistics, and related fields
        "Physics",  # Physics, astronomy, and related fields
        "Biology",  # Biology, biochemistry, and related fields
        # Add more fields of study here
    ]
    return fields_of_study

# Preprocess text data
def preprocess_text(text):
    """
    Preprocess text data for the machine learning model.

    This function preprocesses text data by tokenizing it, converting it to
    lowercase, removing stopwords, and stemming the tokens.

    Args:
        text (str): The text data to preprocess

    Returns:
        str: The preprocessed text data
    """
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords]
    
    # Stem the tokens
    stemmer = nltk.stem.SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join the tokens back into a string
    return ' '.join(tokens)

# Train the machine learning model
def train_model(X, y):
    """
    Train a machine learning model to classify university programs into fields
    of study.

    Args:
        X (list or numpy array): A list of strings or a numpy array of strings
            containing the names of the university programs to classify.
        y (list or numpy array): A list of strings or a numpy array of strings
            containing the fields of study that the programs belong to.

    Returns:
        tuple: A tuple containing the trained model and a function to preprocess
            the text data for the model.
    """
    # Create a TfidfVectorizer to transform the text data into a numerical
    # representation
    vectorizer = TfidfVectorizer()

    # Create a Multinomial Naive Bayes classifier to classify the programs
    classifier = MultinomialNB()

    # Create a pipeline to chain the vectorizer and classifier together
    model = make_pipeline(vectorizer, classifier)

    def preprocess_text_pipeline(X):
        """
        Preprocess the text data for the machine learning model.

        Args:
            X (list or numpy array): A list of strings or a numpy array of
                strings containing the text data to preprocess.

        Returns:
            numpy array: A numpy array containing the preprocessed text data.
        """
        if isinstance(X, np.ndarray):
            X = X.flatten()  # Ensure it's a 1D array
        elif isinstance(X, pd.Series):
            X = X.tolist()  # Convert to list
            
        return [preprocess_text(text) for text in X]

    # Transform the text data into a numerical representation using the vectorizer
    X_transformed = preprocess_text_pipeline(X.ravel())

    # Train the model using the transformed data
    model.fit(X_transformed, y_train.ravel())

    # Return the trained model and the preprocessing function
    return model, preprocess_text_pipeline

def classify_programs(model, preprocess_text_pipeline, mlb, programs_df):
    """
    Classify programs into fields of study using the trained model.

    Args:
        model (sklearn.Pipeline): The trained model to use for classification.
        preprocess_text_pipeline (function): A function to preprocess the text
            data for the model.
        mlb (sklearn.preprocessing.MultiLabelBinarizer): The object to use for
            binarizing the labels.
        programs_df (pandas.DataFrame): The DataFrame containing the program names
            to classify.

    Returns:
        pandas.DataFrame: The DataFrame with the classified fields of study added
            as a new column.
    """
    # Preprocess the program names using the provided function
    programs = programs_df['program_name']
    X_programs = preprocess_text_pipeline(programs)

    # Predict the fields of study using the trained model
    y_pred = model.predict(X_programs)

    # Binarize the predicted labels using the MultiLabelBinarizer
    y_pred_binarized = mlb.transform(y_pred)

    # Convert the binarized labels to a list of labels for each program
    y_pred_labels = [list(mlb.classes_[np.where(row == 1)]) for row in y_pred_binarized]

    # Add the predicted fields of study to the DataFrame
    programs_df['fields_of_study'] = y_pred_labels

    # Return the DataFrame with the predicted fields of study
    return programs_df

# Load your labeled dataset for training
def load_labeled_dataset():
    """
    Load labeled dataset for training.

    This function loads a CSV file containing labeled data for training the
    model. The CSV file should have two columns: 'program_name' and 'fields_of_study'.

    Returns:
        pandas.DataFrame: The loaded labeled dataset.
    """
    # Replace this with your labeled dataset
    labeled_data = pd.read_csv(r".\FieldsOfStudy\fields_of_study.csv")
    return labeled_data


if __name__ == "__main__":
    # Connect to PostgreSQL
    conn = connect_to_db()

    # Fetch university programs
    programs_df = fetch_programs(conn)

    # Load labeled dataset
    labeled_data = load_labeled_dataset()

    # Preprocess labeled dataset
    labeled_data['program_name'] = labeled_data['program_name'].apply(preprocess_text)

    # Fetch fields of study
    fields_of_study = fetch_fields_of_study()

    # Convert labels to binary representation
    mlb = MultiLabelBinarizer(classes=fields_of_study)
    y = mlb.fit_transform(labeled_data['fields_of_study'])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_data['program_name'], y, test_size=0.2, random_state=42)
    
    print(X_train.shape)  # Check if it's (num_samples,)
    print(y_train.shape)  # Should be (num_samples, num_classes)

    # Train the model
    model = train_model(X_train.values.reshape(-1, 1), y_train)

    # Evaluate the model
    y_pred = model.predict(X_test.values.reshape(-1, 1))
    y_pred_binarized = mlb.transform(y_pred)
    y_test_binarized = mlb.transform(labeled_data['fields_of_study'].iloc[X_test.index])
    print(classification_report(y_test_binarized.argmax(axis=1), y_pred_binarized.argmax(axis=1)))

    # Classify university programs
    classified_programs_df = classify_programs(model, mlb, programs_df)

    # Print classified programs
    print("\nClassified University Programs:")
    for _, row in classified_programs_df.iterrows():
        print(f"{row['program_name']}: {', '.join(row['fields_of_study'])}")

    # Close the PostgreSQL connection
    conn.close()