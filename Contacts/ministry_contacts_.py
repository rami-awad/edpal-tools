import psycopg2
import pandas as pd
import json
from openai import OpenAI
import re
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def connect_to_db():
    """
    Establish a connection to the PostgreSQL database using credentials
    from environment variables and return the connection and cursor objects.

    Returns:
        tuple: A tuple containing the database connection and cursor.
    """
    # Connect to the PostgreSQL database using credentials from environment variables
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),  # Database name
        user=os.getenv('DB_USER'),    # Database user
        password=os.getenv('DB_PASSWORD'),  # User password
        host=os.getenv('DB_HOST'),    # Database host
        port=os.getenv('DB_PORT')     # Connection port
    )
    cursor = conn.cursor()  # Create a cursor object to interact with the database
    return conn, cursor     # Return the connection and cursor

def fetch_data(cursor):
    """
    Execute the query to fetch data from the database.

    This function executes a SQL query to fetch data from the `universities`
    table in the database. The query fetches the distinct values of the `country`
    column and sorts them in ascending order. The function returns the fetched
    data as a list of rows.

    Returns:
        list: A list of rows, where each row is a tuple containing the
            distinct values of the `country` column.
    """
    query = """
            SELECT distinct country
            FROM public.universities
            order by country asc
            limit 100;
            """
    cursor.execute(query)
    rows = cursor.fetchall()
    print(f"Number of records retrieved: {len(rows)}")
    return rows

def fetch_university_contact(client, country, model_name):
    """Fetch ministry contact information using the OpenAI local server.

    This function fetches the contact information for the person in charge of
    the Office of Development or Corporate Relations at the Ministry of Education
    or the Ministry of Higher Education or the Ministry of Children Development in
    the specified country. The contact details should provide: first name, last
    name, email, phone number, city, country, job title, ministrty name, website_url.
    Additionally, it should provide the email address for the Office of
    Development or Corporate Relations at the Ministry of Education or the
    Ministry of Higher Education or the Ministry of Children Development in the
    specified country, and add it under the "additional_email_addresses" field.
    If it finds more than one email address, it should separate them with a
    semicolon (;). If it is not sure, it should provide a contact with a job
    similar to the ones provided for a person who is in charge of discussing
    business topics with external businesses.

    The function returns the data exactly in the following JSON format, with no
    additional text or explanation:

    [
        {
            "ministrty_name": <string>,
            "first_name": <string>,
            "last_name": <string>,
            "email": <string>,
            "phone_number": <string>,
            "city": <string>,
            "country": <string>,
            "job_title": <string>,
            "website_url": <string>,
            "additional_email_addresses": <string>
        },
        ...
    ]

    If the contact information is not available, or there's insufficient data,
    it should replace the value with `null`. It should ensure that the response
    is strictly in JSON format as specified, without any additional words or
    explanations.

    Args:
        client (OpenAI): The OpenAI client object.
        country (str): The country name.
        model_name (str): The name of the model to use.

    Returns:
        list: A list of dictionaries, where each dictionary contains the contact
            information for a person in charge of the Office of Development or
            Corporate Relations at the Ministry of Education or the Ministry of
            Higher Education or the Ministry of Children Development in the
            specified country.
    """
    prompt = f"""
    provide the contact details for the person in charge of the Office of Development or Corporate Relations at the Ministry of Education or the Ministry of Higher Education or the Ministry of Children Development in '{country}'.
    The contact details should provide: first name, last name, email, phone number, city, country, job title, ministrty name, website_url.
    Additionally, provide the email address for the Office of Development or Corporate Relations at the Ministry of Education or the Ministry of Higher Education or the Ministry of Children Development in '{country}', and add it under the "additional_email_addresses" field. If you find more than one email address, separate them with a semicolon (;).
    If you are not sure, provide a contact with a job similar to the ones provided for a person who is in charge of discussing business topics with external businesses.
    Return the data exactly in the following JSON format, with no additional text or explanation:

    [
        {{
            "ministrty_name": <string>,
            "first_name": <string>,
            "last_name": <string>,
            "email": <string>,
            "phone_number": <string>,
            "city": <string>,
            "country": <string>,
            "job_title": <string>,
            "website_url": <string>,
            "additional_email_addresses": <string>
        }},
        ...
    ]

    If the contact information is not available, or there's insufficient data, replace the value with `null`.
    Ensure that the response is strictly in JSON format as specified, without any additional words or explanations.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant specialized in ministry business relations worldwide. You will respond strictly with the JSON structure provided by the user without any additional text, explanation, or notes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    try:
        message_content = response.choices[0].message.content.strip()
        print(f"Raw response content: {message_content}")
        records = json.loads(message_content)
        if isinstance(records, list):
            return records
        else:
            print("Unexpected response format. Returning empty list.")
            return []
    except Exception as e:
        print(f"Error parsing API response: {e}")
        return []

def process_records(df: pd.DataFrame, client: OpenAI, model_name: str, start_time: float) -> tuple[pd.DataFrame, int]:
    """
    Process each record to fetch and update university contact.

    Args:
        df (pd.DataFrame): DataFrame containing the records to be processed.
        client (OpenAI): The OpenAI client object.
        model_name (str): The name of the model to use.
        start_time (float): The start time of the processing.

    Returns:
        tuple[pd.DataFrame, int]: A tuple containing the processed DataFrame and the number of records processed.
    """
    contact_columns = ['ministrty_name', 'first_name', 'last_name', 'email', 'phone_number', 
                       'city', 'country', 'job_title', 'website_url', 'additional_email_addresses']
    all_records = []  # List to store all processed records

    for index, row in df.iterrows():
        print(f"Processing country: {row['ministry_name']}")
        contacts = fetch_university_contact(client, row['ministry_name'], model_name)

        for contact in contacts:
            record = {col: contact.get(col, None) for col in contact_columns}
            all_records.append(record)

        print(f"Total records processed so far: {len(all_records)}")

        # Time tracking
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"=================================================\n")

    # Convert the aggregated records to a DataFrame
    processed_df = pd.DataFrame(all_records, columns=contact_columns)
    return processed_df, len(all_records)

def main():
    """
    Main entry point for the script.

    This script processes a list of university names and fetches their contact information
    using the OpenAI API. The results are saved to a CSV file.

    The script takes a couple of minutes to run, depending on the number of records and the
    performance of the OpenAI API.
    """
    start_time = time.time()
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Connect to the PostgreSQL database and fetch data
    conn, cursor = connect_to_db()
    rows = fetch_data(cursor)
    cursor.close()
    conn.close()

    # Create a DataFrame from the fetched data
    df = pd.DataFrame(rows, columns=["ministry_name"])

    # Process records and fetch contact information
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    #model_name = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"  # Change this to the desired model name
    processed_df, record_count = process_records(df, client, model_name, start_time)

    # Add the constant column
    processed_df['nature_of_relationship'] = 'Ministry'

    # Rename and reorder columns
    processed_df.rename(columns={
        'ministrty_name': 'Company Name',
        'first_name': 'First Name',
        'last_name': 'Last Name',
        'email': 'Email',
        'phone_number': 'Phone Number',
        'city': 'City',
        'country': 'Country',
        'job_title': 'Job Title',
        'website_url': 'Website URL',
        'nature_of_relationship': 'Nature of Relationship',
        'additional_email_addresses': 'Additional email addresses'
    }, inplace=True)

    column_order = ['Company Name', 'First Name', 'Last Name', 'Email', 'Phone Number', 
                    'City', 'Country', 'Job Title', 'Website URL', 'Nature of Relationship', 
                    'Additional email addresses']
    processed_df = processed_df[column_order]

    # Save to CSV
    processed_df.to_csv('ministry_contacts.csv', index=False)

    # Print summary
    total_time = time.time() - start_time
    print(f"Total records processed: {record_count}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per record: {total_time / record_count:.2f} seconds")

if __name__ == "__main__":
    main()