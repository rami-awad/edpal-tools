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
    Connect to the PostgreSQL database and return the connection and cursor.

    This function connects to the PostgreSQL database using the environment
    variables `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, and `DB_PORT`.
    It returns a tuple containing the database connection and cursor objects.

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
    cursor = conn.cursor()
    return conn, cursor

def fetch_data(cursor: psycopg2.extensions.cursor) -> list[tuple]:
    """
    Execute the query to fetch data from the database.

    This function executes a SQL query to fetch data from the `universities` table
    in the database. The query fetches the distinct values of the `name_en` column
    and sorts them in ascending order. The function returns the fetched data as a
    list of rows.

    Returns:
        list: A list of rows, where each row is a tuple containing the distinct
            values of the `name_en` column.
    """
    # SQL query to fetch the data
    query = """
            select
                u.name_en as university_name
            from
                public.universities u
            order by
	            u.name_en asc
            limit 100 offset 0;
            """
    # Execute the query
    cursor.execute(query)
    # Fetch the results
    rows = cursor.fetchall()
    # Print the number of records retrieved
    print(f"Number of records retrieved: {len(rows)}")
    # Return the fetched data
    return rows

def fetch_university_contact(client, university_name, model_name):
    """
    Fetch university contact information using the OpenAI local server.

    This function fetches the contact details for the person in charge of the
    Office of Development or Corporate Relations at the specified university.
    The contact details should provide: first name, last name, email, phone
    number, city, country, job title, university name, website_url. Additionally,
    it provides the email address for the Office of Development or Corporate
    Relations at the university, and adds it under the "additional_email_addresses"
    field. If it finds more than one email address, it separates them with a
    semicolon (;). If it is not sure, it provides a contact with a job similar to
    the ones provided for a person who is in charge of discussing business topics
    with external businesses.

    The function returns the data exactly in the following JSON format, with no
    additional text or explanation:

    {
        "university_name": <string>,
        "first_name": <string>,
        "last_name": <string>,
        "email": <string>,
        "phone_number": <string>,
        "city": <string>,
        "country": <string>,
        "job_title": <string>,
        "website_url": <string>,
        "additional_email_addresses": <string>
    }

    If the contact information is not available, or there's insufficient data,
    it replaces the value with `null`. It ensures that the response is strictly in
    JSON format as specified, without any additional words or explanations.
    """
    prompt = f"""
    provide the contact details for the person in charge of the Office of Development or Corporate Relations at '{university_name}'.
    The contact details should provide: first name, last name, email, phone number, city, country, job title, university name, website_url.
    Additionally, provide the email address for the Office of Development or Corporate Relations at '{university_name}', and add it under the "additional_email_addresses" field. If you find more than one email address, separate them with a semicolon (;).
    If you are not sure, provide a contact with a job similar to the ones provided for a person who is in charge of discussing business topics with external businesses.
    Return the data exactly in the following JSON format, with no additional text or explanation:

    {{
        "university_name": <string>,
        "first_name": <string>,
        "last_name": <string>,
        "email": <string>,
        "phone_number": <string>,
        "city": <string>,
        "country": <string>,
        "job_title": <string>,
        "website_url": <string>,
        "additional_email_addresses": <string>
    }}

    If the contact information is not available, or there's insufficient data, replace the value with `null`.
    Ensure that the response is strictly in JSON format as specified, without any additional words or explanations.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant specialized in university business relations worldwide. You will respond strictly with the JSON structure provided by the user without any additional text, explanation, or notes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    try:
        message_content = response.choices[0].message.content.strip()
        print(f"Raw response content: {message_content}")
        json_match = re.search(r'\{.*\}', message_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_str = re.sub(r'(?<=:\s)(\d+),(\d+)', r'\1\2', json_str)
            university_contact = json.loads(json_str)
            return university_contact
        else:
            print("No JSON structure found in the response.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Response content: {message_content}")
        return None
    except KeyError as e:
        print(f"Key error: {e}")
        print(f"Response content: {message_content}")
        return None
    except Exception as e:
        print(f"Error fetching university contact: {e}")
        return None

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
    # Add new columns for contact details
    contact_columns = ['first_name', 'last_name', 'email', 'phone_number', 'city', 'country', 'job_title', 'website_url', 'additional_email_addresses']
    for col in contact_columns:
        df[col] = None  # Initialize with None values

    record_count = 0
    for index, row in df.iterrows():
        print(f"Processing row: {row.to_dict()}")  # Debug: Print row details
        university_contact = fetch_university_contact(client, row['university_name'], model_name)

        # Ensure data is assigned properly
        if university_contact:
            for col in contact_columns:
                df.at[index, col] = university_contact.get(col, None)

        record_count += 1
        print(f"Records processed so far: {record_count} of {len(df)}")
        
        # Time tracking
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_per_record = elapsed_time / record_count
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Time per record: {time_per_record:.2f} seconds")
        print(f"=================================================\n")

    return df, record_count

def main():
    """
    Main entry point for the script.

    This script processes a list of university names and fetches their contact information
    using the OpenAI API. The results are saved to a CSV file.

    The script takes a couple of minutes to run, depending on the number of records and the
    performance of the OpenAI API.
    """
    # Start the timer
    start_time = time.time()

    # Set up OpenAI client to use local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Connect to the PostgreSQL database and fetch data
    conn, cursor = connect_to_db()
    rows = fetch_data(cursor)
    cursor.close()
    conn.close()

    # Define column names and convert query result to a DataFrame
    columns = ["university_name"]
    df = pd.DataFrame(rows, columns=columns)

    # Process records to fetch and update university contact
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    df, record_count = process_records(df, client, model_name, start_time)

    # Add the constant column after processing records
    df['nature_of_relationship'] = 'University'

    # Rename the columns for better readability
    df.rename(columns={
        'university_name': 'University Name',
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

    # Reorder columns for CSV output
    column_order = ['University Name', 'First Name', 'Last Name', 'Email', 'Phone Number', 
                    'City', 'Country', 'Job Title', 'Website URL', 'Nature of Relationship', 
                    'Additional email addresses']
    df = df[column_order]

    # Write the output to a CSV file
    df.to_csv('university_contacts.csv', index=False)

    # End the timer
    end_time = time.time()

    # Calculate the total time taken
    total_time = end_time - start_time

    # Calculate the estimated time per record
    if record_count > 0:
        total_time_per_record = total_time / record_count
    else:
        total_time_per_record = 0

    # Print the results
    print(f"Total records processed: {record_count}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Estimated total time per record: {total_time_per_record:.2f} seconds")

if __name__ == "__main__":
    main()