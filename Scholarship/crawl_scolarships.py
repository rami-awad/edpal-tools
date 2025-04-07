import asyncio
import csv
import logging
from datetime import datetime
from crawl4ai import *
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

# Configuration for each website
websites = [
    {
        'name': 'for9a',
        'url': "https://www.for9a.com/en/opportunity/category/Scholarships",
        'load_more_button_xpath': "//button[text()='Load More']",
        'section_class': 'container',
        'item_class': 'p-5 space-y-4',
        'link_class': 'block text-lg font-bold tracking-tight text-gray-800 hover:text-orange-500 transition-colors duration-300 line-clamp-2 editor_page',
        'title_class': 'text-2xl mt-3 md:text-3xl font-bold mb-6 leading-relaxed text-gray-900 max-w-[750px]',
        'location_label': 'Job location',
        'degree_label': 'Degree',
        'requirements_label': 'Needed documents',
        'fields_label': 'Speciality',
        'details_class': 'content__html section-editor'
    },
    # Add more website configurations here
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def extract_links(config):
    """
    Extracts links from the given website configuration.

    Args:
        config (dict): A dictionary containing the website configuration.
            It should have the following keys:
                name (str): The name of the website.
                url (str): The URL of the website.
                load_more_button_xpath (str): The XPath for the "Load more" button.
                section_class (str): The class name of the section containing the links.
                item_class (str): The class name of the items containing the links.
                link_class (str): The class name of the links.

    Returns:
        list: A list of tuples containing the text and href of the extracted links.
    """
    logging.info(f"Starting extraction for {config['name']}")
    start_time = time.time()
    
    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'eager'
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-infobars')
    options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('blink-settings=imagesEnabled=false')
    options.add_argument("--disable-ads")
    
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(120)
    wait = WebDriverWait(driver, 20)

    driver.get(config['url'])

    # Click the "Load more" button until it's not clickable
    while True:
        try:
            load_more_button = wait.until(EC.element_to_be_clickable((By.XPATH, config['load_more_button_xpath'])))
            driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
            actions = ActionChains(driver)
            actions.move_to_element(load_more_button).click().perform()
            time.sleep(2)
        except Exception:
            break

    # Extract the links from the page
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    
    section = soup.find('section', class_=config['section_class'])
    links_to_visit = []
    if section:
        items = section.find_all('div', class_=config['item_class'])
        for item in items:
            links = item.find_all('a', class_=config['link_class'])
            for link in links:
                href = link.get('href')
                text = link.get_text(strip=True)
                links_to_visit.append((text, href))

    driver.quit()

    elapsed_time = time.time() - start_time
    logging.info(f"Extracted {len(links_to_visit)} links in {elapsed_time:.2f} seconds for {config['name']}")

    return links_to_visit

async def process_link(driver, config, text, href, index, total_links):
    """
    Process a single link by extracting the following information:
        - Title
        - Location
        - Degree
        - Requirements
        - Fields
        - Details

    Args:
        driver (WebDriver): The Selenium WebDriver instance.
        config (dict): The configuration dictionary containing the website's settings.
        text (str): The text of the link.
        href (str): The href of the link.
        index (int): The index of the link in the list of links to visit.
        total_links (int): The total number of links to visit.

    Returns:
        list: A list containing the extracted information.
    """
    retries = 3
    for attempt in range(retries):
        try:
            # Load the page
            driver.get(href)
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # Extract the title
            title = soup.find('h1', class_=config['title_class']).get_text(strip=True) if soup.find('h1', class_=config['title_class']) else ''

            # Extract the location
            location_span = soup.find('span', string=config['location_label'])
            location = location_span.find_next('span').get_text(strip=True) if location_span else ''

            # Extract the degree
            degree_span = soup.find('span', string=config['degree_label'])
            degree = degree_span.find_next('span').get_text(strip=True) if degree_span else ''

            # Extract the requirements
            requirements_span = soup.find('span', string=config['requirements_label'])
            requirements = requirements_span.find_next('span').get_text(strip=True) if requirements_span else ''

            # Extract the fields
            fields_span = soup.find('span', string=config['fields_label'])
            fields = fields_span.find_next('span').get_text(strip=True) if fields_span else ''

            # Extract the details
            details = soup.find('div', class_=config['details_class']).get_text(separator='\n', strip=True) if soup.find('div', class_=config['details_class']) else ''

            # Log the result
            logging.info(f"Processed link {index + 1}/{total_links} for {config['name']}: {href}")

            # Return the extracted information
            return [title, href, location, degree, requirements, fields, details]
        except Exception as e:
            # Log the error
            logging.error(f"Failed to process {href} on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                # Return empty values if the link failed to process
                return [text, href, '', '', '', '', '']

async def process_website(config):
    """
    Process a website by extracting links, then processing each link by extracting the title, location, degree, requirements, fields, and details.

    Args:
        config (dict): A dictionary containing the website configuration.
            It should have the following keys:
                name (str): The name of the website.
                url (str): The URL of the website.
                load_more_button_xpath (str): The XPath for the "Load more" button.
                section_class (str): The class name of the section containing the links.
                item_class (str): The class name of the items containing the links.
                link_class (str): The class name of the links.
                title_class (str): The class name of the title element.
                location_label (str): The label of the location element.
                degree_label (str): The label of the degree element.
                requirements_label (str): The label of the requirements element.
                fields_label (str): The label of the fields element.
                details_class (str): The class name of the details element.
    """
    links_to_visit = await extract_links(config)

    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'eager'
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-infobars')
    options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('blink-settings=imagesEnabled=false')
    options.add_argument("--disable-ads")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(120)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"scholarships_output_{config['name']}_{timestamp}.csv"

    # Open the output file and write the header row
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Title', 'Link', 'Location', 'Degree', 'Requirements', 'Fields', 'Details'])

        # Process each link
        total_links = len(links_to_visit)
        tasks = [process_link(driver, config, text, href, index, total_links) for index, (text, href) in enumerate(links_to_visit)]
        results = await asyncio.gather(*tasks)

        # Write each result to the output file
        for result in results:
            writer.writerow(result)

    driver.quit()

    # Log that the website has been processed
    logging.info(f"Finished processing {config['name']}. Output saved to {output_file}")

async def main():
    """
    Main function to run the script.

    This function runs the process_website function for each website in the
    websites list and logs the total processing time.
    """
    start_time = time.time()
    tasks = [process_website(config) for config in websites]
    # Wait for all the tasks to complete
    await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    # Log the total processing time
    logging.info(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())