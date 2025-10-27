"""
CTTI Data Download Script

This script automatically downloads the latest Clinical Trials Transformation Initiative (CTTI) 
dataset from the official website using Selenium WebDriver. The data is essential for 
clinical trial outcome prediction research.

Requirements:
    - Chrome browser installed
    - selenium package installed
    - Sufficient disk space (>10GB recommended)

Usage:
    python download_ctti.py

Output:
    Downloads CTTI_new.zip to ./downloads/ directory
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import urllib.request
import os

if __name__ == "__main__":
    """
    Main execution function for CTTI data download.
    
    This function:
    1. Sets up Chrome WebDriver in headless mode
    2. Navigates to CTTI pipe files download page
    3. Selects the latest available dataset
    4. Downloads the ZIP file to ./downloads/ directory
    
    The download typically contains ~500MB of pipe-delimited clinical trial data
    covering all trials from ClinicalTrials.gov database.
    """
    download_path = "./downloads"
    os.makedirs(download_path, exist_ok=True)

    # Set Chrome options to configure headless mode and download path
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Configure Chrome to download files to the specified directory
    prefs = {"download.default_directory": download_path}
    chrome_options.add_experimental_option("prefs", prefs)

    # Initialize the Chrome driver with headless options
    driver = webdriver.Chrome(options=chrome_options)

    # Go to the website
    driver.get('https://aact.ctti-clinicaltrials.org/pipe_files')

    # Wait for the select element to be present
    dropdown = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "form-select"))
    )

    # # Find all option elements within the select element
    # options = dropdown.find_elements(By.TAG_NAME, "option")

    # # Select the second option (index 1, as the first is "Select file to download")
    # if len(options) > 1:
    #     first_file_option = options[1]
    #     requests.get(first_file_option.get_attribute('value'))
    # else:
    #     print("No files available in the dropdown to select.")


    # Wait until the dropdown is populated (wait up to 20 seconds)
    wait = WebDriverWait(driver, 20)
    dropdown = wait.until(EC.presence_of_element_located((By.XPATH, '//select[@class="form-select"]')))

    # Create a Select object
    select = Select(dropdown)

    # Select the first option
    select.select_by_index(1)

    url = Select(dropdown).options[1].get_attribute('value')
    print("Downloading url:", url)

    urllib.request.urlretrieve(url, filename=os.path.join(download_path, "CTTI_new.zip"))
    print("Saved to", os.path.join(download_path, "CTTI_new.zip"))