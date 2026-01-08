from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
import os

import argparse
import zipfile



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download CTTI data')
    
    # savepath
    parser.add_argument('--save_path', type=str, default='', help='path to save the data')
    
    args = parser.parse_args()
    # Desired download path
    download_path = os.path.join(args.save_path, 'CTTI_raw')
    
    new_path = os.path.join(args.save_path, 'CTTI_new')
    old_path = os.path.join(args.save_path, 'CTTI_old')
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    if not os.path.exists(old_path):
        os.makedirs(old_path)
        
    # if CTTI_new is not empty, move all files to CTTI_old folder
    if os.listdir(new_path):
        print("Moving files from CTTI_new to CTTI_old")
        for file in os.listdir(new_path):
            os.rename(os.path.join(new_path, file), os.path.join(old_path, file))
    
    # delete any files in the CTTI_new folder
    for file in os.listdir(new_path):
        print(f"Deleting {file}")
        os.remove(os.path.join(new_path, file))
    
    # delete any files in the download folder
    for file in os.listdir(download_path):
        print(f"Deleting old {file}")
        os.remove(os.path.join(download_path, file))

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

    # Wait until the dropdown is populated (wait up to 20 seconds)
    wait = WebDriverWait(driver, 20)
    dropdown = wait.until(EC.presence_of_element_located((By.XPATH, '//select[@class="form-select"]')))

    # Create a Select object
    select = Select(dropdown)

    # Select the first option
    select.select_by_index(1)

    # Wait for the file to download (adjust as needed)
    # check if .crdownload file is still being downloaded
    time.sleep(10)
    start = time.time()
    while any(fname.endswith('.crdownload') for fname in os.listdir(download_path)):
        time.sleep(10)
        if time.time() - start > 600:
            print("File download timed out")
            break

    # Close the driver
    driver.quit()

    print(f"File downloaded to: {download_path}")
    
    #get the name of the downloaded file
    file_name = os.listdir(download_path)[0]
    
    with zipfile.ZipFile(os.path.join(download_path, file_name), 'r') as zip_ref:
        zip_ref.extractall(new_path)
    
    print(f"File extracted to: {new_path}")
    
    