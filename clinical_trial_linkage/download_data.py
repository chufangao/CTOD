import os

import argparse
import zipfile
import wget
import time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download FDA orange book and drug bank data')
    
    # savepath
    parser.add_argument('--save_path', type=str, default='', help='path to save the data')
    

    args = parser.parse_args()
    
    # Desired download path
    download_path = os.path.join(args.save_path, 'other_downloads')
    if not os.path.exists(download_path):
        os.makedirs(download_path) 
        
    #if download_path is not empty, delete all files in the download folder
    for file in os.listdir(download_path):
        print(f"Deleting old {file}")
        os.remove(os.path.join(download_path, file))
    
    # desired FDA path
    fda_path = os.path.join(args.save_path, 'FDA')
    if not os.path.exists(fda_path):
        os.makedirs(fda_path)
    fda_new = os.path.join(fda_path, 'FDA_new')
    if not os.path.exists(fda_new):
        os.makedirs(fda_new)
    fda_old = os.path.join(fda_path, 'FDA_old')
    if not os.path.exists(fda_old):
        os.makedirs(fda_old)
    
    # if FDA_new is not empty, move all files to FDA_old folder
    if os.listdir(fda_new):
        fda_old = os.path.join(fda_old,time.strftime("%Y%m%d"))
        if not os.path.exists(fda_old):
            os.makedirs(fda_old)
        print("Moving files from FDA_new to FDA_old")
        for file in os.listdir(fda_new):
            os.rename(os.path.join(fda_new, file), os.path.join(fda_old, file))
            
    
    # drug bank path
    drug_bank_path = os.path.join(args.save_path, 'drug_bank')
    if not os.path.exists(drug_bank_path):
        os.makedirs(drug_bank_path)
    
    # move path to download_path
    os.chdir(download_path)
    
    # download FDA orange book data
    print("Downloading FDA orange book data")
    url = 'https://www.fda.gov/media/76860/download?attachment'
    wget.download(url)
    
    #unzip the file to FDA_new
    #get the file name in the download folder
    file_name = os.listdir(download_path)[0]
    #unzip the file
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(fda_new)
        
    # delete the download path
    os.remove(file_name)
    os.removedirs(download_path)
    
    # download drug code directory
    print("Downloading drug code directory")
    os.chdir(drug_bank_path)
    #check if there are any .json.zip files in the folder
    for file in os.listdir(drug_bank_path):
        if file.endswith('.json.zip'):
            os.remove(file)
    wget.download('https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip')
    

    
    # download drug bank data - TODO (need access to download drugbank data, currently using older data)
    
    
    
    
