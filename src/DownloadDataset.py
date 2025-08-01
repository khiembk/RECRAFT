import argparse
import os
import requests 
import gdown
import tarfile
import zipfile
import subprocess

def download_satellite(root):
    path = root

    train_file = os.path.join(path, 'satellite_train.npy')
    test_file = os.path.join(path, 'satellite_test.npy')

    if not os.path.isfile(train_file):
        print("Downloading satellite data.")
        with open(train_file, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/satellite/satellite_train.npy").content)
        with open(test_file, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/satellite/satellite_test.npy").content)

    print("Download satellite complete.")

def download_spherical(root):

    if not os.path.isfile(root + '/s2_cifar100.gz'):
        print("downloading spherical data.")
        with open(root + '/s2_cifar100.gz', 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/spherical/s2_cifar100.gz").content)

    print("Download spherical complete.")

def download_deepsea(root):
    filename = root + '/deepsea_filtered.npz'

    if not os.path.isfile(filename):
        print("Downloading deepsea data.")
        with open(filename, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/deepsea/deepsea_filtered.npz").content)

    print("Download deepsea complete.")

def download_darcy_flow(root):
    
    TRAIN_PATH = os.path.join(root, 'piececonst_r421_N1024_smooth1.mat')
    TEST_PATH = os.path.join(root, 'piececonst_r421_N1024_smooth2.mat')

    if not os.path.isfile(TRAIN_PATH):
        print("downloading darcyflow data")
        with open(TRAIN_PATH, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth1.mat").content)
        with open(TEST_PATH, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth2.mat").content)

    print("Download darcy complete.")


def download_ecg(root):
    filename = root + '/challenge2017.pkl'

    if not os.path.isfile(filename):
        print("Downloading ecg data.")
        url = "https://drive.google.com/uc?export=download&id=1vp41dhFqCAsEBUld8tXfzC1lRiGm27Aq"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for request errors
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded ecg data to {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")
    else:
        print(f"File {filename} already exists.")


def download_ninapro(root):
    """
    Check if NinaPro dataset files exist in the specified directory and download them if missing.
    Then load and return the train, validation, and test data and labels.
    
    Args:
        root (str): Root directory where the NinaPro files will be stored.
    
    Returns:
        tuple: (train_data, train_labels, valid_data, valid_labels, test_data, test_labels)
    """
    # Ensure the directory exists
    path = os.path.join(root, 'ninaPro')
    os.makedirs(path, exist_ok=True)

    # Dictionary of files and their Google Drive direct download URLs
    files_to_download = {
        'ninapro_train.npy': 'https://drive.google.com/uc?id=1zcrpm9gPwKk_neZNJ6TygAq8Udmuc4LD',
        'label_train.npy': 'https://drive.google.com/uc?id=1xYd5EqUwxp0wBoIXfFSS6FjjdhVD1z-k',
        'ninapro_val.npy': 'https://drive.google.com/uc?id=1UYMJBh1T8ioLamefI-15QtHrx5yyyDtj',
        'label_val.npy': 'https://drive.google.com/uc?id=1pCRBQjgkx2yyf8K0ReOYmcY1l3BlJgci',
        'ninapro_test.npy': 'https://drive.google.com/uc?id=1rtZvlLnGvK9XSujVyH-faixpvMHLEiSV',
        'label_test.npy': 'https://drive.google.com/uc?id=1-PcyUoOmcFBUexSILpZYndiEioldzxMR'
    }

    # Check and download missing files
    for filename, url in files_to_download.items():
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path):
            print(f"Downloading {filename}...")
            try:
                gdown.download(url, file_path, quiet=False)
                print(f"Downloaded {filename} to {file_path}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                return None  # Return None if any download fails
        else:
            print(f"File {filename} already exists at {file_path}")


def download_cosmic(root):
    """
    Check if Cosmic dataset files exist in the specified directory, download and extract .tar files if missing.
    Then load and return the train and test directories.
    
    Args:
        root (str): Root directory where the Cosmic files will be stored.
    
    Returns:
        tuple: (train_dirs, test_dirs) if successful, None if any step fails.
    """
    # Ensure the directory exists
    path = os.path.join(root, 'cosmic')
    os.makedirs(path, exist_ok=True)

    # Dictionary of files and their Google Drive direct download URLs
    files_to_download = {
        'deepCR.ACS-WFC.train.tar': 'https://drive.google.com/uc?id=1lHsNhPM-73YFdi7V1mkEaa_WyYuz9Waf',
        'deepCR.ACS-WFC.test.tar': 'https://drive.google.com/uc?id=1Pc6p1r6TdGCFZMOYScINMj65sF0BnSod'
    }

    # Expected output files after extraction
    expected_files = ['train_dirs.npy', 'test_dirs.npy']

    # Check if all expected files exist
    all_files_exist = all(os.path.isfile(os.path.join(path, f)) for f in expected_files)

    if not all_files_exist:
        print("Some Cosmic dataset files are missing. Downloading and extracting...")
        for tar_filename, url in files_to_download.items():
            tar_filepath = os.path.join(path, tar_filename)
            
            # Download the .tar file if it doesn't exist
            if not os.path.isfile(tar_filepath):
                print(f"Downloading {tar_filename}...")
                try:
                    gdown.download(url, tar_filepath, quiet=False)
                    print(f"Downloaded {tar_filename} to {tar_filepath}")
                except Exception as e:
                    print(f"Error downloading {tar_filename}: {e}")
                    return None

            # Extract the .tar file
            print(f"Extracting {tar_filename}...")
            try:
                with tarfile.open(tar_filepath, 'r') as tar_ref:
                    tar_ref.extractall(path)
                print(f"Extracted {tar_filename} to {path}")
                # Optionally, remove the .tar file to save space
                os.remove(tar_filepath)
                print(f"Removed {tar_filename}")
            except tarfile.TarError as e:
                print(f"Error extracting {tar_filename}: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error during extraction of {tar_filename}: {e}")
                return None

    # Verify that all expected files are present after downloading and extracting
    for f in expected_files:
        if not os.path.isfile(os.path.join(path, f)):
            print(f"Error: Expected file {f} not found after downloading and extracting.")
            return None



def download_fsd_audio(root):
    """
    Check if the audio.zip file or its contents exist in the specified directory,
    download and unzip it if missing, ensuring the chunks subdirectory exists.
    
    Args:
        root (str): Root directory where the fsd directory will be created.
    
    Returns:
        bool: True if the download and unzip process is successful or files already exist,
              False if any step fails.
    """
    # Ensure the directory exists
    path = os.path.join(root, 'fsd')
    os.makedirs(path, exist_ok=True)

    # File to download
    zip_filename = 'audio.zip'
    zip_filepath = os.path.join(path, zip_filename)
    url = 'https://drive.google.com/uc?id=1IDOkm7IpFfgByDZNM6daxZaVzVJVBNid'

    

    # Download the zip file if it doesn't exist
    if not os.path.isfile(zip_filepath):
        print(f"Downloading {zip_filename}...")
        try:
            gdown.download(url, zip_filepath, quiet=False)
            print(f"Downloaded {zip_filename} to {zip_filepath}")
        except Exception as e:
            print(f"Error downloading {zip_filename}: {e}")
            return False

    # Unzip the file
    print(f"Unzipping {zip_filename}...")
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(path)
        print(f"Unzipped {zip_filename} to {path}")
        # Optionally, remove the zip file to save space
        os.remove(zip_filepath)
        print(f"Removed {zip_filename}")
    except zipfile.BadZipFile as e:
        print(f"Error unzipping {zip_filename}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during unzipping {zip_filename}: {e}")
        return False


def download_psicov(root):
    """
    Check if the protein.zip file or its contents exist in the specified directory,
    download and unzip it if missing, ensuring the protein directory contains the extracted files.
    
    Args:
        root (str): Root directory where the protein directory will be created.
    
    Returns:
        bool: True if the download and unzip process is successful or files already exist,
              False if any step fails.
    """
    # Ensure the directory exists
    path = os.path.join(root, 'protein')
    os.makedirs(path, exist_ok=True)

    # File to download
    zip_filename = 'protein.zip'
    zip_filepath = os.path.join(path, zip_filename)
    url = 'https://drive.google.com/uc?id=1nBs6qJJPK8qNzsmFo3waPsGzHFHplLMo'

    # Check if the protein directory contains any files (indicating successful prior extraction)
    if os.path.isdir(path) and any(os.listdir(path)):
        print(f"Directory {path} already contains files. Skipping download and extraction.")
        return True

    # Download the zip file if it doesn't exist
    if not os.path.isfile(zip_filepath):
        print(f"Downloading {zip_filename}...")
        try:
            gdown.download(url, zip_filepath, quiet=False)
            print(f"Downloaded {zip_filename} to {zip_filepath}")
        except Exception as e:
            print(f"Error downloading {zip_filename}: {e}")
            return False

    # Unzip the file
    print(f"Unzipping {zip_filename}...")
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(path)
        print(f"Unzipped {zip_filename} to {path}")
        # Optionally, remove the zip file to save space
        os.remove(zip_filepath)
        print(f"Removed {zip_filename}")
    except zipfile.BadZipFile as e:
        print(f"Error unzipping {zip_filename}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during unzipping {zip_filename}: {e}")
        return False

    # Verify that the protein directory contains files after unzipping
    if not any(os.listdir(path)):
        print(f"Error: Directory {path} is empty after unzipping.")
        return False

    print("Protein dataset processed successfully.")
    return True

def download_cifar100(root):
    """
    Check if the cifar-100-python.tar.gz file or its contents exist in the specified directory,
    download and extract it if missing, ensuring the cifar-100-python directory exists.
    
    Args:
        root (str): Root directory where the CIFAR-100 files will be stored.
    
    Returns:
        bool: True if the download and extraction process is successful or files already exist,
              False if any step fails.
    """
    # Ensure the directory exists
    os.makedirs(root, exist_ok=True)

    # File to download
    tar_filename = 'cifar-100-python.tar.gz'
    tar_filepath = os.path.join(root, tar_filename)
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    # Check if the cifar-100-python directory exists (indicating successful prior extraction)
    extracted_dir = os.path.join(root, 'cifar-100-python')
    if os.path.isdir(extracted_dir) and any(os.listdir(extracted_dir)):
        print(f"Directory {extracted_dir} already contains files. Skipping download and extraction.")
        return True

    # Download the .tar.gz file if it doesn't exist
    if not os.path.isfile(tar_filepath):
        print(f"Downloading {tar_filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for HTTP errors
            with open(tar_filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {tar_filename} to {tar_filepath}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {tar_filename}: {e}")
            return False
        
def download_pde(root, dataset):
    print("dataset: ", dataset)
    if dataset == 'BG':
        filename = '1D_Burgers_Sols_Nu1.0.hdf5' 
        target_filepath = os.path.join(root, filename)
        url = 'https://darus.uni-stuttgart.de/api/access/datafile/281365'
      
        
    elif dataset == '1DCFD':
        filename = '1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5'
        target_filepath = os.path.join(root, filename)
        url = 'https://darus.uni-stuttgart.de/api/access/datafile/164668'


    elif dataset == 'ADV':
        filename = '1D_Advection_Sols_beta0.4.hdf5'
        target_filepath = os.path.join(root, filename)
        url = 'https://darus.uni-stuttgart.de/api/access/datafile/255674'

    elif dataset == 'DS':
        filename = '1D_diff-sorp_NA_NA.h5'
        target_filepath = os.path.join(root, filename)
        url = 'https://darus.uni-stuttgart.de/api/access/datafile/133020'

    elif dataset == 'RD':
        filename = 'ReacDiff_Nu0.5_Rho1.0.hdf5'
        target_filepath = os.path.join(root, filename)
        url = 'https://darus.uni-stuttgart.de/api/access/datafile/133177'

    elif dataset == 'SW':
        filename = '2D_rdb_NA_NA.h5'
        target_filepath = os.path.join(root, filename)
        url = 'https://darus.uni-stuttgart.de/api/access/datafile/133021'

    elif dataset == 'DC':
        filename = '2D_DarcyFlow_beta0.1_Train.hdf5'
        target_filepath = os.path.join(root, filename)
        url = 'https://darus.uni-stuttgart.de/api/access/datafile/133218'

    elif dataset == 'RD2D':
        filename = '2D_diff-react_NA_NA.h5'
        target_filepath = os.path.join(root, filename)
        url = 'https://darus.uni-stuttgart.de/api/access/datafile/133017'
      
    os.makedirs(root, exist_ok=True)

    
    if os.path.isfile(target_filepath):
        print(f"File {target_filepath} already exists. Skipping download.")
        return True
    
    

    # Download the file using wget
    print(f"Downloading {filename} from {url}...")
    try:
        result = subprocess.run(['wget', '-O', target_filepath, url], capture_output=True, text=True, check=True)
        print(f"Downloaded file to {target_filepath}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file with wget: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: wget command not found. Please ensure wget is installed or use an alternative method.")
        return False
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        return False

    


def main(dataset, root):
    
    if dataset[:3] == 'PDE':
        if root is None:
            root = './PDE_dataset'
        else:
            root = root + '/PDE_dataset'

        download_pde(root, dataset=dataset[3:])
    else:
        if root is None:
            root = './datasets'
        else:
            root = root + '/datasets'
        
        
        if dataset == "CIFAR100":
            
            download_cifar100(root)
    
        elif dataset == "SPHERICAL":
            download_spherical(root)
        elif dataset == "DEEPSEA":
            download_deepsea(root)
        elif dataset == "DARCY-FLOW-5":
            download_darcy_flow(root)
       
        elif dataset == 'PSICOV':
            download_psicov(root)
        elif dataset == "ECG":
            download_ecg(root)
        elif dataset == "SATELLITE":
            download_satellite(root)
        elif dataset == "NINAPRO":
            download_ninapro(root)
        elif dataset == "COSMIC":
            download_cosmic(root)
        
        elif dataset == "FSD":
            download_fsd_audio(root)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Dataset')
    parser.add_argument('--dataset', type=str, default=None, help='Name of dataset to download.')
    parser.add_argument('--path', type=str, default=None, help='path of dataset to download.')
    
    args = parser.parse_args()
    dataset = args.dataset
    path = args.path
    main(dataset, path)