import zipfile

# Util function to unzip datasets if necessary
def unzip_dataset(dataset_zip):
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall()
