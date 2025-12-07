
import zipfile
with zipfile.ZipFile('train.zip', 'r') as zip_ref:
    zip_ref.extractall()
with zipfile.ZipFile('test_set.zip', 'r') as zip_ref:
    zip_ref.extractall()
with zipfile.ZipFile('validation_set.zip', 'r') as zip_ref:
    zip_ref.extractall()