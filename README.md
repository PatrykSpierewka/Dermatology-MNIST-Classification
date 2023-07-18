# Dermatology-MNIST-Classification
### Supervised learning used for skin diseases classification based on data from: https://zenodo.org/record/5208230
The data are skin images with dimensions of 28x28 pixels and RGB color scale. A label has been defined for each image. The downloaded data set was originally divided into training (70%), test (20%) and validation (10%) and contains a seven-class division: 'akiec': 'Actinic keratoses', 'bcc': 'Basal cell carcinoma', 'bkl': 'Benign keratosis-like lesions', 'df': 'Dermatofibroma', 'nv': 'Melanocytic nevi', 'vasc': 'Vascular lesions', 'mel': 'Melanoma'. There is a problem of strong imbalance of records for classes.


<p align="center">
  <img src="https://github.com/PatrykSpierewka/Dermatology-MNIST-Classification/assets/101202344/8850706e-5773-49cc-b167-3db63d52237b" alt="dermo" style="width: 75%; height: auto;">
</p>

### Data preprocessing:
- greyscale: image dim from 28x28x3 to 28x28x1 (train set, test set),
- flattening: image dim from 28x28x1 to 784x1 (train set, test set),
- normalization: pixels values from 0-255 to 0.0-1.0 (train set, test set),
- augmentation: generating additional, similar images for each of the minority classes using tensorflow keras (only train set).

### MLP classification
