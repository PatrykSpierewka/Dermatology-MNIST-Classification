import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


#Data mining
data = np.load('Data.npz')
print(data.files, "\n")

print(data['train_images'].shape)
print(data['train_images'].ndim, "\n")

print(data['val_images'].shape)
print(data['val_images'].ndim, "\n")

print(data['test_images'].shape)
print(data['test_images'].ndim, "\n")

#70% train, 10% val, 20% test

print(data['train_labels'].shape)
print(data['train_labels'].ndim, "\n")

print(data['val_labels'].shape)
print(data['val_labels'].ndim, "\n")

print(data['test_labels'].shape)
print(data['test_labels'].ndim, "\n")


#Data display
images_train = data['train_images']
labels_train = data['train_labels']

num_examples_train = 7
fig, axes = plt.subplots(1, num_examples_train, figsize=(15, 2.5))
fig.suptitle("Train examples")

for i in range(num_examples_train):
    axes[i].imshow(images_train[i])
    axes[i].set_title('Label: {}'.format(labels_train[i][0]))
    axes[i].axis('off')
plt.show()



images_val = data['val_images']
labels_val = data['val_labels']

num_examples_val = 7
fig, axes = plt.subplots(1, num_examples_val, figsize=(15, 2.5))
fig.suptitle("Val examples")

for i in range(num_examples_val):
    axes[i].imshow(images_val[i])
    axes[i].set_title('Label: {}'.format(labels_val[i][0]))
    axes[i].axis('off')
plt.show()



images_test = data['test_images']
labels_test = data['test_labels']

num_examples_test = 7
fig, axes = plt.subplots(1, num_examples_test, figsize=(15, 2.5))
fig.suptitle("Test examples")

for i in range(num_examples_test):
    axes[i].imshow(images_test[i])
    axes[i].set_title('Label: {}'.format(labels_test[i][0]))
    axes[i].axis('off')
plt.show()

labels_train = labels_train.reshape(data['train_images'].shape[0])#reshape klas treningowych z dwywymiarowej macierzy do jednowymiarowego wektora
labels_val = labels_val.reshape(data['val_images'].shape[0])#reshape klas walidacyjnych z dwywymiarowej macierzy do jednowymiarowego wektora
labels_test = labels_test.reshape(data['test_images'].shape[0])#reshape klas testowych z dwywymiarowej macierzy do jednowymiarowego wektora

print(labels_train.shape, labels_train.ndim)
print(labels_val.shape, labels_val.ndim)
print(labels_test.shape, labels_test.ndim)


#Number of records for individual classes
train_class_counts = np.bincount(labels_train)
print(train_class_counts)

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

plt.figure(figsize=(4, 3))
sns.barplot(x=class_labels, y=train_class_counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Count Train')
plt.show()



val_class_counts = np.bincount(labels_val)
print(val_class_counts)

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

plt.figure(figsize=(4, 3))
sns.barplot(x=class_labels, y=val_class_counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Count Val')
plt.show()



test_class_counts = np.bincount(labels_test)
print(test_class_counts)

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

plt.figure(figsize=(4, 3))
sns.barplot(x=class_labels, y=test_class_counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Count Test')
plt.show()


#Preprocessing for training set (grayscale, flattening, normalization)
train_features = tf.image.rgb_to_grayscale(images_train)
train_labels_gray = labels_train
train_images_gray = train_features
random_state = 42
print(train_images_gray.shape)

num_examples_gray = 7
fig, axes = plt.subplots(1, num_examples_gray, figsize=(10, 2))
for i in range(num_examples_gray):
    axes[i].imshow(train_images_gray[i], cmap="gray")
    axes[i].set_title('Label: {}'.format(train_labels_gray[i]))
    axes[i].axis('off')

plt.show()
gray_train_features = train_features.numpy()
print(gray_train_features.shape, "\n")

flat_train_features = np.zeros((gray_train_features.shape[0], gray_train_features.shape[1] * gray_train_features.shape[2]))
print(flat_train_features.shape, "\n")

for i in range(gray_train_features.shape[0]):
    flat_train_features[i] = gray_train_features[i].reshape(gray_train_features.shape[1] * gray_train_features.shape[2])

print(flat_train_features.shape, "\n")
print(flat_train_features)

flat_train_scaled = flat_train_features

flat_train_scaled = np.round(flat_train_features / 255.0, 3)
flat_train_scaled = tf.image.convert_image_dtype(flat_train_scaled, tf.float32)

print(flat_train_scaled.shape, "\n")
print(flat_train_scaled)


#Preprocessing for test set (grayscale, flattening, normalization)
test_features = tf.image.rgb_to_grayscale(images_test)
test_labels_gray = labels_test
test_images_gray = test_features
print(test_images_gray.shape)

num_examples_gray = 7
fig, axes = plt.subplots(1, num_examples_gray, figsize=(10, 2))
for i in range(num_examples_gray):
    axes[i].imshow(test_images_gray[i], cmap="gray")
    axes[i].set_title('Label: {}'.format(test_labels_gray[i]))
    axes[i].axis('off')

plt.show()
gray_test_features = test_features.numpy()
print(gray_test_features.shape,"\n")

flat_test_features = np.zeros((gray_test_features.shape[0], gray_test_features.shape[1] * gray_test_features.shape[2]))
print(flat_test_features.shape, "\n")

for i in range(gray_test_features.shape[0]):
    flat_test_features[i] = gray_test_features[i].reshape(gray_test_features.shape[1] * gray_test_features.shape[2])

print(flat_test_features.shape, "\n")
print(flat_test_features)

flat_test_scaled = flat_test_features

flat_test_scaled = np.round(flat_test_features / 255.0, 3)
flat_test_scaled = tf.image.convert_image_dtype(flat_test_scaled, tf.float32)

df_test_features = pd.DataFrame(flat_test_scaled)
df_test_labels = pd.DataFrame(test_labels_gray)
df_test_labels = df_test_labels.rename(columns={0: 'labels'})
df_test = pd.concat([df_test_features, df_test_labels], axis=1)

df_test_split = df_test.copy()

X_test = df_test_split.drop(['labels'], axis=1)
y_test = df_test_split['labels']

print(X_test.shape)
print(y_test.shape)
print(df_test_split)


#Data augmentation for the training set, upsampling with using tf.ImageDataGenerator
datagen_augment = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    zca_whitening = True,
    samplewise_std_normalization = True,
    brightness_range = (15, 15),

)

train_data_augmented = tf.convert_to_tensor(flat_train_scaled)
train_data_augmented = tf.reshape(train_data_augmented, (train_data_augmented.shape[0], 28, 28, 1))

augmented_data = []
augmented_labels = []

target_num_examples = 7500

for label in range(len(class_labels)):
    data = train_data_augmented[train_labels_gray == label]
    num_data = len(data)

    if num_data < target_num_examples:
        num_augmented_data = target_num_examples - num_data
        augment_iter = datagen_augment.flow(data, batch_size=1, shuffle=True)

        augmented_data.extend(augment_iter.next() for _ in range(num_augmented_data))
        augmented_labels.extend([label] * num_augmented_data)

augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)

flat_train_scaled_augmented = np.concatenate((flat_train_scaled, augmented_data.reshape(-1, 784)))
train_labels_gray_augmented = np.concatenate((train_labels_gray, augmented_labels))

print(flat_train_scaled_augmented.shape)
print(train_labels_gray_augmented.shape)


#Number of records for individual classes from the training set after augmentation
train_class_counts_augmented = np.bincount(train_labels_gray_augmented)

plt.figure(figsize=(4, 3))
sns.barplot(x=class_labels, y=train_class_counts_augmented)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Count Train (Augmented)')
plt.show()


#Training set down-sampling
augmented_data = flat_train_scaled_augmented
augmented_labels = train_labels_gray_augmented

df_augmented = pd.DataFrame(augmented_data)
df_augmented['labels'] = augmented_labels

target_num_examples = 3000

reduced_data = df_augmented.groupby('labels').apply(lambda x: x.sample(n=min(target_num_examples, len(x)), random_state=random_state)).reset_index(drop=True)

flat_train_scaled_reduced = reduced_data.iloc[:, :-1].values
train_labels_gray_reduced = reduced_data['labels'].values

print(flat_train_scaled_reduced.shape)
print(train_labels_gray_reduced.shape)


#Final preparation of data for the classifier
df_train_features = pd.DataFrame(flat_train_scaled_reduced)
df_train_labels = pd.DataFrame(train_labels_gray_reduced)
df_train_labels = df_train_labels.rename(columns={0: 'labels'})
df_train = pd.concat([df_train_features, df_train_labels], axis=1)

shuffled_train_df = df_train.sample(frac = 1, random_state = random_state).reset_index(drop=True)

df_train_split = shuffled_train_df.copy()
X_train = df_train_split.drop(['labels'], axis = 1)
y_train = df_train_split['labels']

print(X_train.shape)
print(y_train.shape)


#Classification using the MLPClassifier from the sklearn package
clf = MLPClassifier(hidden_layer_sizes=(784, 196, 7), random_state=random_state)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", round(accuracy, 3))

cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True).set(title='Confusion Matrix')

classification_report_result = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report_result)
