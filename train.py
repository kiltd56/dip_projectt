from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
num_classes = 26  # Number of English letters
batch_size = 32
num_epochs = 10

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'Data',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'Data',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(None, None, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(GlobalAveragePooling2D())  # Global Average Pooling layer instead of Flatten
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # 26 classes (English letters)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator)

# Save the trained model
model.save('FinalModel/hand_gesture2_model.h5')
