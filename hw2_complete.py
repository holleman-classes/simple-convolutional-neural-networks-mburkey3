### Add lines to import modules as needed
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, Input
## 
print(f"TensorFlow version: {tf.__version__}\n")

def build_model1():

    inputs = Input(shape=inputShape) # SET THIS UP LATER IN CODE
    y = inputs
    y = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4))(y)
    y = layers.Flatten()(y)
    y = layers.Dense(units=128)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(units=10)(y)

    model1 = tf.keras.models.Model(inputs=inputs, outputs=y) # Add code to define model 1.
    return model1

def build_model2():

    inputs = Input(shape=inputShape) # SET THIS UP LATER IN CODE
    y = inputs
    y = layers.SeparableConv2D(32, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.SeparableConv2D(64, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.SeparableConv2D(128, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.SeparableConv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.SeparableConv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.SeparableConv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.SeparableConv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4))(y)
    y = layers.Flatten()(y)
    y = layers.Dense(units=128)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(units=10)(y)

    model2 = tf.keras.models.Model(inputs=inputs, outputs=y) # Add code to define model 2.
    return model2

def build_model3():
    
    inputs = Input(shape=inputShape)
    y = inputs
    # Block 1
    y = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    block1Out = y
    block1Out = layers.Dropout(0.2)(block1Out)
    # Block 2
    y = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(block1Out)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)
    # Block 3
    y = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    block1Out = layers.Conv2D(128, kernel_size=1, strides=4)(block1Out)
    block3Out = layers.add([y, block1Out])
    block3Out = layers.Dropout(0.2)(block3Out)
    # Block 4
    y = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(block3Out)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)
    # Block 5
    y = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    block5Out = layers.add([y, block3Out])
    block5Out = layers.Dropout(0.2)(block5Out)
    # Block 6
    y = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(block5Out)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)
    # Block 7
    y = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)
    block7Out = layers.add([y, block5Out])
    block7Out = layers.Dropout(0.2)(block7Out)
    # Block 8
    y = layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4))(block7Out)
    y = layers.Flatten()(y)
    y = layers.Dense(units=128)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(units=10)(y)

    model3 = tf.keras.models.Model(inputs=inputs, outputs=y) # Add code to define model 3.
    ## This one should use the functional API so you can create the residual connections
    return model3

def build_model50k():

    inputs = Input(shape=inputShape)
    y = inputs
    y = layers.Conv2D(16, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    block1Out = y
    block1Out = layers.Dropout(0.2)(block1Out)

    y = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(block1Out)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)

    y = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(y)
    y = layers.BatchNormalization()(y)
    block1Out = layers.Conv2D(64, kernel_size=1, strides=4)(block1Out)
    block3Out = layers.add([y, block1Out])
    block3Out = layers.Dropout(0.2)(block3Out)

    y = layers.Conv2D(32, kernel_size=3, strides=1, padding='same')(block3Out)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)

    y = layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4))(y)
    y = layers.Flatten()(y)
    y = layers.Dense(units=64)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(units=10)(y)

    model = tf.keras.models.Model(inputs=inputs, outputs=y) # Add code to define model 1.
    return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

    epochsToRun = 50 ## SET TO 50 WHEN DONE

    seed = 777
    tf.random.set_seed(seed)
    np.random.seed(seed)
    ########################################
    ## Add code here to Load the CIFAR10 data set
    (trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.cifar10.load_data()
    
    # validation set
    valPercent = 0.2
    numValSamples = int(len(trainImages)*valPercent)
    valIdx = np.random.choice(np.arange(len(trainImages)), size=numValSamples, replace=False)
    trainIdx = np.setdiff1d(np.arange(len(trainImages)), valIdx)
    # Split the images
    valImages = trainImages[valIdx, :]
    trainImages = trainImages[trainIdx, :]

    valLabels = trainLabels[valIdx, :]
    trainLabels = trainLabels[trainIdx, :]
    
    # Prep data
    trainLabels = trainLabels.squeeze()
    testLabels = testLabels.squeeze()
    valLabels = valLabels.squeeze()
    
    inputShape = trainImages.shape[1:]
    trainImages = trainImages / 255.0
    testImages = testImages / 255.0
    valImages = valImages / 255.0

    print("Training Images range from {:2.5f} to {:2.5f}".format(np.min(trainImages), np.max(trainImages)))
    print("Test     Images range from {:2.5f} to {:2.5f}\n".format(np.min(testImages), np.max(testImages)))
    print(f"Training:   {len(trainLabels)} and image set has shape: {trainImages.shape}")
    print(f"Validation: {len(valLabels)  } and image set has shape: {valImages.shape}")
    print(f"Test:       {len(testLabels) } and image set has shape: {testImages.shape}\n")
    ########################################
    
    
    ## Build and train model 1
    model1 = build_model1()
    # compile and train model 1.
    model1.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    # model1.summary()

    # Run model
    print("\n\tBegin Model 1")
    # model1.fit(trainImages, trainLabels,
    #           validation_data=(valImages, valLabels),
    #           epochs=epochsToRun)
    print("\n\tModel 1 Compelete\n")


    ## Build, compile, and train model 2 (DS Convolutions)
    model2 = build_model2()
    # compile model 2
    model2.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    # model2.summary()

    # Run model
    print("\n\tBegin Model 2")
    # model2.fit(trainImages, trainLabels,
    #           validation_data=(valImages, valLabels),
    #          epochs=epochsToRun)
    print("\n\tModel 2 Compelete\n")


    ## Build model 3
    model3 = build_model3()
    # compile model 3
    model3.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    # model3.summary()

    # Run Model
    print("\n\tBegin Model 3")
    # model3.fit(trainImages, trainLabels,
    #           validation_data=(valImages, valLabels),
    #          epochs=epochsToRun)
    print("\n\tModel 3 Compelete\n")


    # Build model with 50k parameters
    model50k = build_model50k()
    # Compile the model
    model50k.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model50k.summary()
    # model50k.save("best_model.h5")

    # Fit model to data
    print("\n\tBegin Model 50k")
    model50k.fit(trainImages, trainLabels,
              validation_data=(valImages, valLabels),
             epochs=epochsToRun)
    print("\n\tModel 50k Compelete\n")