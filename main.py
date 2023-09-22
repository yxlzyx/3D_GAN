#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import necessary libraries for array manipulation, plotting, and building neural networks
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv3D, Conv3DTranspose, BatchNormalization, LeakyReLU, Activation, MaxPooling3D,
    Concatenate, Flatten, Dense, Reshape
)
from tensorflow.keras.optimizers import Adam

# Set hyperparameters and configurations in a dictionary
config = {
    "latent_dim": 200,
    "discriminator_input_shape": (64, 64, 64, 1),
    "num_epochs": 50,
    "batch_size": 35,
    "num_samples": 5,
    "classifier_epochs": 3,
    "learning_rate_gen": 0.0025,
    "learning_rate_dis": 0.00010,
    "beta1_gen": 0.5,
    "beta1_dis": 0.5
}

# Function to load ModelNet10 dataset
def load_modelnet10_dataset():
    # Load the .npz file and unpack its arrays
    data = np.load("modelnet10.npz", allow_pickle=True)
    return data["train_voxel"], data["test_voxel"], data["train_labels"], data["test_labels"], data["class_map"]

# Function to show some sample 3D models
def show_samples(train_voxel, num_samples=config["num_samples"]):
    for i in range(num_samples):
        # Create 3D plot
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(train_voxel[i])
        plt.show()

# Function to plot 3D voxels
def plot_3d_voxels(voxel, threshold=0.5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.where(voxel > threshold)
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

# Function to generate and visualize samples using the generator
def generate_and_visualize_samples(generator, latent_dim=200):
    for _ in range(4):
        noise = np.random.normal(0, 1, (1, latent_dim))
        generated_voxel = generator.predict(noise)[0, :, :, :, 0]
        plot_3d_voxels(generated_voxel)


# Function to save generator and discriminator models
def save_models(generator, discriminator):
    generator.save('generator_model')
    discriminator.save('discriminator_model')

# Function to load a pre-trained discriminator model
def load_discriminator_model():
    return load_model('discriminator_model')


# Function to construct the generator neural network
def build_generator(latent_dim=config["latent_dim"]):
    noise = Input(shape=(latent_dim,))
    x = Reshape((1, 1, 1, latent_dim))(noise)
    # Generator architecture with Conv3DTranspose layers and Batch Normalization Layers
    x = Conv3DTranspose(512, kernel_size=4, strides=1, padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(256, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(1, kernel_size=4, strides=2, padding="same")(x)
    x = Activation('sigmoid')(x)
    return Model(noise, x, name="Generator")

# Function to construct the discriminator neural network
def build_discriminator(input_shape=config["discriminator_input_shape"]):
    input_voxel = Input(shape=input_shape)
    # Discriminator architecture with Conv3D layers and Batch Normalization Layers
    x = Conv3D(64, kernel_size=4, strides=2, padding="same")(input_voxel)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv3D(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv3D(256, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv3D(512, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv3D(1, kernel_size=4, strides=1, padding="valid")(x)
    x = Flatten()(x)
    x = Activation('sigmoid')(x)
    return Model(input_voxel, x, name="Discriminator")

# Function to compile the discriminator model
def compile_discriminator(discriminator):
    optimizer = Adam(learning_rate=config['learning_rate_dis'], beta_1=config['beta1_dis'])
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Function to compile the combined GAN model
def build_and_compile_combined(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator when training the generator
    noise = Input(shape=(config['latent_dim'],))
    generated_voxels = generator(noise)
    validity = discriminator(generated_voxels)
    combined = Model(noise, validity)
    optimizer = Adam(learning_rate=config['learning_rate_gen'], beta_1=config['beta1_gen'])
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    return combined

# Main function for GAN training
def train_gan(generator, discriminator, combined, train_voxel,
              num_epochs=config["num_epochs"],
              batch_size=config["batch_size"],
              latent_dim=config["latent_dim"]):
    # Initializations for labels and histories
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    d_loss_history = []
    g_loss_history = []
    
    # Initialize a variable to store the last batch accuracy of the discriminator
    last_batch_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training loop logic
        idx = np.random.randint(0, train_voxel.shape[0], batch_size)
        voxels = np.expand_dims(train_voxel[idx], axis=-1)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_voxels = generator.predict(noise)

        # Compute the loss for the real and fake images
        d_loss_real = discriminator.train_on_batch(voxels, valid)
        d_loss_fake = discriminator.train_on_batch(gen_voxels, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Compute the discriminator accuracy for the last batch
        last_batch_acc = 100 * d_loss[1]

        # Conditionally update the discriminator based on its last batch accuracy
        if last_batch_acc <= 80.0:
            d_loss_real = discriminator.train_on_batch(voxels, valid)
            d_loss_fake = discriminator.train_on_batch(gen_voxels, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        g_loss = combined.train_on_batch(noise, valid)

        print(f"Epoch {epoch + 1} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss}]")

        d_loss_history.append(d_loss[0])
        g_loss_history.append(g_loss)
        
    return d_loss_history, g_loss_history

# Function to build a classifier using the discriminator's layers
def build_classifier(discriminator, train_labels):
    # Logic for building classifier
    discriminator.trainable = False
    layer_outputs = [discriminator.layers[i].output for i in [3, 6, 9]]
    feature_model = Model(inputs=discriminator.input, outputs=layer_outputs)

    classifier_input = Input(shape=(64, 64, 64, 1))
    features = feature_model(classifier_input)

    maxpool_1 = MaxPooling3D(pool_size=(8, 8, 8))(features[0])
    maxpool_2 = MaxPooling3D(pool_size=(4, 4, 4))(features[1])
    maxpool_3 = MaxPooling3D(pool_size=(2, 2, 2))(features[2])

    concatenated = Concatenate(axis=-1)([Flatten()(maxpool_1), Flatten()(maxpool_2), Flatten()(maxpool_3)])

    num_classes = len(np.unique(train_labels))
    dense_layer = Dense(num_classes, activation='softmax')(concatenated)

    classifier = Model(classifier_input, dense_layer)
    return classifier

# Function to compile and train the classifier
def compile_and_train_classifier(classifier, train_voxel, train_labels,
                                 num_epochs=config["classifier_epochs"],
                                 batch_size=config["batch_size"]):
    classifier.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier_history = classifier.fit(train_voxel, train_labels, epochs=num_epochs, batch_size=batch_size)
    return classifier_history

# Function to plot classifier training history
def plot_classifier_history(classifier_history):
    # Plotting logic
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(classifier_history.history['loss'], label='Training Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(classifier_history.history['accuracy'], label='Training Accuracy')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
# Function to display classifier parameters
def display_classifier_parameters(classifier):
   print("Total number of parameters in classifier: ", classifier.count_params())
   
def main():
    # Load 3D dataset
    print("Loading dataset...")
    train_voxel, test_voxel, train_labels, test_labels, class_map = load_modelnet10_dataset()  

    print("TASK A STARTS: ")

    # Display a few samples from the dataset
    print("Displaying sample 3D objects...")
    show_samples(train_voxel, num_samples=5)
    
    # Build Generator and Discriminator
    print("Building and compiling the GAN...")
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Compile Discriminator
    compile_discriminator(discriminator)
    
    # Build and Compile Combined GAN Model
    combined = build_and_compile_combined(generator, discriminator)
    
    # Show total parameters in both Generator and Discriminator
    print("Total parameters in Generator:", generator.count_params())
    print("Total parameters in Discriminator:", discriminator.count_params())
    
    # Train the GAN
    print("Training the GAN...")
    d_loss_history, g_loss_history = train_gan(generator, discriminator, combined, train_voxel)
    
    # Plot Training Loss
    print("Plot loss History")
    plt.figure()
    plt.plot(d_loss_history, label="Discriminator Loss")
    plt.plot(g_loss_history, label="Generator Loss")
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # Generate and visualize synthetic samples
    print("Generating and visualizing samples...")
    generate_and_visualize_samples(generator, latent_dim=200)
    
    # Save the trained Generator and Discriminator models
    print("Saving generator and discriminator models...")
    save_models(generator, discriminator)
    
    print("TASK B STARTS: ")

    # Load the trained Discriminator model
    print("Loading saved discriminator model...")
    discriminator_loaded = load_discriminator_model()
    
    # Build and Train Classifier using loaded Discriminator
    print("Building and training classifier...")
    classifier = build_classifier(discriminator_loaded, train_labels)
    classifier_history = compile_and_train_classifier(classifier, train_voxel, train_labels)
    
    # Plot classifier training history and display its parameters
    print("Plotting classifier history...")
    plot_classifier_history(classifier_history)
    print("Displaying classifier parameters...")
    display_classifier_parameters(classifier)
    
# Execute the main function only if the script is run as the main module
if __name__ == "__main__":
    main()





   

