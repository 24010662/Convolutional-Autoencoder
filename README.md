# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Image denoising is a key task in computer vision where the objective is to remove unwanted noise from images while preserving important visual details.  
Traditional filtering techniques like Gaussian or median filters often blur the image, whereas **Convolutional Autoencoders (CAEs)** can learn efficient representations to restore clean images automatically.

In this experiment, a convolutional autoencoder is developed to denoise images by learning compressed representations of noisy inputs.  
The dataset used is the **MNIST** dataset, consisting of 28×28 grayscale handwritten digit images.  
Random noise is added to the images to train the model, which learns to reconstruct the original clean images from their noisy counterparts.



## DESIGN STEPS

### Step 1:
Import the required libraries such as PyTorch, Torchvision, NumPy, and Matplotlib.

### Step 2:
Load the MNIST dataset and add random noise to create noisy input images.

### Step 3:
Normalize the image data and prepare dataloaders for training and testing.

### Step 4:
Define the Convolutional Autoencoder model with encoder and decoder layers.

### Step 5:
Specify the loss function (MSELoss) and optimizer (Adam).

### Step 6:
Train the model using noisy images as input and clean images as target output.

### Step 7:
Monitor training loss to ensure the model learns effective noise removal.

### Step 8:
Test the trained model on noisy test images to evaluate performance.

### Step 9:
Visualize original, noisy, and denoised images for comparison.

## PROGRAM
### Developed By: VUTUKURI SAI KUMAR REDDY
### Register Number: 212224230307
```py
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
      super(DenoisingAutoencoder,self).__init__()
      self.encoder=nn.Sequential(
          nn.Conv2d(1, 16, 3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(16, 32, 3, stride=2, padding=1),
          nn.ReLU()
      )
      self.decoder=nn.Sequential(
          nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
          nn.Sigmoid()
      )
    def forward(self,x):
      x=self.encoder(x)
      x=self.decoder(x)
      return x


# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


## Model Training
# Training Function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:  VUTUKURI SAI KUMAR REDDY                 ")
    print("Register Number:   212224230307           ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

     

```
## OUTPUT

### Model Summary

<img width="550" height="384" alt="image" src="https://github.com/user-attachments/assets/a1fb5476-1b0a-4c9b-a7aa-95ab17fe072a" />


### Original vs Noisy Vs Reconstructed Image


<img width="1543" height="671" alt="image" src="https://github.com/user-attachments/assets/8a20b390-49a0-49b6-89a6-e51a5064ca0e" />



<img width="1317" height="556" alt="image" src="https://github.com/user-attachments/assets/11ad7723-c20d-4773-8c78-5d77d9521a4d" />



## RESULT
The convolutional autoencoder model was successfully implemented and effectively removed noise from images, restoring them close to their original quality.
