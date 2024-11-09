import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(128, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 128, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(128, 32, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 8, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(8, output_size, dtype=torch.float64)
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)

        return out 
    

class ElecDataset(Dataset):
  def __init__(self, x_data, y_data):
    scaler = MinMaxScaler()

    columns_to_scale = x_data.columns[1:]
    x_data[columns_to_scale] = scaler.fit_transform(x_data[columns_to_scale])
    
    self.x_data = x_data.fillna(0)
    self.y_data = y_data.fillna(0)

  def __getitem__(self, index):
    target_y = self.y_data['하루전가격(원/kWh)'].iloc[index]
    targets = self.x_data.drop(columns='datetime').iloc[index].to_numpy()

    return torch.from_numpy(targets), torch.tensor(target_y)

  def __len__(self):
    return int(len(self.y_data))
  
def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model = model.to(device)  # Move model to GPU/CPU

    train_history = []
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        
        running_loss = 0.0  # To keep track of loss
        for inputs, targets in tqdm(train_loader, ncols=100):

            # Move data to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear the gradients
            loss.backward()        # Compute gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()        # Update model parameters

            running_loss += loss.item()
            
            # print(loss.item())

        # Print the loss after each epoch
        avg_loss = running_loss / len(train_loader)
        train_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")
    return model, train_history