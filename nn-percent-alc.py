import torch
import torch.nn as nn
from torchviz import make_dot
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score



# sample dataset with binary features and target probabilities
# data = {
#     'maniHead': [1, 0, 1, 0],  # Binary manipulation for head
#     'maniTail': [0, 1, 1, 0],  # Binary manipulation for tail
#     'maniBody': [1, 0, 0, 1],  # Binary manipulation for body
#     'head_prob': [0.1, 0.7, 0.1, 0.1],  # Target probability for head
#     'normal_prob': [0.7, 0.1, 0.1, 0.1],  # Target probability for normal
#     'tail_prob': [0.2, 0.2, 0.4, 0.2]  # Target probability for tail
# }

# df = pd.DataFrame(data)

# df = pd.read_csv('AlcoholPercentage.csv')
df = pd.read_csv('AlcoholPercentageCombined.csv')

X = df[['Drug','HeadMani', 'TrunkMani', 'TailMani', 'PrePharynxMani', 'PostPharynxMani','PharynxMani']]  # Features
y = df[['MorphologyEye', 'MorphologyHeadPharyn', 'MorphologyNormal', 'MorphologyTailPharyn','MorphologyLethal', 'MorphologyHead']]  # Target probabilities

# convert data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)
#y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # Reshape y to match model output

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# calculate and display statistics
def display_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'R² Score (Coefficient of Determination): {r2:.4f}')

# define the neural network architecture for regression
class MultiOutputNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiOutputNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()  # Add a sigmoid activation function
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.softmax(x)
        # x = self.sigmoid(x)  # Apply sigmoid activation
        return x

# define the size of input, hidden, and output layers
input_size = X_train.shape[1]  # Number of features (maniHead, maniTail, maniBody)

hidden_size = 64
#hidden_size = 32
# hidden_size = 128

# hidden_size = 32
output_size = y_train.shape[1] 

# initialize the model, criterion, and optimizer
model = MultiOutputNN(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multilabel classification
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.01)

#PRINT
dummy_input = torch.randn(1, input_size)

# pass the dummy input through the model to generate a graph
model.eval()  
output = model(dummy_input)

# generate the visualization
make_dot(output, params=dict(model.named_parameters()))
#end print

# training loop
num_epochs = 1000
#num_epochs = 1000

np.set_printoptions(suppress=True)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# validation
with torch.no_grad():
    model.eval()
    val_outputs = model(X_test)
    val_loss = criterion(val_outputs, y_test)
    print(f'Validation Loss: {val_loss.item():.4f}')

     # convert tensors to NumPy arrays for calculating metrics
    y_true_np = y_test.numpy()
    y_pred_np = val_outputs.numpy()

    display_metrics(y_true_np, y_pred_np)

predicted_percentages = model(X_test).detach().numpy()
print("Predicted percentages for the test set:")
print(predicted_percentages)

print("Actual percentages for the test set:")
print(y_true_np)

weights = []
biases = []

for name, param in model.named_parameters():
    if 'layer2' in name: 
        if 'weight' in name:
            weights.append(param.data.view(-1).cpu().numpy())
        elif 'bias' in name:
            biases.append(param.data.cpu().numpy())

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.boxplot(weights)
plt.title('Layer 2 Weights Distribution')
plt.xlabel('Neurons')
plt.ylabel('Weight Values')

plt.subplot(1, 2, 2)
plt.boxplot(biases)
plt.title('Layer 2 Biases Distribution')
plt.xlabel('Neurons')
plt.ylabel('Bias Values')

plt.tight_layout()
plt.show()