import torch
import pickle
from Energy_GAT import Energy_GAT, CrystalGAT_with_K
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_loss(loss_hist_train, loss_hist_test, path: str):    
    X = np.linspace(0, len(loss_hist_train), len(loss_hist_train))
    plt.plot(X, loss_hist_train, label='Train-set loss')
    plt.plot(X, loss_hist_test, label='Test-set loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss hystory', fontweight='bold')
    plt.legend()
    plt.savefig(path + r'\loss.png')
    plt.show()

def plot_errors(mae_hist, mse_hist, path: str):    
    X = np.linspace(0, len(mae_hist), len(mae_hist))
    plt.plot(X, mae_hist, label='MAE')
    plt.plot(X, mse_hist, label='MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Errors history', fontweight='bold')
    plt.legend()
    plt.savefig(path + r'\errors.png')
    plt.show()

def train_with_no_K(lr: float, epochs: int, model_filename: str, loss_filename: str):
    # Upload the train dataset
    train_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TrainSet_interpol.pkl'
    with open(train_path, "rb") as file:
        train_set = pickle.load(file)
    
    # Check on the dimensionality
    if train_set[0].x.size(1) is not train_set[67].x.size(1):
        raise ValueError('Something wrong with the number of node features')
    
    sample_batch = train_set[0]
    in_channels = sample_batch.x.size(1)
    # Defining the model
    model = Energy_GAT(hidden_dim=32, num_heads=2, in_channels=in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # TRAINING LOOP
    loss_hist = []
    update = epochs / 100
    print("\n\nTraining the model: ")
    for epoch in range(epochs):
        model.train()
        loss = 0

        for batch in train_set:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch, batch.band_energies, batch.crystal_params)
            loss = criterion(pred, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss = loss.item()

        loss_hist.append(loss)
        if epoch >= update:
            print(f"Trainied at {int(100 * update / epochs)}%")
            update += epochs / 100

    # Save the weights of the model
    model_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainedModels\NO_K'
    torch.save(model.state_dict(), model_path + model_filename + '.pth')

    # Save the loss history
    loss_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\Results\Losses\NO_K'
    with open(loss_path + loss_filename + '.txt', "w") as file:
        for value in loss_hist:
            file.write(f"{value}\n")

    print("Model and loss history saved successfully.")

def train_with_K(lr: float, epochs: int, model_filename: str, loss_filename: str):
    # Upload the train dataset
    train_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TrainSet_K.pkl'
    with open(train_path, "rb") as file:
        train_set = pickle.load(file)
    
    # Check on the dimensionality
    if train_set[0].x.size(1) is not train_set[67].x.size(1):
        raise ValueError('Something wrong with the number of node features')
    
    sample_batch = train_set[0]
    in_channels = sample_batch.x.size(1)
    # Defining the model
    model = CrystalGAT_with_K(hidden_dim=32, num_heads=2, in_channels=in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # TRAINING LOOP
    loss_hist = []
    update = epochs / 100
    print("\n\nTraining the model: ")
    for epoch in range(epochs):
        model.train()
        loss = 0

        for batch in train_set:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch, batch.band_energies, batch.crystal_params, batch.k_points)
            loss = criterion(pred, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss = loss.item()

        loss_hist.append(loss)
        if epoch >= update:
            print(f"Trainied at {int(100 * update / epochs)}%")
            update += epochs / 100

    # Save the weights of the model
    model_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainedModels\WITH_K'
    torch.save(model.state_dict(), model_path + model_filename + '.pth')

    # Save the loss history
    loss_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\Results\Losses\WITH_K'
    with open(loss_path + loss_filename + '.txt', "w") as file:
        for value in loss_hist:
            file.write(f"{value}\n")

    print("Model and loss history saved successfully.")

def train_SC(lr: float, model_filename: str, loss_filename: str):
    # Upload the train dataset
    train_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TrainSet_interpol.pkl'
    with open(train_path, "rb") as file:
        train_set = pickle.load(file)
    
    # Check on the dimensionality
    if train_set[0].x.size(1) is not train_set[67].x.size(1):
        raise ValueError('Something wrong with the number of node features')
    
    sample_batch = train_set[0]
    in_channels = sample_batch.x.size(1)
    # Defining the model
    model = Energy_GAT(hidden_dim=32, num_heads=2, in_channels=in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # TRAINING LOOP
    loss_hist = []
    print("\n\nTraining the model: ")
    epochs = 0
    diff = 1
    resolution = 5*1e-3
    good_loss = 0.01
    while True:
        model.train()

        for batch in train_set:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch, batch.band_energies, batch.crystal_params)
            loss = criterion(pred, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss = loss.item()

        epochs = epochs + 1    
        loss_hist.append(loss)
        if epochs > 1:
            diff = abs(loss_hist[-1] - loss_hist[-2])
             
        print(f'Finished epoch {epochs}')
        
        if loss <= good_loss:
            if diff <= resolution:
                print(f"Convergence criteria achived: resolution = {diff}; final_loss = {loss}")
                break
        if epochs > 5000: 
            print('Maximum limit exceeded. Interrupted.')
            break

    # Save the weights of the model
    model_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainedModels\NO_K'
    torch.save(model.state_dict(), model_path + model_filename + '.pth')

    # Save the loss history
    loss_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\Results\Losses\NO_K'
    with open(loss_path + loss_filename + '.txt', "w") as file:
        for value in loss_hist:
            file.write(f"{value}\n")

    print("Model and loss history saved successfully.")

def train_and_test(hidden_dimentions: int, heads: int, lr: float, epochs: int, model_filename: str):
    # Upload the train and test datasets
    train_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TrainSet_interpol.pkl'
    test_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TestSet_interpol.pkl'
    with open(train_path, "rb") as file:
        train_set = pickle.load(file)
    with open(test_path, "rb") as file:
        test_set = pickle.load(file)
    
    # Defining the model
    sample_batch = train_set[0]
    in_channels = sample_batch.x.size(1)
    model = Energy_GAT(hidden_dim=hidden_dimentions, num_heads=heads, in_channels=in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # TRAINING LOOP
    loss_train_hist = []
    loss_test_hist = []
    mse_hist = []
    mae_hist = []
    update = epochs / 100
    print("\n\nTraining the model: ")
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Train and modify the internal parameters
        model.train()
        
        for batch in train_set:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch, batch.band_energies, batch.crystal_params)
            loss = criterion(pred, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss = loss.item()
        loss_train_hist.append(loss)
        
        # Test the model with the new parameters
        model.eval()
        with torch.no_grad():
            true_values = []
            predictions = []
            for batch in test_set:
                pred = model(batch.x, batch.edge_index, batch.batch, batch.band_energies, batch.crystal_params)
                predictions.append(pred.item())
                true_values.append(batch.y.item())
                loss = criterion(pred, batch.y.unsqueeze(1)).item()
            loss_test_hist.append(loss)
                
            mse = mean_squared_error(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            mse_hist.append(mse)
            mae_hist.append(mae)
        
        if epoch >= update:
            print(f"Trained at {int(100 * update / epochs)}%")
            update += epochs / 100

    # Save the weights of the model
    model_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainedModels\New'
    torch.save(model.state_dict(), model_path + model_filename + '.pth')
    print("Model and loss history saved successfully.")
    
    # Plot
    path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\Results\FIGURES\Model_8'
    plot_errors(mae_hist, mse_hist, path)
    plot_loss(loss_train_hist, loss_test_hist, path)


if __name__ == '__main__':
    # CHOOSE BETWEEN ONE OF THESE TWO AND COMMENT THE OTHER!!!
    # train_SC(1e-5, r'\Model9_SC(lr=1e-5, b=50)', r'\Model9_SC(lr=1e-5, b=50)')
    # train_with_K(1e-6, 3000, r'\Model4_K(ep=3000, lr=1e-6)', r'\Model4_K(ep=3000, lr=1e-6)')
    train_and_test(16, 2, 5*1e-6, 3000, r'\Model_4(16, 2, -7, 3000)')