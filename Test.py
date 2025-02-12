from Energy_GAT import Energy_GAT
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_residual_histo(residuals_train, residuals_test, bins=40):
    residuals_train = np.delete(residuals_train, np.argmax(residuals_train))
    
    bins_train = np.linspace(min(residuals_train), max(residuals_train), bins)
    bins_test = np.linspace(min(residuals_test), max(residuals_test), int(bins/2))
    
    # Creazione dell'istogramma con trasparenza per la sovrapposizione
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(residuals_train, bins=bins_train, alpha=0.6, color='lime', edgecolor='gray', label='Train set')
    ax[1].hist(residuals_test, bins=bins_test, alpha=0.6, color='orange', edgecolor='gray', label='Test set')
    
    # Titoli e legende
    ax[0].set_xlabel("Residuals")
    ax[0].set_ylabel("Entries")
    ax[0].set_title("Residuals histogram (train set)", fontweight='bold')
    ax[1].set_xlabel("Residuals")
    ax[1].set_ylabel("Entries")
    ax[1].set_title("Residuals histogram (test set)", fontweight='bold')

def plot_parity_predicion(predictions, true_values, predictions_train, true_values_train, mse, mae):
    plt.title(f'Parity plot', fontweight='bold')
    plt.scatter(true_values, predictions, label='Predicted vs True', s=15)
    plt.scatter(true_values_train, predictions_train, c='orange', label='Predicted vs True (train set)', s=2)
    plt.plot([min(true_values_train), max(true_values_train)], [min(true_values_train), max(true_values_train)], 'r--', label='Perfect Prediction')
    plt.xlabel('True GW correction (eV)')
    plt.ylabel('Predicted GW correction (eV)')
    plt.plot([], [], label=f'Mean Squared Error = {mse}', c='white')
    plt.plot([], [], label=f'Mean Absolute Error = {mae}', c='white')
    plt.ylim(top=6)
    plt.xlim(left=0, right=6)
    plt.legend()

def plot_loss(loss_path):
    loss = []
    with open(loss_path, "r") as file:
        for value in file:
            loss.append(float(value.strip()))
    
    X = np.linspace(0, len(loss), len(loss))
    plt.plot(X, loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss per cicle')
    plt.title(f'Loss hystory', fontweight='bold')

if __name__ == '__main__':
    # Train model's parameters
    hidden_dim = 32
    num_heads = 2
    in_channels = 4
    model = Energy_GAT(hidden_dim=hidden_dim, num_heads=num_heads, in_channels=in_channels)

    # Upload the trained model weights
    model_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainedModels\NO_K'
    filename = r'\Model8_SC(lr=1e-5, b=50).pth' # PUT THE NAME HERE
    model.load_state_dict(torch.load(model_path + filename))

    # Eval mode
    model.eval()
    print("Modello caricato con successo!")
    
    # Upload the train and test datasets
    path_train = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TrainSet_interpol.pkl'
    path_tests = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TestSet_interpol.pkl'
    with open(path_train, "rb") as file:
        train_set = pickle.load(file)
    with open(path_tests, "rb") as file:
        test_set = pickle.load(file)

    # Memorizing the results
    predictions = []
    true_values = []
    totally_wrong = 0
    
    predictions_train = []
    true_values_train = []

    with torch.no_grad():
        for batch in test_set:
            pred = model(batch.x, batch.edge_index, batch.batch, batch.band_energies, batch.crystal_params)
            is_wrong = abs(pred.item() - batch.y.item()) > 4
            if not is_wrong and batch.y.item() < 10:
                predictions.append(pred.item())
                true_values.append(batch.y.item())
            else: totally_wrong = totally_wrong + 1
            
        for batch in train_set:
            pred = model(batch.x, batch.edge_index, batch.batch, batch.band_energies, batch.crystal_params)
            predictions_train.append(pred.item())
            true_values_train.append(batch.y.item())
        
    # Calculate the errors
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f'Totally wrong predictions (eliminated): {totally_wrong}')
    
    plot_parity_predicion(predictions, true_values, predictions_train, true_values_train, mse, mae)
    plt.show()
    
    residuls_train = np.array(true_values_train) - np.array(predictions_train)
    residuals_test = np.array(true_values) - np.array(predictions)
    plot_residual_histo(residuls_train, residuals_test)
    plt.show()
    
    loss_path = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\Results\Losses\NO_K'
    loss_name = r'\Model8_SC(lr=1e-5, b=50).txt' # PUT THE NAME HERE
    plot_loss(loss_path + loss_name)
    plt.show()
    
