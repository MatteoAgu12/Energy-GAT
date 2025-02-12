import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import pickle
from pymatgen.core import Structure
from Dict_to_Graph import dict_to_PyTorch_graph
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Energy_GAT import CrystalGAT

# Download the DB
with open(r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\DB.pkl', "rb") as file:
    DB = pickle.load(file)
    
graphs = []
for mat in DB:
    graph = dict_to_PyTorch_graph(mat)
    if graph is not None: graphs.append(graph)

# Ricrea l'istanza del modello con gli stessi iperparametri
model = CrystalGAT(hidden_dim=64, num_heads=4)

# Carica i parametri salvati
model.load_state_dict(torch.load(r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\Results\trained_model.pth'))
model.eval()  # Imposta il modello in modalità valutazione
print("Modello caricato con successo.")


# Lista per memorizzare i risultati
predictions = []
true_values = []

with torch.no_grad():  # Disabilita il calcolo dei gradienti per velocizzare
    for batch in graphs:
        pred = model(batch)  # Previsione del modello
        predictions.append(pred.item())  # Converti in valore scalare
        true_values.append(batch.y.item())  # Valore reale del gap
        
# Calcola metriche di valutazione
mse = mean_squared_error(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Loss history
T = np.linspace(1, epochs, epochs)
plt.title(f'Loss history with learning rate of {lr}')
plt.xlabel('Epochs')
plt.ylabel('Loss function')
plt.plot(T, loss_hist, label=f"Final loss {loss_hist[-1]}")
plt.legend()
plt.show()

# Confronta previsioni e valori reali
plt.title(f'Parity plot with learning rate of {lr}')
plt.scatter(true_values, predictions, label='Predicted vs True')
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--', label='Perfect Prediction')
plt.xlabel('True GW correction')
plt.ylabel('Predicted GW correction')
plt.legend()
plt.show()