import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class Energy_GAT(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, in_channels):
        super(Energy_GAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Definizione delle GAT layers
        self.gat1 = GATConv(in_channels=in_channels, out_channels=hidden_dim, heads=num_heads)
        self.gat2 = GATConv(in_channels=hidden_dim * num_heads, out_channels=hidden_dim, heads=1)

        # Encoder per energie di banda
        self.band_energy_encoder = torch.nn.Linear(100, hidden_dim)  # Modifica la dimensione in base ai tuoi dati
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(hidden_dim + hidden_dim + 13, hidden_dim)  # +6 per i parametri del cristallo
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # Output finale: correzione gap

    def forward(self, x, edge_index, batch, band_energies, crystal_params):
        # GAT layers
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Encode band energies
        band_energy_latent = self.band_energy_encoder(band_energies)

        # Espandiamo le energie di banda per il batch
        if band_energy_latent.dim() == 1:
            band_energy_latent = band_energy_latent.unsqueeze(0)  # Rendi bidimensionale se necessario
        band_energy_latent = band_energy_latent.expand(x.size(0), -1)

        # Espandiamo crystal_params per il batch
        crystal_params = crystal_params.unsqueeze(0).expand(x.size(0), -1)

        # Concatenazione con energie di banda e parametri del cristallo
        x = torch.cat([x, band_energy_latent, crystal_params], dim=1)

        # Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x

# Vsc_1Vcs_2
# e3nn