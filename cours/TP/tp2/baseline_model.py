"""
Modèle Baseline : Oracle de la Guilde
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ============================================================================
# TP2 : Modèle MLP pour stats d'aventuriers
# ============================================================================

class GuildOracle(nn.Module):
    """
    Modèle pour prédire la survie des aventuriers.

    Architecture : MLP avec BatchNorm pour meilleure généralisation  
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 8, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        if num_layers == 0:
            self.network = nn.Linear(input_dim, 1)
        else:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                
            layers.append(nn.Linear(hidden_dim, 1))
            self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# TP3 : Modèle Séquentiel
# ============================================================================

class DungeonOracle(nn.Module):
    """
    Version Hybride Robuste : LSTM + Max Pooling.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 2,
            hidden_dim: int = 258,
            num_layers: int = 1,
            dropout: float = 0.0,
            mode: str = "linear",
            bidirectional: bool = False,
            padding_idx: int = 0,
            max_length: int = 140
            ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.mode = mode.lower().strip()
        self.max_length = max_length

        self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=padding_idx
                )

        # --- BRANCHE LSTM ---
        if self.mode != "linear":
            rnn_class = nn.LSTM if self.mode == "lstm" else nn.RNN
            self.rnn = rnn_class(
                    input_size=embed_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                    )
            
            rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            
            # 2. On concatène [Last_Hidden] + [Max_Pool]
            # Donc l'entrée du classifier est D + D = 2 * D
            classifier_input_dim = rnn_output_dim * 2
            
            self.classifier = nn.Sequential(
                    nn.Linear(classifier_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
            )
        
        self.solo_embeddings = nn.Sequential(
                nn.Flatten(),
                nn.Linear(max_length * embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
                )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x shape: (Batch, 140)
        embedded = self.embedding(x)

        if self.mode != "linear":
            # 1. PACKING (Gestion intelligente du padding)
            if lengths is not None:
                packed_embedded = pack_padded_sequence(
                    embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                
                if self.mode == "lstm":
                    packed_output, (hidden, cell) = self.rnn(packed_embedded)
                else:
                    packed_output, hidden = self.rnn(packed_embedded)

                rnn_output, _ = pad_packed_sequence(
                    packed_output, 
                    batch_first=True, 
                    total_length=x.size(1) 
                )
            else:
                # Fallback sans longueurs
                if self.mode == "lstm":
                    rnn_output, (hidden, cell) = self.rnn(embedded)
                else:
                    rnn_output, hidden = self.rnn(embedded)

            if self.bidirectional:
                last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                last_hidden = hidden[-1]

            if lengths is not None:
                mask = (x != 0).unsqueeze(2)
                # On remplit le padding avec -infini pour que le Max ne le choisisse jamais
                rnn_output = rnn_output.masked_fill(~mask, -1e9)

            # Max sur la dimension du temps (dim=1)
            max_pool, _ = torch.max(rnn_output, dim=1)

            # 3. FUSION
            # On combine les deux intelligences
            combined = torch.cat([last_hidden, max_pool], dim=1)

            # 4. CLASSIFICATION
            logits = self.classifier(combined)
            return logits
        else:
            return self.solo_embeddings(embedded)

    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x, lengths)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        proba = self.predict_proba(x, lengths)
        return (proba > 0.5).float()

    def get_embeddings(self) -> torch.Tensor:
        return self.embedding.weight.detach().clone()


# ============================================================================
# Fonctions utilitaires globales
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: nn.Module):
    print("=" * 50)
    print("Résumé du modèle")
    print("=" * 50)
    print(model)
    print("-" * 50)
    print(f"Nombre de paramètres : {count_parameters(model):,}")