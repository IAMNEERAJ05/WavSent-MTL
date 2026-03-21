# WavSent-MTL — Architecture Specification

## MTL Model (identical wrapper for all 4 encoders)

Input [batch, 5, n_features]
           ↓
    ┌──────────────────┐
    │     Encoder      │ ← TKAN / LSTM / GRU / TCN
    │ (hidden_size=hs) │   from random search
    └──────────────────┘
           ↓
    [batch, hidden_size]
       ↓              ↓
 ┌──────────┐   ┌──────────┐
 │ Reg Head │   │ Clf Head │
 ├──────────┤   ├──────────┤
 │Linear    │   │Linear    │
 │(hs, 16)  │   │(hs, 16)  │
 │ReLU      │   │ReLU      │
 │Linear    │   │Linear    │
 │(16, 1)   │   │(16, 1)   │
 │(linear)  │   │Sigmoid   │
 └──────────┘   └──────────┘
       ↓               ↓
 return magnitude   P(up)∈[0,1]

## PyTorch Implementation

class MTLModel(nn.Module):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.log_sigma1 = nn.Parameter(torch.zeros(1))
        self.log_sigma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.encoder(x)
        return self.reg_head(h), self.clf_head(h)

## Uncertainty Weighted Loss

def uncertainty_weighted_loss(mse, bce, log_s1, log_s2):
    return (torch.exp(-log_s1) * mse + log_s1 +
            torch.exp(-log_s2) * bce + log_s2)

Reference: Kendall et al. CVPR 2018

## Encoder Specifications

### LSTMEncoder
- nn.LSTM(input_size=n_feat, hidden_size=hs,
         num_layers=nl, batch_first=True,
         dropout=dr if nl>1 else 0)
- Return: h_n[-1] → [batch, hs]

### GRUEncoder
- nn.GRU(input_size=n_feat, hidden_size=hs,
        num_layers=nl, batch_first=True,
        dropout=dr if nl>1 else 0)
- Return: h_n[-1] → [batch, hs]

### TCNEncoder
- Causal dilated residual blocks
- num_channels=[hs]*num_levels
- kernel_size=ks, dropout=dr
- Return: output[:, -1, :] → [batch, hs]

### TKANEncoder
- From github.com/remigenet/TKAN
- hidden_size=hs, dropout=dr
- spline_order=3 (fixed)
- Return: final output → [batch, hs]

## PSO Ensemble Pipeline

Training phase:
  Each model trained independently (30 seeds)
  Best seed val predictions saved per model:
    tkan_val_probs   → [n_val,]
    lstm_val_probs   → [n_val,]
    gru_val_probs    → [n_val,]
    tcn_val_probs    → [n_val,]

PSO weight search (on val predictions only):
  w = softmax([w1, w2, w3, w4])
  ensemble_val = w1*tkan + w2*lstm + w3*gru + w4*tcn
  fitness = -accuracy(ensemble_val > 0.5, val_labels)
  n_particles=20, iterations=50

Test phase:
  Apply learned w to test predictions → ensemble result
  Store individual test metrics BEFORE ensemble

## Ablation Config Summary

| Config | Input | Encoder | Runs |
|--------|-------|---------|------|
| A | returns+polarity_mean | TKAN | 30 |
| B | denoised OHLCV+polarity_mean | TKAN | 30 |
| C | denoised technicals+polarity_mean | TKAN | 30 |
| D | BEST_REPR | LSTM | 30 |
| E | BEST_REPR | GRU | 30 |
| F | BEST_REPR | TCN | 30 |
| G | PSO ensemble | All 4 | 1 |

BEST_REPR = winner of B vs C on val set
Config G reuses saved predictions from C, D, E, F
(best seed of each)

## Hyperparameter Search Spaces

| Param | Values | Models |
|-------|--------|--------|
| hidden_size | 32,64,128 | All |
| dropout | 0.1,0.2,0.3 | All |
| learning_rate | 1e-3,5e-4,1e-4 | All |
| batch_size | 16,32,64 | All |
| num_layers | 1,2 | LSTM,GRU |
| kernel_size | 2,3 | TCN |
| num_levels | 2,3 | TCN |

Random search: 40 trials per model
Metric: minimize val_loss
Done once on Kotekar, applied to Kaggle

## Output Shapes

Input:              [batch, 5, n_features]
After encoder:      [batch, hidden_size]
Regression output:  [batch, 1]
Classification out: [batch, 1]