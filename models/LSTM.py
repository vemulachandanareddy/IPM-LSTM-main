import torch
import torch.nn as nn
from models.selfattention import SelfAttention  

"""
Model inputs: optimizees
Model outputs: [X_1, X_2, ..., X_T]
This version of the model now uses two stacked LSTM layers.
"""
class LSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 iter_step,
                 device,
                 num_layers=2):
        # input_dim: the dimension of model inputs, for example, [X_t, ▽f(X_t)]
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.iter_step = iter_step
        self.device = device
        self.num_layers = num_layers  

       
        self.attention = SelfAttention(embed_dim=input_dim, num_heads=1).to(self.device)

        # For layer 1, the input comes directly from the attention output
        self.W_i_1 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim), device=self.device))
        self.U_i_1 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.b_i_1 = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=torch.float32))

        self.W_f_1 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim), device=self.device))
        self.U_f_1 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.b_f_1 = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=torch.float32))

        self.W_o_1 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim), device=self.device))
        self.U_o_1 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.b_o_1 = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=torch.float32))

        self.W_u_1 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim), device=self.device))
        self.U_u_1 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.b_u_1 = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=torch.float32))
        self.dropout = nn.Dropout(p=0.2)

       
        # The input to layer 2 now has dimension `hidden_dim`.
        self.W_i_2 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.U_i_2 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.b_i_2 = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=torch.float32))

        self.W_f_2 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.U_f_2 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.b_f_2 = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=torch.float32))

        self.W_o_2 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.U_o_2 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.b_o_2 = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=torch.float32))

        self.W_u_2 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.U_u_2 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device))
        self.b_u_2 = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=torch.float32))
        
        self.W_h = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, 1), device=self.device))
        self.b_h = nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.float32))
        self.dropout = nn.Dropout(p=0.2)

    def name(self):
        return 'stacked_lstm'

    def forward(self, data, y, J, F, states=((None, None), (None, None))):
        
        batch_size, num_var = y.shape[0], y.shape[1]

        # Initialize layer 1 states if not provided.
        if states[0][0] is None:
            H1 = torch.zeros((batch_size, num_var, self.hidden_dim), device=self.device)
            C1 = torch.zeros((batch_size, num_var, self.hidden_dim), device=self.device)
        else:
            H1, C1 = states[0]

        # Initialize layer 2 states if not provided.
        if states[1][0] is None:
            H2 = torch.zeros((batch_size, num_var, self.hidden_dim), device=self.device)
            C2 = torch.zeros((batch_size, num_var, self.hidden_dim), device=self.device)
        else:
            H2, C2 = states[1]

        final_y = 0.0
        final_loss = 0.0
        losses = []
        loss0 = 10000.0

        for iter in range(self.iter_step):
            # Compute the gradient from the original data function.
            grad_init = data.sub_smooth_grad(y, J, F)
            # Concatenate current solution and gradient along the last dimension.
            inputs = torch.concat((y, grad_init), dim=-1)  # [batch_size, num_var, input_dim]
            # Apply self-attention on the inputs.
            attn_inputs, _ = self.attention(inputs)         # shape remains [batch_size, num_var, input_dim]

            # ----- Layer 1 LSTM computations -----
            I1 = torch.sigmoid(attn_inputs @ self.W_i_1 + H1 @ self.U_i_1 + self.b_i_1)
            F1 = torch.sigmoid(attn_inputs @ self.W_f_1 + H1 @ self.U_f_1 + self.b_f_1)
            O1 = torch.sigmoid(attn_inputs @ self.W_o_1 + H1 @ self.U_o_1 + self.b_o_1)
            U1 = torch.tanh(attn_inputs @ self.W_u_1 + H1 @ self.U_u_1 + self.b_u_1)
            C1 = I1 * U1 + F1 * C1
            H1 = O1 * torch.tanh(C1)

            # ----- Layer 2 LSTM computations -----
            # Layer 2 takes the hidden state output from layer 1 as input.
            I2 = torch.sigmoid(H1 @ self.W_i_2 + H2 @ self.U_i_2 + self.b_i_2)
            F2 = torch.sigmoid(H1 @ self.W_f_2 + H2 @ self.U_f_2 + self.b_f_2)
            O2 = torch.sigmoid(H1 @ self.W_o_2 + H2 @ self.U_o_2 + self.b_o_2)
            U2 = torch.tanh(H1 @ self.W_u_2 + H2 @ self.U_u_2 + self.b_u_2)
            C2 = I2 * U2 + F2 * C2
            H2 = O2 * torch.tanh(C2)

            # ----- Final gradient computation -----
            grad = H2 @ self.W_h + self.b_h
            y = y - grad

            loss = (data.sub_objective(y, J, F).mean()) / self.iter_step
            final_loss += loss
            losses.append(loss.detach().cpu())
            if (loss < loss0) or (iter == 0):
                loss0 = loss
                final_y = y

        return final_y, final_loss, losses