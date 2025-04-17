import os
import torch
from models.LSTM import LSTM

def main():
    input_dim = 2
    hidden_dim = 50
    iter_step = 50
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    orig_checkpoint_path = "./results/stacked_lstm/params/QP_RHS_100_50_50_100_50.pth"
    quantized_checkpoint_path = "./results/stacked_lstm/params/quantized_QP_RHS_100_50_50_100_50.pth"
    model = LSTM(input_dim, hidden_dim, iter_step, device)
    model.to(device)
    if not os.path.exists(orig_checkpoint_path):
        print(f"Error: Original checkpoint not found: {orig_checkpoint_path}")
        return
    state_dict = torch.load(orig_checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Original model loaded successfully.")
    model.half()
    print("Model converted to 16-bit floating point.")
    torch.save(model.state_dict(), quantized_checkpoint_path)
    print(f"Quantized model saved at: {quantized_checkpoint_path}")

if __name__ == "__main__":
    main()