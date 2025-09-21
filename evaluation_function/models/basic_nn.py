"""
A simple feedforward neural network in PyTorch to illustrate 
the basic features of a neural network.

Dev only:
- Data: add random noise to a time series
- Model: a tiny neural network with one hidden layer, using PyTorch nn.Module
- Training setup: mean squared error loss and Adam optimizer
- Training loop: runs for a fixed number of epochs, printing loss occasionally
- Save the model to disk after training

Production:

- Load the trained model
- Test the model for the argument given by the student (infer value and compare to underlying 'True' function)

"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from lf_toolkit.evaluation import Result, Params

from pathlib import Path
import os

# Setup paths for saving/loading model and data
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "storage"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "basic_nn.pt"

def f(x):
    """Target function with noise (sine wave)."""
    return torch.sin(x)

def x_on_model(v, dev):
    """ Helper: put scalar value on same device as model. """
    return torch.tensor([[v]], device=dev, dtype=torch.float32)

class TinyNet(nn.Module):
    """A tiny feedforward neural network."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 16)
        self.act = nn.Tanh()
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        return self.out(self.act(self.hidden(x)))

def train_model(device):
    torch.manual_seed(0)
    x = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(1).to(device)
    y = (f(x) + 0.1*torch.randn_like(x)).to(device)

    model = TinyNet().to(device)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(2000):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 400 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

    return model

def run(response, answer, params: Params) -> Result:
    print("GPU") if torch.backends.mps.is_available() else print("CPU")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    refresh = params.get("refresh", False)
    if refresh:
        model = train_model(device)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)

    else:
        model = TinyNet().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

    with torch.no_grad():
        # For now just test one point
        x_val = x_on_model(float(response), device)
        y_pred = model(x_val).cpu().item()

    absolute_tolerance = params.get("absolute_tolerance", 0.1)
    y_true = f(torch.tensor([[float(response)]])).item()
    diff = abs(y_pred - y_true)
    is_correct=diff < absolute_tolerance
    return Result(is_correct=is_correct,feedback_items=[("general",f"Model({response}) = {y_pred:.4f}, f({response}) = {y_true:.4f} (this is the 'true' value), Diff = {diff:.4f} (tolerance {absolute_tolerance}). Valid model: {is_correct}")])

# --- runnable code only executes if script is run directly ---

if __name__ == "__main__":

    result = run("some_response", "some_answer", Params())
    print(result)

"""     # 5. Plot results (eval mode, extended domain)
    with torch.no_grad():
        # Make domain twice as wide as training range
        x_plot = torch.linspace(2*x.min().item(), 2*x.max().item(), 800, device=x.device).unsqueeze(1)
        y_plot = model(x_plot)

        plt.scatter(x.cpu(), y.cpu(), s=10, label="Data")
        plt.plot(x_plot.cpu(), y_plot.cpu(), color="red", label="Model")
        plt.legend()
        plt.show() """
