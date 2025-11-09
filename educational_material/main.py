import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, argparse
from pathlib import Path
from evaluation_function.models.basic_nn import TinyNet, f, train_model, MODEL_PATH

def plot_letter_histogram(show_plots: bool=False, media_dir: Path=None):
    """Plot a histogram from norvig_letter_single.csv."""
    csv_path = Path(__file__).parent.parent / "evaluation_function" / "models" / "storage" / "norvig_letter_single.csv"
    df = pd.read_csv(csv_path)

    df = df.sort_values(by="Percent", ascending=False)

    plt.bar(df["Letter"], df["Percent"], color="skyblue", edgecolor="black")
    plt.xlabel("Letter")
    plt.ylabel("Frequency")
    plt.tight_layout()

    out_path = media_dir / "letter_histogram.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        print(f"Plot saved to {out_path}, displaying plot now.")
        plt.show()
    else:
        print(f"Plot saved to {out_path}.")  

def plot_wordlength_histogram(show_plots: bool=False, media_dir: Path=None):
    """Plot a histogram from norvig_word_frequencies.csv."""
    csv_path = Path(__file__).parent.parent / "evaluation_function" / "models" / "storage" / "norvig_word_frequencies.csv"
    df = pd.read_csv(csv_path)

    df = df.sort_values(by="Percent", ascending=False)

    plt.bar(df["wordLength"], df["Percent"], color="skyblue", edgecolor="black")
    plt.xlabel("Word length")
    plt.ylabel("Frequency")
    plt.tight_layout()

    out_path = media_dir / "word_histogram.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        print(f"Plot saved to {out_path}, displaying plot now.")
        plt.show()
    else:
        print(f"Plot saved to {out_path}.")  

def plot_neural_network_results(show_plots: bool=False, media_dir: Path=None):
    """Plot the results of a neural network model against the data.

    Args:
        x (torch.Tensor): Input data.
        y (torch.Tensor): Target data.
        model (torch.nn.Module): Trained neural network model.
    """
    # Load trained model (or train if needed)
    model = TinyNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Recreate training data for plotting
    x = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(1).to(device)
    y = (f(x) + 0.1*torch.randn_like(x)).to(device)

    with torch.no_grad():
        # Make domain twice as wide as training range
        x_plot = torch.linspace(2*x.min().item(), 2*x.max().item(), 800, device=x.device).unsqueeze(1)
        y_plot = model(x_plot)

        plt.scatter(x.cpu(), y.cpu(), s=10, label="Data")
        plt.plot(x_plot.cpu(), y_plot.cpu(), color="red", label="Model")
        plt.legend()
        out_path = media_dir / "basic_nn_plot.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")  # good web resolution
        if show_plots:
            print(f"Plot saved to {out_path}, displaying plot now.")
            plt.show()
        else:
            print(f"Plot saved to {out_path}.")  


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively instead of just saving them."
    )
    args = parser.parse_args()
    media_dir = Path(__file__).parent / "media"
    media_dir.mkdir(exist_ok=True)
    #plot_letter_histogram(show_plots=args.show_plots, media_dir=media_dir)
    plot_wordlength_histogram(show_plots=args.show_plots, media_dir=media_dir)
    #plot_neural_network_results(show_plots=args.show_plots, media_dir=media_dir)
    