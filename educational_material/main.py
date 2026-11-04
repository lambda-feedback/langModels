# In the root directory:
# -> poetry run python -m educational_material.main --show-plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, argparse
from pathlib import Path
from graphviz import Digraph

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
    csv_path = Path(__file__).parent.parent / "evaluation_function" / "models" / "storage" / "norvig_word_length_frequencies.csv"
    df = pd.read_csv(csv_path)

    df = df.sort_values(by="Percent", ascending=False)
    
    plt.bar(df["WordLength"], df["Percent"], color="skyblue", edgecolor="black")
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

def plot_nn_structure_single(show_plots: bool=False, media_dir: Path=None):
    g = Digraph(); g.attr(rankdir='LR')
    g.attr('node', shape='oval', style='filled', fillcolor='lightblue', color='black', fontname='Helvetica')
    
    g.node('x','x')
    D=16
    for j in range(7):
        d = j if j < 3 else D - 7 + j
        if j == 3:
            g.node(f'h{j}', '\u22EE', fillcolor='white', color='white', fontsize='24')
        else:
            g.node(f'h{j}', f"<<i>v<sub>{d}</sub></i>tanh(<i>w<sub>{d}</sub>x+b<sub>{d}</sub></i>)+<i>c</i>>", fillcolor='lightblue', color='black')
    g.node('o','out')
    for j in range(7):
        color = 'white' if j == 3 else 'black'
        g.edge('x', f'h{j}', color=color)
        g.edge(f'h{j}', 'o', color=color)
        
    out_path = media_dir / "basic_nn_structure_single"
    g.render(out_path, format='svg', view=False)
    g.render(out_path, format='png', view=False)


def plot_nn_structure_multi(show_plots: bool=False, media_dir: Path=None):
    g = Digraph()
    g.attr(rankdir='LR', splines='line', nodesep='0.6', ranksep='0.8', ordering='out')
    g.attr('node', shape='plain', fontname='Helvetica')

    def html_node(name, text, fill="lightblue", border="black", roundness=8):
        return f'''<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" STYLE="rounded" COLOR="{border}" BGCOLOR="{fill}">
                     <TR>
                       <TD PORT="in" WIDTH="8"></TD>
                       <TD BORDER="0" ALIGN="CENTER" FIXEDSIZE="FALSE" CELLPADDING="6">{text}</TD>
                       <TD PORT="out" WIDTH="8"></TD>
                     </TR>
                   </TABLE>>'''

    # Inputs and output
    g.node('x', html_node('x', '<B>x</B>'))
    g.node('y', html_node('y', '<B>y</B>'))
    g.node('o', html_node('o', '<B>out</B>'))

    D = 16
    for j in range(7):
        d = j if j < 3 else D - 7 + j
        if j == 3:
            g.node(f'h{j}', html_node(f'h{j}', '<FONT POINT-SIZE="24">⋮</FONT>', fill='white', border='white'))
        else:
            txt = f"<I>v<sub>{d}</sub></I> tanh(<I>w<sub>{d}</sub>x+u<sub>{d}</sub>y+b<sub>{d}</sub></I>)+<I>c</I>"
            g.node(f'h{j}', html_node(f'h{j}', txt))

    for j in range(7):
        color = 'white' if j == 3 else 'black'
        g.edge('x:out', f'h{j}:in', color=color)
        g.edge('y:out', f'h{j}:in', color=color)
        g.edge(f'h{j}:out', 'o:in', color=color)

    if media_dir is None:
        media_dir = Path('.')
    out_path = media_dir / "basic_nn_structure_multi"
    g.render(out_path, format='svg', view=False)
    g.render(out_path, format='png', view=False)


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
        out_path = media_dir / "basic_nn_plot_bare.png"
        out_path2 = media_dir / "basic_nn_plot_fitted.png"

        plt.scatter(x.cpu(), y.cpu(), s=10, label="Data")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")  # good web resolution
        plt.plot(x_plot.cpu(), y_plot.cpu(), color="red", label="Model")
        plt.legend()
        
        plt.savefig(out_path2, dpi=150, bbox_inches="tight")  # good web resolution
        # Third plot: same as above, but restricted x-limits
        out_path3 = media_dir / "basic_nn_plot_zoomed.png"

        plt.xlim(-2 * torch.pi, 2 * torch.pi)   # <---- restrict x-axis range
        plt.savefig(out_path3, dpi=150, bbox_inches="tight")
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
    #plot_wordlength_histogram(show_plots=args.show_plots, media_dir=media_dir)
    #plot_nn_structure_single(show_plots=args.show_plots, media_dir=media_dir)
    plot_nn_structure_multi(show_plots=args.show_plots, media_dir=media_dir)
    #plot_neural_network_results(show_plots=args.show_plots, media_dir=media_dir)
    