import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_series(series, plot_name):
    for name, values in series:
        plt.plot(values, label=name)
    plt.xlabel("Steps")
    plt.ylabel(plot_name)
    plt.ylim(0, 100)
    plt.title(plot_name + " plot")
    plt.legend()
    plt.grid(True)
    output_path = plot_name + "_plot.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")
