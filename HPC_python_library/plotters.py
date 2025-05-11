import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ipywidgets import interact, IntSlider
import ipywidgets as widgets
import os
import re
from collections import defaultdict 
from matplotlib.tri import Triangulation


def plotter_not_CVS(filename):
    data = np.loadtxt(filename)
    x=data[:, 0]
    y=data[:, 1]
    
    plt.plot(x, y, color='b')
    title = filename + " plot"
    plt.title(title)
    
    plt.show()

def plotter(filename):
    data = np.genfromtxt(filename, delimiter=",")
    x=data[:, 0]
    y=data[:, 1]
    
    plt.plot(x, y, color='b')
    title = filename + " plot"
    plt.title(title)
    
    plt.show()


def plotter_3D(filename):
    data = np.loadtxt(filename)
    title = filename + " plot"

    #the first value is null 
    x = data[0, 1:]  
    y = data[1:, 0]
    z = data[1:, 1:]

    X, Y = np.meshgrid(x,y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, z, cmap='viridis', edgecolor='green')
    ax.set_title(title)
    plt.show()

def plotter_slider(filename):
    data = np.loadtxt(filename)
    title = filename + " plot"

    # Extract x, y, z
    x = data[0, 1:]
    y = data[1:, 0]
    z = data[1:, 1:]

    # Interactive update function
    def update(y_index):
        plt.figure(figsize=(8, 4))
        plt.plot(x, z[y_index, :], label=f"y = {y[y_index]}")
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.ylim(np.min(z[y_index, :]) - 0.1, np.max(z[y_index, :]) + 0.1)
        plt.legend()
        plt.grid(True)
        plt.show()

    # Use ipywidgets to create an interactive slider
    interact(update, y_index=IntSlider(min=0, max=len(y)-1, step=1, value=512, description='y index'))


def plotter_colormap_not_CSV(filename):
    data = np.loadtxt(filename)
    title = filename + " plot"

    #the first value is null 
    x = data[0, 1:]  
    y = data[1:, 0]
    z = data[1:, 1:]

    colormap = plt.imshow(z, aspect='auto', origin='lower', cmap='viridis', extent=[x[0], x[-1], y[0], y[-1]])
    plt.show()

def plotter_colormap(filename):
    data = np.genfromtxt(filename, delimiter=",")
    title = filename + " plot"

    #the first value is null 
    x = data[0, 1:]  
    y = data[1:, 0]
    z = data[1:, 1:]

    colormap = plt.imshow(z, aspect='auto', origin='lower', cmap='viridis', extent=[x[0], x[-1], y[0], y[-1]])
    plt.show()

def parse_time(time_str):
    """Parse a time string like '123ns' or '1.2ms' and return time in float and unit."""
    match = re.match(r"([\d.]+)([a-zA-Z]+)", time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    return float(match.group(1)), match.group(2)

    
def benchmark_plotter(path, name, *benchmarks):
    benchmark_data = defaultdict(lambda: defaultdict(list))  # {benchmark_name: {unit: [(threads, time)]}}
    pattern = re.compile(rf"{re.escape(name)}_(\d+)_threads$")

    for file in os.listdir(path):
        match = pattern.match(file)
        if match:
            num_threads = int(match.group(1))
            with open(os.path.join(path, file), 'r') as f:
                for line in f:
                    if ',' not in line:
                        continue
                    bname, time_str = line.strip().split(',', 1)
                    if benchmarks and bname not in benchmarks:
                        continue
                    time_val, unit = parse_time(time_str)
                    benchmark_data[bname][unit].append((num_threads, 1./time_val))

    if not benchmark_data:
        print("No matching data found.")
        return

    plt.figure()
    for bname, unit_data in benchmark_data.items():
        for unit, data_points in unit_data.items():
            data_points.sort()  # sort by number of threads
            threads, times = zip(*data_points)
            plt.plot(threads, times, label=f"{bname} ({unit})")
            y_unit = unit  # use last unit found (assuming same unit per benchmark)

    plt.title(name) 
    plt.xlabel("Number of Threads")
    plt.ylabel(f"1/Time ({y_unit})^-1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_graphene_ldos(filename, Nx, Ny, a=1.0):
    """
    Plot interactive LDOS colormap over a 2D graphene lattice.
    
    Parameters:
    - filename: path to the LDOS data file
    - Nx, Ny: number of unit cells in x and y directions
    - a: lattice constant (default = 1.0)
    """
    # Load data
    data = np.genfromtxt(filename, delimiter=",")
    energies = data[1:, 0]  # energy axis
    site_indices = data[0, 1:].astype(int)
    ldos = data[1:, 1:]

    def index_to_pos(index):
        """Convert unwrapped index to (x, y) position on hex lattice."""
        base = index // 2
        typ = index % 2
        x = base // Ny
        y = base % Ny
        dx = a * 3/2 * x
        dy = a * np.sqrt(3) * (y + 0.5 * (x % 2))
        if typ == 0:
            return dx, dy
        else:
            return dx + a * 0.5, dy + a * np.sqrt(3)/2

    positions = np.array([index_to_pos(i) for i in site_indices])
    x_pos, y_pos = positions[:, 0], positions[:, 1]

    def plot_ldos_at_index(idx):
        e = energies[idx]
        plt.figure(figsize=(12, 6))
        sc = plt.scatter(x_pos, y_pos, c=ldos[idx, :], cmap='viridis', s=50, edgecolors='k')
        plt.colorbar(sc, label=f'Local density of states')
        plt.title(f"Graphene LDOS (E = {e:.3f})")
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    # Use integer slider to select exact column
    interact(plot_ldos_at_index,
             idx=IntSlider(value=len(energies)//2, min=0, max=len(energies)-1, step=1, description='Energy index'))

def scatter_matrix(filename):
    data = np.genfromtxt(filename, delimiter=",")
    x_values = data[0, 1:]
    y_all_values = data[1:, 1:]

    for i, x in enumerate(x_values):
        y_values = y_all_values[:, i]
        x_values = np.full_like(y_values, x)  # Repeat x for all y values
        plt.scatter(x_values, y_values)

    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.show()
