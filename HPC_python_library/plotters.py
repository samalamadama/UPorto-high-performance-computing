import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

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

    #the first value is null 
    x = data[0, 1:]  
    y = data[1:, 0]
    z = data[1:, 1:]

    # Create figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    
    # Plot initial data
    line, = ax.plot(x, z[512, :], label=f"y = {y[0]}")  # Start with first row
    ax.legend()
    
    # Create a discrete slider
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'y', 0, len(y) - 1, valinit=512)

    # Update function
    def update(val):
        y_index = int(slider.val)  # Get index from slider
        line.set_ydata(z[y_index, :])  # Update plot
        ax.set_ylim(np.min(z[y_index, :]) - 0.1, np.max(z[y_index, :]) + 0.1)
        line.set_label(f"y = {y[y_index]}")  # Update legend
        ax.legend()

        fig.canvas.draw_idle()
    
    # Connect the update function to the slider
    slider.on_changed(update)
    plt.show()

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
