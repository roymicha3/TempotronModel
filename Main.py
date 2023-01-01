from plots.GraphPlot import GraphPlot

def main():
    
    plotter = GraphPlot(T=500, time_step=1)
    plotter.plot_perceptron_tempotron_comparison()
        
if __name__ == '__main__':
    main()