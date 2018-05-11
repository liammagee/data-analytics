# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import savReaderWriter
import seaborn as sns




class DataSet:
    def __init__(self, filename = None):
        self.data = {}
        self.metadata = {}
        if filename != None:
            self.load_spss_files(filename)
            self.assign_columns()

    # Functions
    def load_spss_files(self, filename):
        with savReaderWriter.SavHeaderReader(filename) as header:
            self.metadata = header.all()
        with savReaderWriter.SavReader(filename) as reader:
            self.data = pd.DataFrame(list(reader))

    def assign_columns(self):
        self.data.columns = [x.decode('utf-8') for x in self.metadata.varNames]

    def get_var_label(self, var):
        key = bytes(var, encoding='utf-8')
        if key in self.metadata.varLabels:
            return self.metadata.varLabels[bytes(var, encoding='utf-8')].decode('windows-1252')
        else:
            return None

    def get_value_labels(self, var):
        key = bytes(var, encoding='windows-1252')
        if key in self.metadata.valueLabels:
            return dict([(k, v.decode('windows-1252')) for k, v in self.metadata.valueLabels[key].items()])
        else:
            return None

    def freq_table(self, var, weights = True, weight_col = 'WEIGHT'):
        value_labels = self.get_value_labels(var)
        var_label = self.get_var_label(var)
        if value_labels != None:
            return pd.DataFrame(list(self.get_value_labels(var).items()), columns=['index', self.get_var_label(var)]) \
                    .set_index('index', drop=True) \
                    .merge(pd.DataFrame({ 'Freq.': self.data[var].value_counts().sort_index(),
                                          'Freq. Rel.': np.round(100. * self.data[var].value_counts().sort_index() / np.size(self.data[var]), 1),
                                          'Freq. Weighted': self.data.groupby(var).apply(lambda x: np.sum(x[weight_col])),
                                          'Freq. Weighted Rel.': np.round(100. * self.data.groupby(var).apply(lambda x: np.sum(x[weight_col])) / np.sum(self.data[weight_col]), 1),
                                        }), \
                           left_index = True, right_index = True)
        else:
            return pd.DataFrame(self.data[var].value_counts().sort_index())


    def gen_histogram(self, cols, stacked = True, legend_labels = None, normalise = False, use_weights = True):
        import tabulate
        from IPython.display import HTML, display
        
        # For cases where weights are included, get the first column along
        col = cols[0]
        
        # Print the variable name
        display(HTML("<strong>"+self.get_var_label(col)+"<strong>"))
        display(HTML("<br/>"))

        # Set up the plot
        fig, ax = plt.subplots()
        labels = self.get_value_labels(col).values()
        bin_length = len(labels)
        
        # If legend labels exist, split the data by the first index value (assumes len(legend_labels) == len(self.data.index.levels))
        if legend_labels != None and hasattr(self.data.index, 'levels') and len(legend_labels) == len(self.data.index.levels):
            d = pd.DataFrame(self.data[cols].dropna(), index = self.data.index).unstack(level=0).as_matrix().T
            d = np.array([x[~np.isnan(x)] for x in d])
        else:
            d = self.data[cols].dropna()
            
        values, weights = np.split(d, 2)
        
        if use_weights == False:
            weights = [np.ones(len(x)) for x in weights]
        if stacked == False and normalise == True:
            hists = [np.histogram(x, np.arange(1, bin_length + 2) - 0.5, weights = weights[i]) for i, x in enumerate(values)]
            hists_norm = [(100. * x / np.sum(x), y) for (x, y) in hists]
            offset = 1. / (len(hists_norm) + 1)
            [ax.bar(x[1][:-1] + offset * i, x[0], width = 0.3, label = legend_labels[i]) for i, x in enumerate(hists_norm)];
            bins = np.array(hists)[0, 1]
            n = [np.round(x[0], 1) for x in hists_norm] # np.array(hists)[:, 0]
        else:
            n, bins, patches = ax.hist(values, weights = weights, bins = np.arange(1, bin_length + 2) - 0.5, rwidth = 0.8, stacked=stacked, label = legend_labels)

        # Add grid
        ax.grid(True)
        
        # Legend
        if legend_labels != None:
            ax.legend(prop={'size': 10})

        # Apply colours
        cm = plt.get_cmap('Blues')
        #for c, patch in zip(bns, patches):
        #    patch.set_facecolor(cm(c))

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # scale values to interval [0,1]
        bns = bin_centers - min(bin_centers)
        bns /= max(bns)

        # rescale values to interval [0.2,0.8]
        scale_inc = 0.2
        bns *= (1.0 - scale_inc * 2.)
        bns += scale_inc

        # Add title and axes
        # plt.title(self.get_var_label(col), loc='left')
        ax.set_xticks(np.arange(1, bin_length + 1))
        ax.set_xticklabels(self.get_value_labels(col).values(), rotation=45, ha='right')
        plt.xlabel('Level')
        if normalise == True:
            plt.ylabel('% of responses')
        else:
            plt.ylabel('No. of responses')

        # Tweak spacing to prevent clipping of ylabel
    #     fig.tight_layout()
        plt.show()
        
        # Generate the frequency table
        display(HTML("<br/>"))
        display(HTML("Responses, in percentages"))
        df = pd.DataFrame(np.asarray(n).T, index=labels)
        if legend_labels == None:
            legend_labels = ['Frequencies'] 
        s = tabulate.tabulate(df, headers=legend_labels, tablefmt='html')
        display(HTML(s))

        display(HTML("<br/>"))

