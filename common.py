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
        return self.metadata.varLabels[bytes(var, encoding='utf-8')].decode('windows-1252')

    def get_value_labels(self, var):
        return dict([(k, v.decode('windows-1252')) for k, v in self.metadata.valueLabels[bytes(var, encoding='windows-1252')].items()])

    def freq_table(self, var):
        return pd.DataFrame(list(self.get_value_labels(var).items()), columns=['index', self.get_var_label(var)]) \
                .set_index('index', drop=True) \
                .merge(pd.DataFrame(self.data[var].value_counts().sort_index()) \
                       .rename(columns = {var: 'Frequency'}), \
                       left_index = True, right_index = True) \


    def gen_histogram(self, col, legend_labels = None):
        fig, ax = plt.subplots()
        labels = self.get_value_labels(col).values()
        bin_length = len(labels)
        
        # If legend labels exist, split the data by the first index value (assumes len(legend_labels) == len(self.data.index.levels))
        if legend_labels != None and hasattr(self.data.index, 'levels') and len(legend_labels) == len(self.data.index.levels):
            d = pd.Series(self.data[col], index = self.data.index).unstack(level=0).as_matrix().T
            d = [x[~np.isnan(x)] for x in d]
        else:
            d = self.data[col].dropna()
        n, bins, patches = ax.hist(d, bins = np.arange(1, bin_length + 2) - 0.5, rwidth = 0.8, stacked=True, label = legend_labels)

        # Add grid
        ax.grid(True)
        
        # Legend
        if legend_labels != None:
            ax.legend(prop={'size': 10})

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # scale values to interval [0,1]
        bns = bin_centers - min(bin_centers)
        bns /= max(bns)

        # rescale values to interval [0.2,0.8]
        scale_inc = 0.2
        bns *= (1.0 - scale_inc * 2.)
        bns += scale_inc

        # Apply colours
        cm = plt.get_cmap('Blues')
        #for c, patch in zip(bns, patches):
        #    patch.set_facecolor(cm(c))

        # Add title and axes
        plt.title(self.get_var_label(col), loc='left')
        ax.set_xticks(np.arange(1, bin_length + 1))
        ax.set_xticklabels(self.get_value_labels(col).values(), rotation=45, ha='right')
        plt.xlabel('Level')
        plt.ylabel('No. of responses')

        # Tweak spacing to prevent clipping of ylabel
    #     fig.tight_layout()
        plt.show()
        
        import tabulate
        from IPython.display import HTML, display
        df = pd.DataFrame(np.asarray(n).T, index=labels)
        if legend_labels == None:
            legend_labels = ['Frequencies'] 
        s = tabulate.tabulate(df, headers=legend_labels, tablefmt='html')
        display(HTML(s))


