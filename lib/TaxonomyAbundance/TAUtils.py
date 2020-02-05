################################################ Start of my code ##############################################
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random


class GraphData:

    # This class takes parameters: pd.read_csv(\'datafile\'), OTU_Placement_in_Matrix. #
    def __init__(self, df, mdf, OTU_Placement_in_Matrix='Bottom'):
        # Set datafile.csv variable #
        self.df = df
        self.mdf = mdf

        # Know layout of data matrix in file #
        self.OTU_Placement_in_Matrix = OTU_Placement_in_Matrix

        # Number of rows and columns #
        self.cols = len(self.df.columns)
        self.rows = self.df.shape[0]
        # Sample list #
        if self.OTU_Placement_in_Matrix == 'Bottom':
            self.samples = list(df.iloc[:-1, 0])
        else:
            pass  ### Come back to this when you have right-sided taxonomy
        # Initialize dictionary #
        self.the_dict = {}
        # Metadata Dict #
        self.metadata_dict = {}
        # Array of percentages #
        self.perc_arr = []
        # Number of OTUs in 'Other' category #
        self.other_count = 0
        # Init sample index #
        self.sample_index = int()

        # Get column_sums OR row_sums array depending on OTU_Placement_in_Matrix
        if self.OTU_Placement_in_Matrix == 'Bottom':
            self.OTU_sums = self.compute_column_sums()
        elif self.OTU_Placement_in_Matrix == 'Right':
            self.OTU_sums = self.compute_row_sums()

        # Get sample_sums #
        if self.OTU_Placement_in_Matrix == 'Bottom':
            self.sample_sums = self.compute_row_sums()
        elif self.OTU_Placement_in_Matrix == 'Right':
            self.sample_sums = self.compute_column_sums()

        # Set total_sum of entire Matrix #
        self.total_sum = 0
        for i in range(len(self.sample_sums)):
            self.total_sum += self.sample_sums[i]

        ### METADATA ###
        self.mdf_categories = self.compute_mdf_categories()

    def compute_mdf_categories(self):
        return list(self.mdf.iloc[0, :])

    def compute_column_sums(self):
        col_sums = []
        for i in range(1, len(self.df.columns) - (1 if self.OTU_Placement_in_Matrix == 'Right' else 0)):
            col_sums.append(pd.to_numeric(self.df.iloc[0:self.rows - 1, i]).sum())
        return np.array(col_sums)

    def compute_row_sums(self):
        row_sums = []
        for i in range(self.rows - (1 if self.OTU_Placement_in_Matrix == 'Bottom' else 0)):
            row_sums.append(pd.to_numeric(self.df.iloc[i, 1:]).sum())
        return np.array(row_sums)

    ####################################################################################################################
    def push_to_the_dict(self, level=1, peek='all'):
        """ The first part, if else statements, gets taxonomic string to use as dictionary key.
            The last part, try except, pushes data into dictionary."""
        self.the_dict.clear()
        for i in range(1, len(self.df.columns)):
            # d: p: c: o: f: g:
            col_num = i
            col_values_np_array = np.array(pd.to_numeric(self.df.iloc[:-1, i]))

            taxonomic_str = self.df.iloc[self.rows - 1, col_num]  # Get Taxonomy string

            ''' Find Domain '''
            level_str = 'unclassified'
            if level > 0:
                pos = taxonomic_str.find('d:')
                if pos != -1:
                    level_str = taxonomic_str[
                                pos + 2:(len(taxonomic_str)) if (
                                        taxonomic_str.find(',', pos) == -1) else taxonomic_str.find(
                                    ',', pos)]

            ''' Find Phylum '''
            if level > 1:
                pos = taxonomic_str.find('p:')
                if pos != -1:
                    level_str = level_str + ';' + taxonomic_str[pos + 2:(len(taxonomic_str)) if (
                            taxonomic_str.find(',', pos) == -1) else taxonomic_str.find(',', pos)]
                else:
                    level_str = level_str + ';' + 'unclassified'

            ''' Find Class '''
            if level > 2:
                pos = taxonomic_str.find('c:')
                if pos != -1:
                    level_str = level_str + ';' + taxonomic_str[pos + 2:(len(taxonomic_str)) if (
                            taxonomic_str.find(',', pos) == -1) else taxonomic_str.find(',', pos)]
                else:
                    level_str = level_str + ';' + 'unclassified'

            ''' Find Order '''
            if level > 3:
                pos = taxonomic_str.find('o:')
                if pos != -1:
                    level_str = level_str + ';' + taxonomic_str[pos + 2:(len(taxonomic_str)) if (
                            taxonomic_str.find(',', pos) == -1) else taxonomic_str.find(',', pos)]
                else:
                    level_str = level_str + ';' + 'unclassified'

            ''' Find Family '''
            if level > 4:
                pos = taxonomic_str.find('f:')
                if pos != -1:
                    level_str = level_str + ';' + taxonomic_str[pos + 2:(len(taxonomic_str)) if (
                            taxonomic_str.find(',', pos) == -1) else taxonomic_str.find(',', pos)]
                else:
                    level_str = level_str + ';' + 'unclassified'

            ''' Find Genus '''
            if level > 5:
                pos = taxonomic_str.find('g:')
                if pos != -1:
                    level_str = level_str + ';' + taxonomic_str[pos + 2:(len(taxonomic_str)) if (
                            taxonomic_str.find(',', pos) == -1) else taxonomic_str.find(',', pos)]
                else:
                    level_str = level_str + ';' + 'unclassified'

            """ Push to dictionary """
            try:
                self.the_dict[level_str] += col_values_np_array
            except:
                self.the_dict.update({level_str: col_values_np_array})

    ####################################################################################################################

    def percentize_the_dict(self, cutoff=-1.0, peek='all'):
        """ Changes the_dict values to percentages based on total matrix.
            Also groups based on given cutoff value into 'Other' group """
        to_del = list()
        if peek == 'total':
            self.the_dict.update((x, y / self.total_sum) for x, y in self.the_dict.items())
        else:
            self.the_dict.update((x, y / self.sample_sums) for x, y in self.the_dict.items())
        self.the_dict['Other'] = [0.0]
        for x, y in self.the_dict.items():
            if all(a < cutoff for a in y) and x != 'Other':
                self.other_count += 1
                try:
                    self.the_dict['Other'] += y
                except:
                    self.the_dict.update({'Other': y})
                to_del.append(x)
        if all(a == 0.0 for a in self.the_dict['Other']):
            to_del.append('Other')
        for key in to_del:
            if key in self.the_dict:
                del self.the_dict[key]
        for x, y, in self.the_dict.items():
            self.perc_arr.append(np.around(y * 100, decimals=5))

    def cumulative_percentize_the_dict(self):
        """ Makes values in the_dict reverse cumulative. Starting at 1 and then subtracting previous percentage.
            This is done so when the data is graphed, the higher ones are graphed first as to not overlap and cover
            up smaller ones """
        prev = np.zeros(len(self.sample_sums))
        ones = np.ones(len(self.sample_sums))
        for x, y in self.the_dict.items():
            temp = self.the_dict[x] + prev
            self.the_dict[x] = ones - prev
            prev = temp

    def make_grp_dict(self, category_field_name):
        """ returns a dictionary with keys being the different categories for grouping and values being lists of index
            numbers, the numbers correlate to the location the sample has in self.samples. These lists of numbers will
            be used to graph elements of the array values in self.the_dict in the order of group. So like elements
             pertaining to group1 are first then groups2.. etc."""
        grp_dict = {}
        metadata_samples = list(self.mdf.iloc[1:, 0])
        indx = self.mdf_categories.index(category_field_name)
        self.metadata_grouping = list(self.mdf.iloc[1:, indx])
        for key, val in zip(self.metadata_grouping, metadata_samples):
            try:
                grp_dict[key].append(self.samples.index(val))
            except:
                grp_dict.update({key: [self.samples.index(val)]})
        return grp_dict

    def graph_by_group(self, level, legend_font_size, cutoff, category_field_name):
        grp_dict = self.make_grp_dict(category_field_name)
        ord_l = []
        x_tick_grp_str_l = []
        x_tick_grp_pos_l = [-0.5]
        end = 0
        for key, val in grp_dict.items():
            start = end
            ord_l += val
            x_tick_grp_str_l.append(key + ':')
            for i in val:
                x_tick_grp_str_l.append(self.samples[i])
            end = len(ord_l)
            x_tick_grp_pos_l += range(start, end)
            x_tick_grp_pos_l.append(len(ord_l) - 0.5)

        i = 0
        for x, y in self.the_dict.items():
            rand_color = ("#%06x" % random.randint(0, 0xFFFFFF))
            plt.bar(range(len(ord_l)), np.array(y)[ord_l], color=rand_color, width=0.9, label=(
                    str(np.array(self.perc_arr[i])[ord_l]) + '% ' + x + (
                '(' + str(self.other_count) + ') cutoff: ' + str(cutoff) if x == 'Other' else '')))
            i += 1
        plt.xticks(x_tick_grp_pos_l, x_tick_grp_str_l, rotation=25)
        # plt.xticks(range(len(self.samples)), np.array(self.samples)[ord_l], rotation=25)
        plt.title('Level: ' + str(level))
        plt.xlabel("Samples")
        plt.legend(loc='center right', prop={'size': legend_font_size})
        plt.tight_layout()
        plt.show()
        plt.savefig('/kb/module/work/tmp/group_bar_graph.png')

    def graph_all(self, level, legend_font_size, cutoff):
        bars = range(len(self.sample_sums))
        i = 0
        for x, y in self.the_dict.items():
            plt.bar(bars, y, label=(str(self.perc_arr[i]) + '% ' + x + (
                '(' + str(self.other_count) + ') cutoff: ' + str(cutoff) if x == 'Other' else '')),
                    color=("#%06x" % random.randint(0, 0xFFFFFF)))
            i += 1

        plt.title('Level: ' + str(level))
        plt.xlabel("Samples")
        plt.xticks(bars, self.samples, rotation=25)
        plt.legend(loc='center right', prop={'size': legend_font_size})
        plt.tight_layout()
        plt.show()
        plt.savefig('/kb/module/work/tmp/bar_graph.png')

    def graph_this(self, level=1, legend_font_size=8, cutoff=-1.0, peek='all', category_field_name=''):
        """ MAIN GRAPHING METHOD """
        if level < 1:
            print('ERROR: Level must be a value greater than zero!')
            return

        if peek != 'total' and peek != 'all':
            self.sample_index = self.samples.index(peek)
        self.push_to_the_dict(level, peek)
        self.percentize_the_dict(cutoff, peek)
        self.cumulative_percentize_the_dict()
        if len(category_field_name) > 0:
            self.graph_by_group(level, legend_font_size, cutoff, category_field_name)
        else:
            self.graph_all(level, legend_font_size, cutoff)

def run(csv_filepath, xls_filepath):
    df = pd.read_csv(csv_filepath)
    mdf = pd.read_excel(xls_filepath)
    g1 = GraphData(df=df, mdf=mdf)
    g1.graph_this(level=1, legend_font_size=8, cutoff=0.005, peek='all', category_field_name='Field name (informal classification)')
    print(g1.mdf_categories)

############################################# End of my code ###################################################