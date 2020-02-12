################################################ Start of my code ##############################################
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import uuid
import random
from installed_clients.WorkspaceClient import Workspace as Workspace
from installed_clients.DataFIleUtilClient import DataFileUtil

# GraphData
class GraphData:

    # This class takes parameters: pandas.DataFrame datafile with taxonomy being bottom row, and another
    # pandas.DataFrame having range(len()) indexes and two columns for sample ID and category name
    def __init__(self, df, mdf, otu_placement_in_matrix1='Columns', scratch=None):
        # Know layout of data matrix in file #
        self.otu_placement_in_matrix = otu_placement_in_matrix1

        # Set datafile.csv variable #
        self.df = df
        self.mdf = mdf
        self.scratch = scratch

        # Number of rows and columns #
        self.number_of_cols = len(self.df.columns)
        self.number_of_rows = len(self.df.index)
        # Sample list #
        self.samples = list(self.df.index)
        # Initialize dictionary #
        self.the_dict = {}
        # Metadata Dict #
        self.metadata_dict = {}
        # Array of percentages #
        self.percent_arr = []
        # Number of taxonomic_stings in 'Other' category #
        self.other_count = 0
        # Init sample index #
        self.sample_index = int()

        # Get column_sums OR row_sums array depending on otu_placement_in_matrix
        self.OTU_sums = self.compute_column_sums()

        # Get sample_sums #
        self.sample_sums = self.compute_row_sums()

        # Set total_sum of entire Matrix #
        self.total_sum = 0
        for i in range(len(self.sample_sums)):
            self.total_sum += self.sample_sums[i]

        # METADATA #
        if mdf.empty != True:
            self.mdf_categories = self.compute_mdf_categories()
        self.metadata_grouping = []

        #IMG Path
        self.img_paths = []

    def append_taxonomy(self, tax_dict={}):
        pass

    def compute_mdf_categories(self):
        return list(self.mdf.columns)

    def compute_column_sums(self):
        col_sums = []
        for otu in self.df.columns:
            col_sums.append(pd.to_numeric(self.df[otu][0:-1]).sum())  # -1 is to stop before taxonomy row
        return np.array(col_sums)

    def compute_row_sums(self):
        row_sums = []
        for sample in self.df.index:
            try:
                row_sums.append(pd.to_numeric(self.df.loc[sample]).sum())
            except ValueError:
                return np.array(row_sums)
        return np.array(row_sums)

    ####################################################################################################################
    def push_to_the_dict(self, level=1):
        """ The first part, if else statements, gets taxonomic string to use as dictionary key.
            The last part, try except, pushes data into dictionary."""
        self.the_dict.clear()
        for i in range(1, len(self.df.columns)):
            # d: p: c: o: f: g:
            col_num = i
            col_values_np_array = np.array(pd.to_numeric(self.df.iloc[:-1, i]))

            taxonomic_str = self.df.iloc[self.number_of_rows - 1, col_num]  # Get Taxonomy string

            ''' Find Domain '''
            level_str = 'unclassified'
            if level > 0:
                pos = taxonomic_str.find('d:')
                if pos != -1:
                    level_str = taxonomic_str[
                             pos + 2:(len(taxonomic_str)) if (taxonomic_str.find(',', pos) == -1) else taxonomic_str.find(
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
            self.the_dict.update((x, y/self.total_sum) for x, y in self.the_dict.items())
        else:
            self.the_dict.update((x, y/self.sample_sums) for x, y in self.the_dict.items())  # perhaps just values
        self.the_dict['Other'] = [0.0]
        for x, y in self.the_dict.items():
            if all(a < cutoff for a in y) and x != 'Other':
                self.other_count += 1
                try:
                    self.the_dict['Other'] += y
                except ValueError:
                    self.the_dict.update({'Other': y})
                to_del.append(x)
        if all(a == 0.0 for a in self.the_dict['Other']):
            to_del.append('Other')
        for key in to_del:
            if key in self.the_dict:
                del self.the_dict[key]
        for x, y, in self.the_dict.items():
            self.percent_arr.append(np.around(y * 100, decimals=5))

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
        metadata_samples = list(self.mdf.iloc[0:, 0])
        indx = self.mdf_categories.index(category_field_name)
        self.metadata_grouping = list(self.mdf.iloc[0:, indx])
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
        pos_list_to_plot_samples = []
        next_sample_pos = 0
        end = 0
        for key, val in grp_dict.items():  # key = grouping_group, val = indexes_of_samples_in_group
            start = end
            ord_l += val
            x_tick_grp_str_l.append(key+':')
            for i in val:
                x_tick_grp_str_l.append(self.samples[i])
                pos_list_to_plot_samples.append(next_sample_pos)
                next_sample_pos += 1
            next_sample_pos += 1
            end = len(ord_l)
            x_tick_grp_pos_l += range(start, end)
            x_tick_grp_pos_l.append(len(ord_l)-0.5)

        i = 0
        for x, y in self.the_dict.items():  # x = taxonomic_string, y = cumulative_percentage
            rand_color = ("#%06x" % random.randint(0, 0xFFFFFF))
            plt.bar(pos_list_to_plot_samples, np.array(y)[ord_l], color=rand_color, width=0.9, label=(str(np.array(self.percent_arr[i])[ord_l])+'% ' + x + ('('+str(self.other_count)+') cutoff: '+str(cutoff) if x == 'Other' else '')))
            i += 1
        plt.xticks(range(-1, len(x_tick_grp_str_l)), x_tick_grp_str_l, rotation=25)
        plt.title('Level: '+str(level))
        plt.xlabel('Samples')
        fig = plt.gcf()

        output_dir = os.path.join(self.scratch, str(uuid.uuid4()))
        os.mkdir(output_dir)

        bar_graph_path0 = os.path.join(output_dir, 'bar_graph_0.png')
        bar_graph_path1 = os.path.join(output_dir, 'bar_graph_1.png')

        fig.savefig(bar_graph_path0)
        self.img_paths.append(bar_graph_path0)
        plt.legend(loc='center right', prop={'size': legend_font_size})
        fig = plt.gcf()

        fig.savefig(bar_graph_path1)
        self.img_paths.append(bar_graph_path1)
        plt.show()


    def graph_all(self, level, legend_font_size, cutoff):
        bars = range(len(self.sample_sums))
        i = 0
        for x, y in self.the_dict.items():
            plt.bar(bars, y, label=(str(self.percent_arr[i])+'% '+ x + ('('+str(self.other_count)+') cutoff: '+str(cutoff) if x == 'Other' else '')), color=("#%06x" % random.randint(0, 0xFFFFFF)))
            i += 1

        plt.title('Level: '+str(level))
        plt.xlabel("Samples")
        plt.xticks(bars, self.samples, rotation=25)
        plt.legend(loc='center right', prop={'size': legend_font_size})
        plt.show()

    def graph_this(self, level=1, legend_font_size=8, cutoff=-1.0, peek='all', category_field_name=''):
        """ MAIN GRAPHING METHOD """
        if level < 1:
            print('ERROR: Level must be a value greater than zero!')
            return

        if peek != 'total' and peek != 'all':
            self.sample_index = self.samples.index(peek)
        self.push_to_the_dict(level)
        self.percentize_the_dict(cutoff, peek)
        self.cumulative_percentize_the_dict()
        if len(category_field_name) > 0:
            self.graph_by_group(level, legend_font_size, cutoff, category_field_name)
        else:
            self.graph_all(level, legend_font_size, cutoff)


# Methods that retrieve KBase data from Matrixs and Mappings ###
def get_df(amp_permanent_id, test_row_attributes_permanent_id, callback_url, token):
    # Get Amplicon Matrix Data then make Pandas.DataFrame(), also get taxonomy data and add it to df, then transpose and return
    # Amplicon data
    # ws = Workspace(url, token=token)
    dfu = DataFileUtil(callback_url)
    obj = dfu.get_objects({'object_refs' : [amp_permanent_id]})
    amp_data = obj['data'][0]['data']

    row_ids = amp_data['data']['row_ids']
    col_ids = amp_data['data']['col_ids']
    values = amp_data['data']['values']
    # Add 'taxonomy' column
    col_ids.append('taxonomy')
    # Make pandas DataFrame
    df = pd.DataFrame(index=row_ids, columns=col_ids)
    for i in range(len(row_ids)):
        df.iloc[i,:-1] = values[i]

    # Get object
    # obj = ws.get_objects2({'objects' : [{'ref' : test_row_attributes_permanent_id}]})
    obj = dfu.get_objects({'object_refs' : [test_row_attributes_permanent_id]})
    tax_dict = obj['data'][0]['data']['instances']

    # Add taxonomy data and transpose matrix
    for row_indx in df.index:
        df.loc[row_indx]['taxonomy'] = tax_dict[row_indx][0]
    Tdf = df.T
    return Tdf

def get_mdf(attributeMappingId, category_name, callback_url, token):
    # Metadata: make range(len()) index matrix with ID and Category columns
    # Get object
    # ws = Workspace(url, token=token)
    # obj = ws.get_objects2({'objects' : [{'ref' : attributeMappingId}]})
    dfu = DataFileUtil(callback_url)
    obj = dfu.get_objects({'object_refs': [attributeMappingId]})
    meta_dict = obj['data'][0]['data']['instances']
    attr_l = obj['data'][0]['data']['attributes']

    # Find index of specified category name
    indx = 0
    for i in range(len(attr_l)):
        if attr_l[i]['attribute'] == category_name:
            indx = i
            break;
    # Set metadata_samples
    metadata_samples = meta_dict.keys()
    # Make pandas DataFrame
    mdf = pd.DataFrame(index=range(len(metadata_samples)), columns=['ID',category_name])
    # Print sample and their grouping value
    i = 0
    for key, val in meta_dict.items():
        mdf.iloc[i] = [key, val[indx]]
        i+=1
    return mdf
# End of KBase data retrieving methods ###


def run(amp_id, row_attributes_id, attri_map_id, grouping_label, threshold, taxonomic_level, callback_url, token, scratch):
    df = get_df(amp_permanent_id=amp_id, test_row_attributes_permanent_id=row_attributes_id, callback_url=callback_url, token=token)
    if len(grouping_label) > 0:
        mdf = get_mdf(attributeMappingId=attri_map_id, category_name=grouping_label, callback_url=callback_url, token=token)
        g1 = GraphData(df=df, mdf=mdf, scratch=scratch)
    else:
        g1 = GraphData(df=df,mdf=pd.DataFrame(), scratch=scratch)
    g1.graph_this(level=taxonomic_level, legend_font_size=12, cutoff=threshold, peek='all', category_field_name=grouping_label)
    return g1.img_paths

############################################# End of my code ###################################################