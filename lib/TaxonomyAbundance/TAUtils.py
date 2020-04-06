from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
import uuid
import logging
from installed_clients.WorkspaceClient import Workspace as Workspace
from installed_clients.DataFIleUtilClient import DataFileUtil


# GraphData
class GraphData:

    def __init__(self, df, mdf, scratch=None, dfu=None):
        """
        This class takes parameters: pandas.DataFrame datafile with taxonomy being bottom row, and another
        pandas.DataFrame having range(len()) indexes and two columns for sample ID and category name
        :param df:
        :param mdf:
        :param scratch:
        :param dfu:
        """
        logging.info('GraphData class initializing...')
        # Set datafile.csv variable #
        self.df = df
        self.mdf = mdf
        # kbase client stuff
        self.scratch = scratch
        self.dfu = dfu

        # Number of rows and columns #
        self.number_of_cols = len(self.df.columns)
        self.number_of_rows = len(self.df.index)
        # Sample list #
        self.samples = list(self.df.index)
        # Initialize dictionary #
        self.the_dict = {}
        # Metadata Dict #
        self.metadata_dict = {}
        # Number of taxonomic_stings in 'Other' category #
        self.other_count = 0

        # Get sample_sums #
        self.sample_sums = self.compute_row_sums()

        # Set total_sum of entire Matrix #
        self.total_sum = 0
        for i in range(len(self.sample_sums)):
            self.total_sum += self.sample_sums[i]

        # METADATA #
        if not mdf.empty:
            self.mdf_categories = self.compute_mdf_categories()
        self.metadata_grouping = []

        # Paths
        self.html_paths = []
        self.img_paths = []

    def compute_mdf_categories(self):
        return list(self.mdf.columns)

    def compute_row_sums(self):
        row_sums = []
        for sample in self.df.index:
            try:
                row_sums.append(pd.to_numeric(self.df.loc[sample]).sum())
            except ValueError:
                return np.array(row_sums)
        return np.array(row_sums)

    def push_to_the_dict(self, level=1):
        """
        The first part, if else statements, gets taxonomic string to use as dictionary key.
        The last part, try except, pushes data into dictionary.
        :param level:
        :return:
        """
        logging.info('Pushing into main dictionary for level: {}'.format(level))
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
                             pos + 2:(len(taxonomic_str)) if (taxonomic_str.find(',', pos) == -1) else
                             taxonomic_str.find(',', pos)]

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
            except KeyError:
                self.the_dict.update({level_str: col_values_np_array})

    def percentize_the_dict(self, cutoff=-1.0):
        """
        Changes the_dict values to percentages based on sample_sums.
        Also groups based on given cutoff value into 'Other' group
        :param cutoff
        :return:
        """
        logging.info('Calculating percentages and making Other category for cutoff: {}'.format(cutoff))
        # Averages by dividing array 'y' by array 'self.sample_sums'
        self.the_dict.update((x, y/self.sample_sums) for x, y in self.the_dict.items())

        # Makes 'Other' category and deletes the elements that that went in there as to not repeat
        to_del = list()
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

    def make_grp_dict(self, category_field_name):
        """
        returns a dictionary with keys being the different categories for grouping and values being lists of index
        numbers, the numbers correlate to the location the sample has in self.samples. These lists of numbers will
        be used to graph elements of the array values in self.the_dict in the order of group. So like elements
        pertaining to group1 are first then groups2.. etc.
        :param category_field_name:
        :return: grp_dict
        """
        logging.info('Making group dictionary for metadata column: {}'.format(category_field_name))
        grp_dict = {}
        metadata_samples = list(self.mdf.iloc[0:, 0])
        col_indx = self.mdf_categories.index(category_field_name)
        self.metadata_grouping = list(self.mdf.iloc[0:, col_indx])
        for category, sample in zip(self.metadata_grouping, metadata_samples):
            try:
                grp_dict[category].append(self.samples.index(sample))
            except KeyError:
                grp_dict.update({category: [self.samples.index(sample)]})
        return grp_dict

    def graph_by_group(self, level, cutoff, category_field_name):
        """
        Method for grouped plotting with plotly and calling self._save_fig() method
        :param level:
        :param cutoff:
        :param category_field_name:
        :return:
        """
        logging.info('Graphing by group. level: {}, category: {}'.format(level, category_field_name))
        taxonomy_levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
        grp_dict = self.make_grp_dict(category_field_name)

        plot_list = []
        for taxo_str, vals in self.the_dict.items():
            plot_x = []
            plot_y = []
            for grp_cat, grp_indxs in grp_dict.items():
                plot_x.append(grp_cat)
                for i in grp_indxs:
                    plot_x.append(self.samples[i])
                plot_y.append(0)
                for i in grp_indxs:
                    plot_y.append(vals[i])
            if taxo_str == 'Other':
                taxo_str += '(' + str(self.other_count) + '), cutoff: ' + str(cutoff)
            plot_list.append(go.Bar(name=taxo_str, x=plot_x, y=plot_y, hovertext=taxo_str))

        taxo_fig = go.Figure(data=plot_list)
        taxo_fig.update_layout(barmode='stack', title=('Bar Plot level: ' + taxonomy_levels[level-1]), bargap=0.05,
                               xaxis_title='Samples', yaxis_title='Percentage')
        taxo_fig.update_xaxes(tickangle=-90)

        html_folder = self._mk_dir()
        # Save plotly_fig.html and return path
        plotly_html_file_path = os.path.join(html_folder, "plotly_fig.html")
        plot(taxo_fig, filename=plotly_html_file_path)
        self.img_paths.append(plotly_html_file_path)
        plotly_html_file_path = os.path.join(html_folder, "plotly_fig_without_legend.html")
        taxo_fig.update_layout(showlegend=False)
        plot(taxo_fig, filename=plotly_html_file_path)
        self.img_paths.append(plotly_html_file_path)

        self._shock_and_set_paths(html_folder)

    def graph_all(self, level, cutoff):
        """
        Method for plotting with plotly and calling self._save_fig() method
        :param level:
        :param cutoff:
        :return:
        """
        logging.info('Graphing. level: {}'.format(level))
        taxonomy_levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
        plot_list = []
        for key, val in self.the_dict.items():
            if key == 'Other':
                key += '(' + str(self.other_count) + '), cutoff: ' + str(cutoff)
            plot_list.append(go.Bar(name=key, x=self.samples, y=val))

        taxo_fig = go.Figure(data=plot_list)
        taxo_fig.update_layout(barmode='stack', title=('Bar Plot level: ' + taxonomy_levels[level-1]), bargap=0.05,
                               xaxis_title='Samples', yaxis_title='Percentage')
        taxo_fig.update_xaxes(tickangle=-90)

        html_folder = self._mk_dir()
        # Save plotly_fig.html and return path
        plotly_html_file_path = os.path.join(html_folder, "plotly_fig.html")
        plot(taxo_fig, filename=plotly_html_file_path)
        self.img_paths.append(plotly_html_file_path)
        plotly_html_file_path = os.path.join(html_folder, "plotly_fig_without_legend.html")
        taxo_fig.update_layout(showlegend=False)
        plot(taxo_fig, filename=plotly_html_file_path)
        self.img_paths.append(plotly_html_file_path)

        self._shock_and_set_paths(html_folder)

    def _mk_dir(self):
        """
        Makes a folder directory to save the plotly html figures.
        :return: html_folder_path
        """
        logging.info('Making html directory')
        # set up directory in scratch
        output_dir = os.path.join(self.scratch, str(uuid.uuid4()))
        os.mkdir(output_dir)
        # set up directory for html folder
        html_folder = os.path.join(output_dir, 'html')
        os.mkdir(html_folder)
        return html_folder

    def _shock_and_set_paths(self, html_folder):
        """
        Sets "self.html_paths" after calling dfu.file_to_shock
        :param html_folder:
        :return:
        """
        # have needed files saved to folder before shock
        shock = self.dfu.file_to_shock({'file_path': html_folder,
                                        'make_handle': 0,
                                        'pack': 'zip'})
        # list that goes to 'html_links'
        self.html_paths.append({'shock_id': shock['shock_id'],
                                'name': 'plotly_fig_without_legend.html',
                                'label': 'Barplot without legend'})
        self.html_paths.append({'shock_id': shock['shock_id'],
                                'name': 'plotly_fig.html',
                                'label': 'Barplot with legend'})

    def graph_this(self, level=1, cutoff=-1.0, category_field_name=''):
        """
        Calls methods to analyze data and graph. Determines whether to graph with grouping or not.
        :param level:
        :param cutoff:
        :param category_field_name:
        :return:
        """
        logging.info('graph_this(level={}, cutoff={}, category_field_name={})'.format(level, cutoff,
                                                                                      category_field_name))
        self.push_to_the_dict(level)
        self.percentize_the_dict(cutoff)
        if len(category_field_name) > 0:
            self.graph_by_group(level, cutoff, category_field_name)
        else:
            self.graph_all(level, cutoff)


# Methods that retrieve KBase data from Matrixs and Mappings ###
def get_df(amp_permanent_id, dfu):
    """
    Get Amplicon Matrix Data then make Pandas.DataFrame(),
    also get taxonomy data and add it to df, then transpose and return
    :param amp_permanent_id:
    :param dfu:
    :return:
    """
    logging.info('Getting DataObject')
    # Amplicon data
    obj = dfu.get_objects({'object_refs': [amp_permanent_id]})
    amp_data = obj['data'][0]['data']

    row_ids = amp_data['data']['row_ids']
    col_ids = amp_data['data']['col_ids']
    values = amp_data['data']['values']
    # Add 'taxonomy' column
    col_ids.append('taxonomy')
    # Make pandas DataFrame
    df = pd.DataFrame(index=row_ids, columns=col_ids)
    for i in range(len(row_ids)):
        df.iloc[i, :-1] = values[i]

    # Get object
    test_row_attributes_permanent_id = obj['data'][0]['data']['row_attributemapping_ref']
    obj = dfu.get_objects({'object_refs': [test_row_attributes_permanent_id]})
    tax_dict = obj['data'][0]['data']['instances']

    # Add taxonomy data and transpose matrix
    for row_indx in df.index:
        df.loc[row_indx]['taxonomy'] = tax_dict[row_indx][0]
    df = df.T
    return df


def get_mdf(attribute_mapping_obj_ref, category_name, dfu):
    """
    Metadata: make range(len()) index matrix with ID and Category columns
    :param attribute_mapping_obj_ref:
    :param category_name:
    :param dfu:
    :return:
    """
    logging.info('Getting MetadataObject')
    # Get object
    obj = dfu.get_objects({'object_refs': [attribute_mapping_obj_ref]})
    meta_dict = obj['data'][0]['data']['instances']
    attr_l = obj['data'][0]['data']['attributes']

    # Find index of specified category name
    indx = 0
    for i in range(len(attr_l)):
        if attr_l[i]['attribute'] == category_name:
            indx = i
            break
    # Set metadata_samples
    metadata_samples = meta_dict.keys()
    # Make pandas DataFrame
    mdf = pd.DataFrame(index=range(len(metadata_samples)), columns=['ID', category_name])
    i = 0
    for key, val in meta_dict.items():
        mdf.iloc[i] = [key, val[indx]]
        i += 1
    return mdf
# End of KBase data retrieving methods ###


def run(amp_id, attri_map_id, grouping_label, threshold, taxonomic_level, callback_url, scratch):
    """
    First method that is ran. Makes instance of GraphData class. Determines whether to get metadata or not.
    :param amp_id:
    :param attri_map_id:
    :param grouping_label:
    :param threshold:
    :param taxonomic_level:
    :param callback_url:
    :param scratch:
    :return: {
        'img_paths': g1.img_paths,
        'html_paths': g1.html_paths
        }
    """
    logging.info('run(grouping: {}, cutoff: {}, level: {})'.format(grouping_label, threshold, taxonomic_level))
    dfu = DataFileUtil(callback_url)
    df = get_df(amp_permanent_id=amp_id, dfu=dfu)
    try:
        if len(grouping_label) > 0:
            mdf = get_mdf(attribute_mapping_obj_ref=attri_map_id, category_name=grouping_label,
                          dfu=dfu)
            g1 = GraphData(df=df, mdf=mdf, scratch=scratch, dfu=dfu)
    except TypeError:
        g1 = GraphData(df=df, mdf=pd.DataFrame(), scratch=scratch, dfu=dfu)
        grouping_label = ""
    g1.graph_this(level=int(taxonomic_level), cutoff=threshold, category_field_name=grouping_label)
    return {
        'img_paths': g1.img_paths,
        'html_paths': g1.html_paths
    }
