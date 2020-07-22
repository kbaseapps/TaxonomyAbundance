from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly
import pandas as pd
import numpy as np
import os
import uuid
import logging
import itertools

from installed_clients.WorkspaceClient import Workspace as Workspace
from installed_clients.DataFileUtilClient import DataFileUtil

from dprint import dprint

taxonomy_levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus']

# GraphData
class GraphData:

    def __init__(self, df, mdf, scratch=None, dfu=None):
        """
        :param df: pandas.DataFrame dataframe with taxonomy being bottom row
        :param mdf: pandas.DataFrame having range(len()) indexes
                    and two columns for sample ID and category name
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
        self.samples = list(self.df.index)[:-1]
        # Metadata Dict #
        self.metadata_dict = {}

        # Get sample_sums #
        self.sample_sums = self.compute_row_sums()
        # Initialize dictionary #
        self.the_dicts = {}
        # Set total_sum of entire Matrix #
        self.total_sum = 0
        for i in range(len(self.sample_sums)):
            self.total_sum += self.sample_sums[i]

        # METADATA #
        if not mdf.empty:
            self.mdf_categories = self.compute_mdf_categories()
            dprint('self.mdf', 'self.mdf_categories', run=locals()) # ?
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
            except ValueError: # last row is taxonomies
                return np.array(row_sums)
        return np.array(row_sums)

    def push_to_the_dict(self, level=1):
        """
        Iter through columns of self.df,
        which consist of amplicon's AmpliconMatrix data and taxonomy.
        Truncate taxonomy to `level`
        and sum up all data for that amplicon

        :param level: in {1, ..., 6} corresponding to {Domain, ..., Genus}
        :return:
        """
        logging.info('Pushing into main dictionary for level: {}'.format(level))
        the_dict = dict()

        for label, content in self.df.iteritems(): # iter through amplicon cols
            data = np.array(content[:-1], dtype=float)
            taxonomy = content[-1]

            taxonomy = ';'.join([
                tax if tax != '' 
                    else 'unclassified' 
                    for tax in taxonomy.split(';')[:level]
            ])

            if taxonomy in the_dict:
                the_dict[taxonomy] += data
            else:
                the_dict[taxonomy] = data

        return the_dict


    def percentize_and_cutoff_the_dict(self, the_dict, cutoff=-1.0):
        """
        Changes the_dict values to percentages based on sample_sums.
        Also groups based on given cutoff value into 'Other' group
        :param cutoff
        :return:
        """
        logging.info('Calculating percentages and making Other category for cutoff: {}'.format(cutoff))
        # Averages by dividing array 'y' by array 'self.sample_sums'
        the_dict.update((x, y/self.sample_sums) for x, y in the_dict.items())

        # Makes 'Other' category and deletes the elements that that went in there as to not repeat
        to_del = list()
        the_dict['Other'] = [0.0]
        other_count = 0
        for x, y in the_dict.items():
            if all(a < cutoff for a in y) and x != 'Other':
                other_count += 1
                try:
                    the_dict['Other'] += y
                except ValueError:
                    the_dict.update({'Other': y})
                to_del.append(x)
        if all(a == 0.0 for a in the_dict['Other']):
            to_del.append('Other')
        for key in to_del:
            if key in the_dict:
                del the_dict[key]

        return the_dict, other_count

    def sort_the_dict(self, the_dict):
        '''
        Sort `the_dict` by taxonomy key, but with `Other` last
        '''
        Other = the_dict.pop('Other') if 'Other' in the_dict else None

        the_dict = dict(sorted(the_dict.items()))
        
        if Other is not None:
            the_dict.update(Other=Other)

    def make_grp_dict(self, category_field_name):
        """
        returns a dictionary with keys being the different categories for grouping and values being lists of index
        numbers, the numbers correlate to the location the sample has in self.samples. These lists of numbers will
        be used to graph elements of the array values in the_dict in the order of group. So like elements
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

    def graph_by_group(self, the_dict, other_count, level, cutoff, category_field_name=None):
        """
        VARS:

        self
        level
        cutoff
        category_field_name
        grp_dict
        the_dict (DE-INSTANTIATE)
        other_count (DE-INSTANTIATE)
        self.samples
        html_folder
        self.img_paths

        FUNCS:

        self.make_grp_dict
        self._mk_dir
        self._set_paths

        """
        logging.info('Graphing by group. level: {}, category: {}'.format(level, category_field_name))
        grp_dict = self.make_grp_dict(category_field_name); dprint('grp_dict', run=locals(), where=True)


        taxo_fig = make_subplots(
            rows=1, 
            cols=len(grp_dict),
            horizontal_spacing=0.05,
            x_title="Sample<br>Grouping by: %s" % category_field_name,
            subplot_titles=list(grp_dict.keys()),
            column_widths=[len(ind_l) for ind_l in grp_dict.values()]
        )

        dprint(
            '[len(ind_l) for ind_l in grp_dict.values()]',
            '[len(ind_l)/len(self.mdf) for ind_l in grp_dict.values()]', 
            'sum([len(ind_l)/len(self.mdf) for ind_l in grp_dict.values()])', 
            run=locals()
        )
        
        color_iter = itertools.cycle(px.colors.qualitative.Plotly) # reset color iter
        traces = []
        for taxo_str, vals in the_dict.items():
            if taxo_str == 'Other':
                taxo_str += ' (' + str(other_count) + '), cutoff: ' + str(cutoff)            
            marker_color = next(color_iter)
            for col, (grp, grp_indxs) in zip(range(1, len(grp_dict) + 1), grp_dict.items()):
                plot_x = []
                plot_y = []
                for i in grp_indxs:
                    plot_x.append(self.samples[i])
                    plot_y.append(vals[i])
                taxo_fig.add_trace(
                    go.Bar(
                        name=taxo_str, 
                        x=plot_x, 
                        y=plot_y, 
                        hovertext=taxo_str,
                        legendgroup=taxo_str,
                        marker_color=marker_color,
                        showlegend=True if col == 1 else False,
                    ), 
                    row=1,
                    col=col,
                )
                taxo_fig.add_trace(
                    go.Bar(
                        name=taxo_str.upper(),
                        x=plot_x,
                        y=plot_y[::-1],
                        hovertext=taxo_str.upper(),
                        legendgroup=taxo_str,
                        marker_color=marker_color,
                        showlegend=True if col == 1 else False,
                        visible=False,
                    ),
                    row=1,
                    col=col
                )

        num_tax = len(taxo_fig.data)

        taxo_fig.update_layout(
            barmode='stack', 
            bargap=0.03,
            title_text=('Rank: ' + taxonomy_levels[level-1]), 
            title_y=0.97,
            title_x=0.5,
            title_xref='paper',
            yaxis_title='Proportion',
            legend_traceorder='reversed',
            margin=dict(b=115),
        )
        
        # lower x_title
        taxo_fig.layout.annotations[-1].update(
            dict(
                y=-0.04,            
            )
        )

        taxo_fig.update_yaxes(
            range=[0,1],
        )
        
        taxo_fig.update_xaxes(
            tickangle=15,
        )

        dropdown_y = 1.1

        taxo_fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=['showlegend', True],
                            label='Show legend',
                            method='relayout',
                        ),
                        dict(
                            args=['showlegend', False],
                            label='Hide legend',
                            method='relayout',
                        ),
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=dropdown_y,
                    yanchor="top"
                ),
                dict(
                    buttons=[
                        dict(
                            args=[
                                {
                                    'visible': [True, False] * num_tax,
                                    'show_legend': [True, False] * num_tax
                                },
                                {
                                    'title': 'Ordered'
                                }

                            ],
                            label='Ordered',
                            method='update'
                        ),
                        dict(
                            args=[
                                {
                                    'visible': [False, True] * num_tax,
                                    'show_legend': [False, True] * num_tax
                                },
                                {
                                    'title': 'Reverse Ordered'
                                }
                            ],
                            label='Reverse Ordered',
                            method='update'
                        )
                    ],
                    direction='down',
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.2,
                    xanchor="left",
                    y=dropdown_y,
                    yanchor="top"
                )
            ],
        )

        dprint(
            'taxo_fig.layout.annotations',
            'len(taxo_fig.layout.annotations)',
            run=locals()
        )

        html_folder = self._mk_dir()
        # Save plotly_fig.html and return path
        plotly_html_file_path = os.path.join(html_folder, "plotly_fig.html")
        plot(taxo_fig, filename=plotly_html_file_path)
        self.img_paths.append(plotly_html_file_path)

        self._set_paths(html_folder)

        dprint(
            'level',
            'cutoff',
            'category_field_name',
            'grp_dict',
            'the_dict',
            'other_count',
            'self.samples',
            'html_folder',
            'self.img_paths',
            run=locals()
        )

    def graph_all(self, level, cutoff):
        """
        Method for plotting with plotly and calling self._save_fig() method
        :param level:
        :param cutoff:
        :return:
        """
        logging.info('Graphing. level: {}'.format(level))
        plot_list = []
        for key, val in self.the_dict.items():
            if key == 'Other':
                key += '(' + str(self.other_count) + '), cutoff: ' + str(cutoff)
            plot_list.append(go.Bar(name=key, x=self.samples, y=val))

        taxo_fig = go.Figure(data=plot_list)
        taxo_fig.update_layout(barmode='stack', title=('Bar Plot level: ' + taxonomy_levels[level-1]), bargap=0.05,
                               xaxis_title='Samples', yaxis_title='Proportion')
        taxo_fig.update_xaxes(tickangle=15)

        html_folder = self._mk_dir()
        # Save plotly_fig.html and return path
        plotly_html_file_path = os.path.join(html_folder, "plotly_fig.html")
        plot(taxo_fig, filename=plotly_html_file_path)
        self.img_paths.append(plotly_html_file_path)
        plotly_html_file_path = os.path.join(html_folder, "plotly_fig_without_legend.html")
        taxo_fig.update_layout(showlegend=False)
        plot(taxo_fig, filename=plotly_html_file_path)
        self.img_paths.append(plotly_html_file_path)

        self._set_paths(html_folder)

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

    def _set_paths(self, html_folder):
        """
        Sets "self.html_paths" after calling dfu.file_to_shock
        :param html_folder:
        :return:
        """
        self.html_paths.append({'path': os.path.join(html_folder, 'plotly_fig.html'),
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
        for level in range(1, len(taxonomy_levels)+1):
            the_dict = self.push_to_the_dict(level); dprint('the_dict', 'len(the_dict)', run=locals(), where=True)
            the_dict, other_count = self.percentize_and_cutoff_the_dict(the_dict, cutoff); dprint('the_dict', 'len(the_dict)', run=locals(), where=True)
            self.the_dicts[taxonomy_levels[level-1]] = [
                the_dict,
                other_count
            ]
        dprint('self.the_dicts', run=locals())
        if len(category_field_name) > 0:
            self.graph_by_group(cutoff, category_field_name)
        else:
            self.graph_all(level, cutoff)



def get_id2taxonomy(attributes, instances) -> dict:
    """
    (1) Find acceptable taxonomy source in `attributes` 
        rn limited to RDP taxonomy and taxonomy parsed/standardized at upload
    (2) Collect into dict

    :attributes: one of the fields of an AttributeMapping object, list of dicts
    :instances: one of the fields of an AttributeMapping object, dict of lists
    """
 
    # Check for index of 'attribute'/'source' of either:
    # * 'parsed_user_taxonomy'/'upload', or
    # * 'RDP Classifier taxonomy'/'kb_rdp_classifier/run_classify, conf=*, gene=*, minWords=*', or
    # TODO selecting amongst multiple taxonomy attributes
    # TODO particularly: upload, rdp, multiple rdp with diff paramms
    # TODO allow other formats by calling GenericsAPI process_taxonomy
    d_usr = {'attribute': 'parsed_user_taxonomy', 'source': 'upload'} 
    d_rdp = {
        'attribute': 'RDP Classifier taxonomy', 
        'source': 'kb_rdp_classifier/run_classify, conf=*, gene=*, minWords=*'
    }

    # TODO more precise matching
    rdp_match_l = [d_rdp['attribute'] == attribute['attribute'] for attribute in attributes] # match by 'attribute' field

    if d_usr in attributes:
        ind = attributes.index(d_usr)
        logging.info('Using source/attribute `%s`' % str(d_usr))
    elif any(rdp_match_l):
        ind = rdp_match_l.index(True)
        logging.info('Using source/attribute `%s`' % str(d_rdp))
    else:
        raise Exception(
            'Sorry, the row AttributeMapping `%s` referenced by input AmpliconMatrix `%s` '
            'does not have one of the expected taxonomy fields, '
            'which are `%s` or `%s`.' 
            % (amp_mat_name, row_attrmap_name, str(d_usr), str(d_rdp))
        )
   
    #
    id2taxonomy = {id: instance[ind] for id, instance in instances.items()}
    return id2taxonomy



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
    amp_mat_name = obj['data'][0]['info'][1]
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
    test_row_attributes_permanent_id = obj['data'][0]['data']['row_attributemapping_ref'] # TODO field is optional
    obj = dfu.get_objects({'object_refs': [test_row_attributes_permanent_id]})
    row_attrmap_name = obj['data'][0]['info'][1]
    attributes = obj['data'][0]['data']['attributes']
    instances = obj['data'][0]['data']['instances']

    # Get id2taxonomy dict
    id2taxonomy = get_id2taxonomy(attributes, instances)

    # Add taxonomy data and transpose matrix
    for row_indx in df.index:
        df.loc[row_indx]['taxonomy'] = id2taxonomy[row_indx]
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
    df = get_df(amp_permanent_id=amp_id, dfu=dfu)  # transpose of AmpMat df with taxonomy col appended
    try:
        if len(grouping_label) > 0:
            mdf = get_mdf(attribute_mapping_obj_ref=attri_map_id, category_name=grouping_label,
                          dfu=dfu)  # df of sample to group
            g1 = GraphData(df=df, mdf=mdf, scratch=scratch, dfu=dfu)
    except TypeError:
        g1 = GraphData(df=df, mdf=pd.DataFrame(), scratch=scratch, dfu=dfu)
        grouping_label = ""
    g1.graph_this(level=int(taxonomic_level), cutoff=threshold, category_field_name=grouping_label)
    return {
        'img_paths': g1.img_paths,
        'html_paths': g1.html_paths
    }
