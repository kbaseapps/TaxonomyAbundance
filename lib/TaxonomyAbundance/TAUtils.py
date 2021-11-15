from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import uuid
import logging
import itertools
import copy

from .debug import dprint

RANKS = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus']


class GraphData:

    def __init__(self, df, sample2group_df, cutoff, scratch):
        """
        :param df: pandas.DataFrame dataframe with taxonomy being bottom row
        :param sample2group_df: pandas.DataFrame having range(len()) indexes
                    and two columns for sample ID and category name
        :param scratch:
        :param dfu:
        """
        logging.info('GraphData class initializing...')
        # Set datafile.csv variable #
        self.df = df
        self.sample2group_df = sample2group_df

        # Directory structure #
        self.run_dir = os.path.join(scratch, 'tax_bar_' + str(uuid.uuid4()))
        os.mkdir(self.run_dir)

        # Sample list #
        self.samples = list(self.df.index)[:-1]

        # Get sample_sums #
        self.sample_sums = self.compute_row_sums()
        # Set total_sum of entire Matrix #
        self.total_sum = 0
        for i in range(len(self.sample_sums)):
            self.total_sum += self.sample_sums[i]

        # METADATA #

        # Calc tax2vals_d for each level #
        self.the_dict = {}
        self.compute_the_dict(cutoff)

    def compute_row_sums(self):
        row_sums = []
        for sample in self.df.index:
            try:
                row_sums.append(pd.to_numeric(self.df.loc[sample]).sum())
            except ValueError:  # last row is taxonomies
                return np.array(row_sums)
        return np.array(row_sums)

    def get_tax2vals_d(self, level):
        """
        Iter through columns of self.df,
        which consist of amplicon's AmpliconMatrix data and taxonomy.
        Truncate taxonomy to `level`
        and sum up all data for that amplicon

        :param level: in {0, ..., 5} corresponding to {Domain, ..., Genus}
        :return:
        """
        logging.info('Pushing into main dictionary for level: {}'.format(level))
        tax2vals_d = dict()

        for label, content in self.df.iteritems():  # iter through amplicon cols
            data = np.array(content[:-1], dtype=float)
            taxonomy = content[-1]

            taxonomy = ';'.join([
                tax if tax != '' else 'unclassified' for tax in taxonomy.split(';')[:level + 1]
            ])

            if taxonomy in tax2vals_d:
                tax2vals_d[taxonomy] += data
            else:
                tax2vals_d[taxonomy] = data

        return tax2vals_d

    def percentize_cutoff_tax2vals_d(self, tax2vals_d, cutoff):
        """
        Changes tax2vals_d values to percentages based on sample_sums.
        Also groups based on given cutoff value into 'Other' group
        :param cutoff
        :return:
        """
        logging.info('Calculating percentages and making Other category for cutoff: {}'.format(
            cutoff))
        # Averages by dividing array 'y' by array 'self.sample_sums'
        tax2vals_d.update((x, y / self.sample_sums) for x, y in tax2vals_d.items())

        # Makes 'Other' category and deletes the elements that that went in there as to not repeat
        to_del = list()
        tax2vals_d['Other'] = [0.0]
        other_count = 0
        for x, y in tax2vals_d.items():
            if all(a < cutoff for a in y) and x != 'Other':
                other_count += 1
                try:
                    tax2vals_d['Other'] += y
                except ValueError:
                    tax2vals_d.update({'Other': y})
                to_del.append(x)
        if all(a == 0.0 for a in tax2vals_d['Other']):
            to_del.append('Other')
        else:
            tax2vals_d['Other (num=%d, cutoff=%g)' % (other_count, cutoff)] = tax2vals_d.pop(
                'Other')

        for key in to_del:
            if key in tax2vals_d:
                del tax2vals_d[key]

        return tax2vals_d

    def compute_the_dict(self, cutoff):
        """
        Calls methods to analyze data and graph. Determines whether to graph with grouping or not.
        :param level:
        :param cutoff:
        :return:
        """
        logging.info('graph_this(cutoff={})'.format(cutoff))
        for level, rank in enumerate(RANKS):
            tax2vals_d = self.get_tax2vals_d(level)
            tax2vals_d = self.percentize_cutoff_tax2vals_d(tax2vals_d, cutoff)
            self.the_dict[rank] = tax2vals_d

    def make_grp2inds_d(self, category_field_name):
        """
        returns a dictionary with keys being the different categories for grouping and values
        being lists of index numbers, the numbers correlate to the location the sample has in
        self.samples. These lists of numbers will be used to graph elements of the array values
        in tax2vals_d in the order of group. So like elements pertaining to group1 are first then
        groups2.. etc.
        :param category_field_name:
        :return: grp2inds_d
        """
        logging.info('Making group dictionary for metadata column: {}'.format(category_field_name))

        grp2inds_d = {}
        metadata_samples = list(self.sample2group_df.iloc[:, 0])
        metadata_grouping = list(self.sample2group_df.iloc[:, 1])
        for category, sample in zip(metadata_grouping, metadata_samples):
            try:
                grp2inds_d[category].append(self.samples.index(sample))
            except KeyError:
                grp2inds_d.update({category: [self.samples.index(sample)]})
        return grp2inds_d

    def graph(self, category_field_name):
        """
        """
        logging.info('Graphing by group. category: {}'.format(category_field_name))

        if not category_field_name:
            grp2inds_d = {'': list(range(len(self.samples)))}
            num_grps = 1
        else:
            grp2inds_d = self.make_grp2inds_d(category_field_name)
            dprint('grp2inds_d', run=locals(), where=True)
            num_grps = len(grp2inds_d)

        taxo_fig = make_subplots(
            rows=1,
            cols=num_grps,
            horizontal_spacing=0.05,
            x_title="Sample" + ("" if category_field_name else "<br>Grouped by: %s" % category_field_name),
            subplot_titles=list(grp2inds_d.keys()),
            column_widths=[len(inds) for inds in grp2inds_d.values()],  # TODO account for horizontal_space and bargap
        )

        start_rank = 'Class'
        start_level = 2

        for rank, tax2vals_d in self.the_dict.items():
            color_iter = itertools.cycle(px.colors.qualitative.Alphabet)  # TODO choose color
            for taxo_str, vals in tax2vals_d.items():
                marker_color = next(color_iter)
                for col, (grp, grp_inds) in zip(range(1, len(grp2inds_d) + 1), grp2inds_d.items()):
                    plot_x = []
                    plot_y = []
                    for i in grp_inds:
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
                            visible=True if rank == start_rank else False,
                        ),
                        row=1,
                        col=col,
                    )

        taxo_fig.update_layout(
            barmode='stack',
            bargap=0.03,
            legend_traceorder='reversed',
            title_text='Rank: %s' % start_rank,
            title_y=0.94 if category_field_name else 0.97,
            title_x=0.5,
            title_yref='container',
            title_xref='paper',
            yaxis_title='Proportion',
            yaxis_range=[0, 1],
            xaxis_tickangle=45,
            margin=dict(b=115),  # since xaxis title is annotation, needs to be lowered, liable to fall off
        )

        # lower x_title
        taxo_fig.layout.annotations[-1].update(
            dict(
                y=-0.05 if category_field_name else -0.04,
            )
        )

        # update axes here
        # to affect all subplots
        taxo_fig.update_yaxes(range=[0, 1])
        taxo_fig.update_xaxes(tickangle=45)

        # number traces per rank
        num_taxonomy = [len(tax2vals_d) for tax2vals_d in self.the_dict.values()]
        num_traces = [len(tax2vals_d) * num_grps for tax2vals_d in self.the_dict.values()]

        dprint('num_grps', 'num_taxonomy', 'num_traces', run=locals(), json=False)

        dropdown_y = 1.05 if category_field_name else 1.10

        def get_vis_mask(rank_ind, select):
            """For toggling trace visibilities when selecting rank"""
            mask = []
            for i in range(len(RANKS)):
                if i != rank_ind:
                    mask += [False] * num_traces[i]
                elif select == 'trace':
                    mask += [True] * num_traces[i]
                elif select == 'legend':
                    mask += ([True] + [False] * (num_grps - 1)) * num_taxonomy[i]
                else:
                    raise Exception()
            return mask

        buttons = [
            dict(
                args=[
                    {
                        'visible': get_vis_mask(i, 'trace'),
                        'showlegend': get_vis_mask(i, 'legend')
                    },
                    {
                        'title': 'Rank: %s' % rank
                    }
                ],
                label=rank,
                method='update'
            )
            for i, rank in enumerate(RANKS)
        ]

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
                    yanchor="bottom"
                ),
                dict(
                    buttons=buttons,
                    active=start_level,
                    direction='down',
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.15,
                    xanchor="left",
                    y=dropdown_y,
                    yanchor="bottom"
                )
            ],
        )

        dprint(
            'taxo_fig.layout.annotations',
            'len(taxo_fig.layout.annotations)',
            run=locals()
        )

        # write plotly html

        plotly_html_flpth = os.path.join(self.run_dir, "plotly_bar.html")
        plot(taxo_fig, filename=plotly_html_flpth)

        return {
            'path': plotly_html_flpth,
            'name': 'plotly_bar.html',
        }


def get_id2taxonomy(attributes, instances, tax_field):
    for ind, d in enumerate(attributes):
        if d['attribute'] == tax_field:
            break
    #
    id2taxonomy = {id: instance[ind] for id, instance in instances.items()}
    return id2taxonomy


# Methods that retrieve KBase data from Matrixs and Mappings ###
def get_df(amp_data, tax_field, dfu, associated_matrix_obj_ref=None, associated_matrix_row=None,
           ascending=1):
    """
    Get Amplicon Matrix Data then make Pandas.DataFrame(),
    also get taxonomy data and add it to df, then transpose and return
    :param amp_permanent_id:
    :param dfu:
    :return:
    """
    logging.info('Getting DataObject')
    # Amplicon data

    row_ids = amp_data['data']['row_ids']
    col_ids = amp_data['data']['col_ids']
    values = amp_data['data']['values']
    # Add 'taxonomy' column

    df_col_ids = copy.deepcopy(col_ids)
    df_col_ids.append('taxonomy')
    # Make pandas DataFrame
    df = pd.DataFrame(index=row_ids, columns=df_col_ids)
    for i in range(len(row_ids)):
        df.iloc[i, :-1] = values[i]

    # Get object
    test_row_attributes_permanent_id = amp_data['row_attributemapping_ref']
    obj = dfu.get_objects({'object_refs': [test_row_attributes_permanent_id]})
    # row_attrmap_name = obj['data'][0]['info'][1]
    attributes = obj['data'][0]['data']['attributes']
    instances = obj['data'][0]['data']['instances']

    # Get id2taxonomy dict
    id2taxonomy = get_id2taxonomy(attributes, instances, tax_field)

    # Add taxonomy data and transpose matrix
    for row_ind in df.index:
        df.loc[row_ind]['taxonomy'] = id2taxonomy[row_ind]

    # order samples by associated matrix row data
    if associated_matrix_row is not None:
        logging.info('Start reordering matrix')
        asso_matrix_obj = dfu.get_objects({
            'object_refs': [associated_matrix_obj_ref]})['data'][0]['data']
        asso_matrix_data = asso_matrix_obj['data']

        asso_row_ids = asso_matrix_data['row_ids']
        asso_col_ids = asso_matrix_data['col_ids']
        asso_values = asso_matrix_data['values']

        asso_matrix_df = pd.DataFrame(asso_values, index=asso_row_ids, columns=asso_col_ids)

        try:
            asso_matrix_df = asso_matrix_df[col_ids]
        except KeyError as e:
            err_msg = 'Some samples are not found in the associated matrix\n{}'.format(e)
            raise ValueError(err_msg)

        asso_matrix_df.sort_values(associated_matrix_row, axis=1, ascending=ascending,
                                   inplace=True)

        reordered_columns = copy.deepcopy(list(asso_matrix_df.columns))
        reordered_columns.append('taxonomy')

        df = df[reordered_columns]

    df = df.T

    return df


def get_sample2group_df(col_attrmap_ref, category_name, dfu, order):
    """
    Metadata: make range(len()) index matrix with ID and Category columns
    :param col_attrmap_ref:
    :param category_name:
    :param dfu:
    :return:
    """
    logging.info('Getting MetadataObject')
    # Get object
    obj = dfu.get_objects({'object_refs': [col_attrmap_ref]})
    meta_d = obj['data'][0]['data']['instances']
    attr_l = obj['data'][0]['data']['attributes']

    # Find index of specified category name
    ind = 0
    for i in range(len(attr_l)):
        if attr_l[i]['attribute'] == category_name:
            ind = i
            break
    # Set metadata_samples
    metadata_samples = meta_d.keys()
    # Make pandas DataFrame
    sample2group_df = pd.DataFrame(index=range(len(metadata_samples)),
                                   columns=['ID', category_name])

    for idx, sample_name in enumerate(order[:-1]):
        value = meta_d[sample_name][ind]
        sample2group_df.iloc[idx] = [sample_name, value]

    dprint('sample2group_df', run=locals())

    return sample2group_df
# End of KBase data retrieving methods ###


def run(amp_id, tax_field, grouping_label, cutoff, dfu, scratch, associated_matrix_obj_ref=None,
        associated_matrix_row=None, ascending=1):
    """
    First method that is ran. Makes instance of GraphData class.
    Determines whether to get metadata or not.

    :param amp_id:
    :param tax_field:
    :param cutoff:
    :param grouping_label:
    :param callback_url:
    :param scratch:
    :return:
    """
    logging.info('run(grouping: {}, cutoff: {})'.format(grouping_label, cutoff))

    matrix_obj = dfu.get_objects({'object_refs': [amp_id]})['data'][0]['data']

    # transpose of AmpMat df with taxonomy col appended
    df = get_df(matrix_obj, tax_field, dfu, associated_matrix_obj_ref=associated_matrix_obj_ref,
                associated_matrix_row=associated_matrix_row, ascending=ascending)
    if grouping_label:
        sample2group_df = get_sample2group_df(
            col_attrmap_ref=matrix_obj.get('col_attributemapping_ref'),
            category_name=grouping_label,
            dfu=dfu,
            order=list(df.index))  # df of sample to group
    else:
        sample2group_df = None
    return GraphData(df=df, sample2group_df=sample2group_df,
                     cutoff=cutoff, scratch=scratch).graph(category_field_name=grouping_label)
