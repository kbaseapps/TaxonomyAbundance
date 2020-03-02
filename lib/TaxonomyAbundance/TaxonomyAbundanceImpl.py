# -*- coding: utf-8 -*-
#BEGIN_HEADER
import logging
import os
import uuid
from TaxonomyAbundance.TAUtils import run

from installed_clients.KBaseReportClient import KBaseReport

#END_HEADER


class TaxonomyAbundance:
    '''
    Module Name:
    TaxonomyAbundance

    Module Description:
    A KBase module: TaxonomyAbundance
    '''

    ######## WARNING FOR GEVENT USERS ####### noqa
    # Since asynchronous IO can lead to methods - even the same method -
    # interrupting each other, you must be *very* careful when using global
    # state. A method could easily clobber the state set by another while
    # the latter method is running.
    ######################################### noqa
    VERSION = "0.0.1"
    GIT_URL = ""
    GIT_COMMIT_HASH = ""

    #BEGIN_CLASS_HEADER
    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.token = os.environ['KB_AUTH_TOKEN']
        self.wsURL = config['workspace-url']
        self.shared_folder = config['scratch']
        logging.basicConfig(format='%(created)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        #END_CONSTRUCTOR
        pass


    def run_TaxonomyAbundance(self, ctx, params):
        """
        This example function accepts any number of parameters and returns results in a KBaseReport
        :param params: instance of mapping from String to unspecified object
        :returns: instance of type "ReportResults" -> structure: parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_TaxonomyAbundance

        amplicon_matrix_ref = params.get('amplicon_matrix_ref')
        test_row_attri_ref = params.get('test_row_attri_ref')
        attri_mapping_ref = params.get('attri_mapping_ref')
        threshold = params.get('threshold')
        taxonomy_level = params.get('taxonomy_level')
        grouping_label = params.get('grouping_label')
        if grouping_label is not None:
            grouping_label = grouping_label['meta_group'][0]

        csv_fp = "/kb/module/data/smalltx.csv"
        xls_fp = "/kb/module/data/moss_f50_metadata.xls"

        img_paths_and_html_paths_dict = run(amp_id=amplicon_matrix_ref, row_attributes_id=test_row_attri_ref, attri_map_id=attri_mapping_ref,
            grouping_label=grouping_label, threshold=threshold, taxonomic_level=taxonomy_level, callback_url=self.callback_url,
            token=self.token, scratch=self.shared_folder)

        file_links = list()
        for path in img_paths_and_html_paths_dict['img_paths']:
            file_links.append({
                'path': path,
                'name': os.path.basename(path),
                'label': "Bar chart",
                'description': "A bar graph without the legend, and another bar graph with the legend."
            })
        report_client = KBaseReport(self.callback_url, token=self.token)
        report_name = "Bar_chart_amplicon_sheet_report_" + str(uuid.uuid4())
        report_info = report_client.create_extended_report({
            'direct_html_link_index': 0,
            'file_links': file_links,
            'html_links': img_paths_and_html_paths_dict['html_paths'],
            'report_object_name': report_name,
            'workspace_name': params['workspace_name']
        })
        output = {
            'report_ref': report_info['ref'],
            'report_name': report_info['name'],
        }

        #END run_TaxonomyAbundance

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method run_TaxonomyAbundance return value ' +
                             'output is not type dict as required.')
        # return the results
        return [report_info]
    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        #END_STATUS
        return [returnVal]
