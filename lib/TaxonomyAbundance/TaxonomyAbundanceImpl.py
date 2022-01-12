# -*- coding: utf-8 -*-
#BEGIN_HEADER
import logging
import os
import uuid
from TaxonomyAbundance.TAUtils import run
from installed_clients.KBaseReportClient import KBaseReport
from installed_clients.DataFileUtilClient import DataFileUtil

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
    VERSION = "1.0.0"
    GIT_URL = "git@github.com:Tianhao-Gu/TaxonomyAbundance.git"
    GIT_COMMIT_HASH = "5bcceee1e5ce831c22f7e0a95c38798fe1ba523f"

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
        :param params: instance of type "TaxonomyAbundanceInput" ->
           structure: parameter "workspace_name" of String, parameter
           "workspace_id" of Long, parameter "amplicon_matrix_ref" of String,
           parameter "attri_mapping_ref" of String, parameter
           "associated_matrix_obj_ref" of String, parameter "threshold" of
           Double, parameter "taxonomy_level" of Long, parameter
           "grouping_label" of mapping from String to String, parameter
           "meta_group" of String, parameter "tax_field" of String, parameter
           "associated_matrix_row" of String, parameter "ascending" of Long
        :returns: instance of type "ReportResults" -> structure: parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_TaxonomyAbundance

        logging.info('start run_TaxonomyAbundance with:\n{}'.format(params))

        # extract params
        # do some transformations against narrative ui

        amplicon_matrix_ref = params['amplicon_matrix_ref']
        tax_field = params['tax_field']
        cutoff = params.get('threshold', 0.005)
        grouping_label = params.get('meta_group')
        associated_matrix_row = params.get('associated_matrix_row')
        associated_matrix_obj_ref = params.get('associated_matrix_obj_ref')
        ascending = params.get('ascending', 1)
        self.dfu = DataFileUtil(self.callback_url)

        html_link, warnning = run(amp_id=amplicon_matrix_ref,
                                  tax_field=tax_field, cutoff=cutoff,
                                  grouping_label=grouping_label,
                                  dfu=self.dfu, scratch=self.shared_folder,
                                  associated_matrix_obj_ref=associated_matrix_obj_ref,
                                  associated_matrix_row=associated_matrix_row,
                                  ascending=ascending)

        report_client = KBaseReport(self.callback_url, token=self.token)
        report_name = "TaxonomyAbundance_report_" + str(uuid.uuid4())
        report_info = report_client.create_extended_report({
            'direct_html_link_index': 0,
            'html_links': [html_link],
            'report_object_name': report_name,
            'warnings': [warnning],
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
        return [output]
    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        #END_STATUS
        return [returnVal]
