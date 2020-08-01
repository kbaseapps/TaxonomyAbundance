# -*- coding: utf-8 -*-
#BEGIN_HEADER
import logging
import os
import uuid
from TaxonomyAbundance.TAUtils import run

from installed_clients.KBaseReportClient import KBaseReport

from .dprint import dprint

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

        logging.info(params)

        # extract params
        # do some transformations against narrative ui
        amplicon_matrix_ref = params.get('amplicon_matrix_ref') # required
        attri_mapping_ref = params.get('attri_mapping_ref') # can be None, 'u/p/a'
        cutoff = params.get('threshold') # required
        grouping_label = params.get('meta_group') if params.get('meta_group') != '' else None # can be: '', 'label' -> None, 'label'

        csv_fp = "/kb/module/data/smalltx.csv"
        xls_fp = "/kb/module/data/moss_f50_metadata.xls"

        html_link = run(amp_id=amplicon_matrix_ref,
                    col_attrmap_ref=attri_mapping_ref, grouping_label=grouping_label, cutoff=cutoff,
                    callback_url=self.callback_url, scratch=self.shared_folder)


        report_client = KBaseReport(self.callback_url, token=self.token)
        report_name = "TaxonomyAbundance_report_" + str(uuid.uuid4())
        report_info = report_client.create_extended_report({
            'direct_html_link_index': 0,
            'html_links': [html_link],
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
