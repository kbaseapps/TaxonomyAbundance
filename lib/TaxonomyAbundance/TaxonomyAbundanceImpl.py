# -*- coding: utf-8 -*-
#BEGIN_HEADER
import logging
import os
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

        csv_fp = "/kb/module/data/smalltx.csv"
        xls_fp = "/kb/module/data/moss_f50_metadata.xls"
        print(self.shared_folder)
        run(amp_id=amplicon_matrix_ref, row_attributes_id=test_row_attri_ref, attri_map_id=attri_mapping_ref,
            grouping_label=grouping_label, threshold=threshold, taxonomic_level=taxonomy_level, url=self.callback_url)



        report = KBaseReport(self.callback_url)
        report_info = report.create({'report': {'objects_created':[],
                                                'text_message': params['parameter_1']},
                                                'workspace_name': params['workspace_name']})
        output = {
            'report_name': report_info['name'],
            'report_ref': report_info['ref'],
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
