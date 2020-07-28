from unittest.mock import create_autospec
import os
import sys
from shutil import rmtree, copytree
import logging
import json

from installed_clients.DataFileUtilClient import DataFileUtil
from installed_clients.KBaseReportClient import KBaseReport

from TaxonomyAbundance.dprint import dprint


##################################
##################################
testData_dir = '/kb/module/test/data'
##################################
##################################
# prod
moss_amp_AmpMat = '67610/5/2' # AmpliconMatrix
moss_amp_colAttrMap = '67610/2/1' # col AttributeMapping
moss_amp_rowAttrMap ='59732/4/1' # row AttributeMapping

# CI
secret_wRDP_AmpMat = '49926/5/12' # AmpliconMatrix. With row AttributeMapping with taxonomy. Do not share
##################################
##################################


def get_mock_dfu(dataset):
    '''
    Avoid lengthy `get_objects` and `save_objects`
    '''
    # validate
    if dataset not in ['moss-amp', 'moss-amp_standardizedTax']:
        raise NotImplementedError('Input dataset `%s` not recognized' % dataset)

    mock_dfu = create_autospec(DataFileUtil, instance=True)

    ##
    ## mock `save_objects`
    def mock_dfu_save_objects(params):
        params_str = str(params)
        if len(params_str) > 100: params_str = params_str[:100] + ' ...'
        logging.info('Mocking `dfu.save_objects` with `params=%s`' % params_str)

        return [['-1111', 1, 2, 3, '-1111', 5, '-1111']] # UPA made from pos 6/0/4
    
    mock_dfu.save_objects.side_effect = mock_dfu_save_objects

    ##
    ## mock `get_objects`
    def mock_dfu_get_objects(params):
        logging.info('Mocking `dfu.get_objects` with `params=%s`' % str(params))

        upa = params['object_refs'][0]
        flnm = {
            moss_amp_AmpMat: 'AmpliconMatrix.json',
            moss_amp_colAttrMap: 'col_AttributeMapping.json',
            moss_amp_rowAttrMap: 'row_AttributeMapping.json',
        }[upa]
        flpth = os.path.join(testData_dir, 'by_dataset_input', dataset, 'get_objects', flnm)

        with open(flpth) as f:
            obj = json.load(f)

        return obj

    mock_dfu.get_objects.side_effect = mock_dfu_get_objects

    return mock_dfu



def get_mock_kbr(dataset=None): 
    '''
    Avoid lengthy `create_extended_report`

    Does not use input currently
    '''

    mock_kbr = create_autospec(KBaseReport, instance=True) 

    # mock `create_extended_report`
    def mock_create_extended_report(params):
        logging.info('Mocking `kbr.create_extended_report`')

        return {
            'name': 'kbr_mock_name',
            'ref': 'kbr/mock/ref',
        }

    mock_kbr.create_extended_report.side_effect = mock_create_extended_report
    
    return mock_kbr




__all__ = [
    'moss_amp_AmpMat', 'moss_amp_rowAttrMap', 'moss_amp_colAttrMap', 
    'secret_wRDP_AmpMat',
    'get_mock_dfu', 'get_mock_kbr'
]

