from unittest.mock import create_autospec
import os
import sys
from shutil import rmtree, copytree
import logging
import json
from pathlib import Path

from installed_clients.DataFileUtilClient import DataFileUtil
from installed_clients.KBaseReportClient import KBaseReport

from TaxonomyAbundance.debug import dprint
from mocks import *


##################################
##################################
# prod
moss_AmpMat = '67610/5/2' # AmpliconMatrix
moss_colAttrMap = '67610/2/1' # col AttributeMapping
moss_rowAttrMap ='59732/4/1' # row AttributeMapping

Test_amp_mat = '88567/3/1'
Test_amp_mat_row_attributes = '88567/2/1'
##################################
##################################


TEST_DATA_DIR = '/kb/module/test/data'
GET_OBJECTS_DIR = TEST_DATA_DIR + '/get_objects'
WORK_DIR = '/kb/module/work/tmp'
CACHE_DIR = WORK_DIR + '/cache_test_data'

## MOCK DFU ##


def mock_dfu_save_objects(params):
    logging.info('Mocking dfu.save_objects(%s)' % str(params)[:200] + '...' if len(str(params)) > 200 else params)

    return [['mock', 1, 2, 3, 'dfu', 5, 'save_objects']] # UPA made from pos 6/0/4

def mock_dfu_get_objects(params):
    logging.info('Mocking dfu.get_objects(%s)' % params)

    upa = ref_leaf(params['object_refs'][0])
    fp = _glob_upa(GET_OBJECTS_DIR, upa)

    # Download and cache
    if fp is None:
        logging.info('Calling in cache mode `dfu.get_objects`')

        dfu = DataFileUtil(os.environ['SDK_CALLBACK_URL'])
        obj = dfu.get_objects(params)
        fp = os.path.join(
            mkcache(GET_OBJECTS_DIR),
            file_safe_ref(upa) + '_' + obj['data'][0]['info'][1] + '.json'
        )
        with open(fp, 'w') as fh: json.dump(obj, fh)
        return obj

    # Pull from cache
    else:
        with open(fp) as fh:
            obj = json.load(fh)
        return obj
def get_mock_dfu():
    mock_dfu = create_autospec(DataFileUtil, instance=True, spec_set=True)
    mock_dfu.get_objects.side_effect = mock_dfu_get_objects
    return mock_dfu
mock_dfu = get_mock_dfu()


## MOCK KBR ##

def mock_create_extended_report(params):
    logging.info('Mocking `kbr.create_extended_report`')

    return {
        'name': 'kbr_mock_name',
        'ref': 'kbr/mock/ref',
    }

mock_kbr = create_autospec(KBaseReport, instance=True, spec_set=True) 
mock_kbr.create_extended_report.side_effect = mock_create_extended_report


## UTIL ##

def mkcache(dir):
    dir = dir.replace(TEST_DATA_DIR, CACHE_DIR)
    os.makedirs(dir, exist_ok=True)
    return dir

def _glob_upa(data_dir, upa):
    p_l = list(Path(data_dir).glob(file_safe_ref(upa) + '*'))
    if len(p_l) == 0:
        return None
    elif len(p_l) > 1:
        raise Exception(upa)

    src_p = str(p_l[0])

    return src_p

def ref_leaf(ref):
    return ref.split(';')[-1]

def file_safe_ref(ref):
    return ref.replace('/', '.').replace(';', '_')
