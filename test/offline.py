'''
For testing outside of Docker container
Filepaths are relative
'''
import os
import sys
import unittest
from unittest.mock import patch
import uuid
import logging

# relative paths
cwd = os.path.dirname(os.path.abspath(__file__))
testData_dir = os.path.join(cwd, '../../test/data')
test_dir = os.path.join(cwd, '../../test')
lib_dir = os.path.join(cwd, '../../lib')

# sys.path
sys.path.append(test_dir)
sys.path.append(lib_dir)

import mocks
from mocks import *
import TAUtils
from dprint import dprint

# make paths in other files relative
mocks.testData_dir = testData_dir

logging.basicConfig(format='%(created)s %(levelname)s: %(message)s',
                    level=logging.INFO)


class Test(unittest.TestCase):

    @patch('TAUtils.DataFileUtil', new=lambda callback_url: get_mock_dfu('moss-amp_standardizedTax'))
    def test_TAUtils(self):
        '''

        '''
        scratch = os.path.join(cwd, '../../test_local/workdir/tmp', 'test_TAUTils_' + str(uuid.uuid4()))
        os.mkdir(scratch)

        TAUtils.run(
            moss_amp_AmpMat, 
            moss_amp_colAttrMap, 
            'Field name (informal classification)',
            0.005,
            'dummy.com',
            scratch
        )



