# -*- coding: utf-8 -*-
import os
import time
import unittest
from unittest.mock import patch
import logging
from configparser import ConfigParser

from TaxonomyAbundance.TaxonomyAbundanceServer import MethodContext
from TaxonomyAbundance.authclient import KBaseAuth as _KBaseAuth
from installed_clients.WorkspaceClient import Workspace

from TaxonomyAbundance.TaxonomyAbundanceImpl import TaxonomyAbundance
from TaxonomyAbundance.error import * # custom exceptions
from TaxonomyAbundance import TAUtils
from mocks import * # mocks, upas ...


######################################
######################################
######### TOGGLE PATCH ###############
######################################
###################################### 
do_patch = True # toggle this to turn on/off @patch decorators

if do_patch:
    patch_ = patch
    patch_dict_ = patch.dict

else:
    patch_ = lambda *args, **kwargs: lambda f: f
    patch_dict_ = lambda *args, **kwargs: lambda f: f
######################################
######################################
######################################
######################################


class TaxonomyAbundanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.info('setUpClass')

        token = os.environ.get('KB_AUTH_TOKEN', None)
        config_file = os.environ.get('KB_DEPLOYMENT_CONFIG', None)
        cls.cfg = {}
        config = ConfigParser()
        config.read(config_file)
        for nameval in config.items('TaxonomyAbundance'):
            cls.cfg[nameval[0]] = nameval[1]
        # Getting username from Auth profile for token
        authServiceUrl = cls.cfg['auth-service-url']
        auth_client = _KBaseAuth(authServiceUrl)
        user_id = auth_client.get_user(token)
        # WARNING: don't call any logging methods on the context object,
        # it'll result in a NoneType error
        cls.ctx = MethodContext(None)
        cls.ctx.update({'token': token,
                        'user_id': user_id,
                        'provenance': [
                            {'service': 'TaxonomyAbundance',
                             'method': 'please_never_use_it_in_production',
                             'method_params': []
                             }],
                        'authenticated': 1})
        cls.wsURL = cls.cfg['workspace-url']
        cls.wsClient = Workspace(cls.wsURL)
        cls.serviceImpl = TaxonomyAbundance(cls.cfg)
        cls.scratch = cls.cfg['scratch']
        cls.callback_url = os.environ['SDK_CALLBACK_URL']
        suffix = int(time.time() * 1000)
        cls.wsName = "test_ContigFilter_" + str(suffix)
        ret = cls.wsClient.create_workspace({'workspace': cls.wsName})  # noqa

    @classmethod
    def tearDownClass(cls):

        logging.info('tearDownClass')

        if hasattr(cls, 'wsName'):
            cls.wsClient.delete_workspace({'workspace': cls.wsName})
            print('Test workspace was deleted')

    def shortDescription(self):
        '''Override unittest using test*() docstrings in lieu of test*() method name in output summary'''
        return None

    # TODO try looking up col AttrMap from AmpMat?

    @patch('TaxonomyAbundance.TAUtils.DataFileUtil', new=lambda *a, **k: get_mock_dfu('moss-amp_standardizedTax'))
    @patch_('TaxonomyAbundance.TaxonomyAbundanceImpl.KBaseReport', new=lambda *a, **k: get_mock_kbr())
    def test_local_data(self):
        '''
        Don't un-patch since the `parsed_user_taxonomy` doesn't exist on the remote version
        '''
        logging.info('test_your_method')

        # with grouping
        ret = self.serviceImpl.run_TaxonomyAbundance(
            self.ctx, {
                'workspace_name': self.wsName,
                'amplicon_matrix_ref': moss_amp_AmpMat,
                'attri_mapping_ref': moss_amp_colAttrMap,
                'threshold': 0.005,
                'meta_group': ['Field name (informal classification)'],
            })

        # without grouping
        ret = self.serviceImpl.run_TaxonomyAbundance(
            self.ctx, {
                'workspace_name': self.wsName,
                'amplicon_matrix_ref': moss_amp_AmpMat,
                'attri_mapping_ref': moss_amp_colAttrMap,
                'threshold': 0.005,
                'meta_group': [],
            })


    @patch_('TaxonomyAbundance.TaxonomyAbundanceImpl.KBaseReport', new=lambda *a, **k: get_mock_kbr())
    def test_remote_data(self):
        ret = self.serviceImpl.run_TaxonomyAbundance(
            self.ctx, {
                'workspace_name': self.wsName,
                'amplicon_matrix_ref': secret_wRDP_AmpMat,
                'attri_mapping_ref': None,
                'threshold': 0.005,
                'meta_group': [],
            })
 

    def test_no_taxonomy(self):
        with self.assertRaises(ObjectException) as cm:
            ret = self.serviceImpl.run_TaxonomyAbundance(
                self.ctx, {
                    'workspace_name': self.wsName,
                    'amplicon_matrix_ref': moss_amp_AmpMat,
                    'attri_mapping_ref': moss_amp_colAttrMap,
                    'threshold': 0.005,
                    'meta_group': ['Field name (informal classification)'],
                })
            logging.info(str(cm))


run_tests = ['test_local_data']
local_tests = ['test_local_data']
CI_tests = ['test_remote_data']
prod_tests = ['test_no_taxonomy']


for key, value in TaxonomyAbundanceTest.__dict__.copy().items():
    if type(key) == str and key.startswith('test') and callable(value):
        if key not in run_tests:
            delattr(TaxonomyAbundanceTest, key)
            pass
