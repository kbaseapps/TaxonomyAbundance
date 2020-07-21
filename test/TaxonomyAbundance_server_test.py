# -*- coding: utf-8 -*-
import os
import time
import unittest
from unittest.mock import patch
import logging
from configparser import ConfigParser

from TaxonomyAbundance.TaxonomyAbundanceImpl import TaxonomyAbundance
from TaxonomyAbundance.TaxonomyAbundanceServer import MethodContext
from TaxonomyAbundance.authclient import KBaseAuth as _KBaseAuth
from installed_clients.WorkspaceClient import Workspace

from TaxonomyAbundance import TAUtils
from mocks import *


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

    # NOTE: According to Python unittest naming rules test method names should start from 'test'. # noqa
    @patch_('TaxonomyAbundance.TAUtils.DataFileUtil', new=lambda callback_url: get_mock_dfu('moss-amp_standardizedTax'))
    @patch_('TaxonomyAbundance.TaxonomyAbundanceImpl.KBaseReport', new=lambda *args, **kwargs: get_mock_kbr())
    def test_your_method(self):
        # Prepare test objects in workspace if needed using
        # self.getWsClient().save_objects({'workspace': self.getWsName(),
        #                                  'objects': []})
        #
        # Run your method by
        # ret = self.getImpl().your_method(self.getContext(), parameters...)
        #
        # Check returned data with
        # self.assertEqual(ret[...], ...) or other unittest methods

        logging.info('test_your_method')

        ret = self.serviceImpl.run_TaxonomyAbundance(self.ctx, {'amplicon_matrix_ref': '37967/3/2',
                                                                'attri_mapping_ref': '37967/4/1',
                                                                'threshold': 0.005,
                                                                'taxonomy_level': 3,
                                                                'grouping_label': {
                                                                    'meta_group': ['Field name (informal classification)']
                                                                },
                                                                'workspace_name': self.wsName})

        



