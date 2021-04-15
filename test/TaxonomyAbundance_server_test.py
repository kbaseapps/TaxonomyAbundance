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
from TaxonomyAbundance import TAUtils
from mocks import * # mocks, upas ...


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
        if hasattr(cls, 'wsName'):
            cls.wsClient.delete_workspace({'workspace': cls.wsName})
            print('Test workspace was deleted')

    def shortDescription(self):
        return None

    @patch('TaxonomyAbundance.TaxonomyAbundanceImpl.DataFileUtil', new=lambda *a, **k: mock_dfu)
    @patch('TaxonomyAbundance.TaxonomyAbundanceImpl.KBaseReport', new=lambda *a, **k: mock_kbr)
    def test_with_col_attr_map(self):
        # with grouping
        ret = self.serviceImpl.run_TaxonomyAbundance(
            self.ctx, {
                'workspace_name': self.wsName,
                'amplicon_matrix_ref': moss_AmpMat,
                'tax_field': 'taxonomy',
                'threshold': 0.005,
                'meta_group': 'Field name (informal classification)',
            })

        # without grouping
        ret = self.serviceImpl.run_TaxonomyAbundance(
            self.ctx, {
                'workspace_name': self.wsName,
                'amplicon_matrix_ref': moss_AmpMat,
                'tax_field': 'taxonomy',
                'threshold': 0.005,
                'meta_group': '',
            })

    @patch('TaxonomyAbundance.TaxonomyAbundanceImpl.DataFileUtil', new=lambda *a, **k: mock_dfu)
    @patch('TaxonomyAbundance.TaxonomyAbundanceImpl.KBaseReport', new=lambda *a, **k: mock_kbr)
    def test_no_col_attr_map(self):
        # without grouping
        ret = self.serviceImpl.run_TaxonomyAbundance(
            self.ctx, {
                'workspace_name': self.wsName,
                'amplicon_matrix_ref': Test_amp_mat,
                'tax_field': 'parsed_user_taxonomy',
                'threshold': 0.005,
                'meta_group': '',
            })

    def test_unmocked(self):
        ret = self.serviceImpl.run_TaxonomyAbundance(
            self.ctx, {
                'workspace_name': self.wsName,
                'amplicon_matrix_ref': moss_AmpMat,
                'tax_field': 'taxonomy',
                'threshold': 0.005,
                'meta_group': '',
            })
 

