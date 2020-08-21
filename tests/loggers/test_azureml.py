from unittest.mock import patch

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import AzureMlLogger


def test_azureml_logger_exists():
    """ Test launching two independent loggers. """
    logger = AzureMlLogger()
    logger2 = AzureMlLogger()
    assert logger.experiment.id != logger2.experiment.id


def test_azureml_logger_online():
    """ Test Azure ML logger online with mocks. """
    pass


@patch('azureml.core.Run.get_context')
def test_azureml_additional_methods(azureml):
    logger = AzureMlLogger()
