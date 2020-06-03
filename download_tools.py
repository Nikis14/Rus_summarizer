"""
This code contains functions for downloading models
"""
import io
import zipfile
import requests
import os
import logging

def download(ref, path):
    """download model by ref, save to path

    Parameters
    ----------
    ref : str
        reference of a source to download model from
    path: str
        folder to save model
    """
    if not os.path.isdir(path):
        #logging.info('Download multilingual_SBERT... Please wait')
        print('Download multilingual_SBERT... Please wait')
        r = requests.get(ref)
        with r, zipfile.ZipFile(io.BytesIO(r.content)) as archive:
            archive.extractall('models')
        #logging.info('Complete!')
        print('Download complete!')

def download_MlSBERT(path):
    """download model MlSBERT

    Parameters
    ----------
    path: str
        folder to save model
    """
    ref = 'https://www.dropbox.com/s/wfx5opc3h46g1qu/multilingual_SBERT.zip?dl=1'
    download(ref, path)


def download_FastText(path):
    """download model FastText

    Parameters
    ----------
    path: str
        folder to save model
    """
    path = "models/ft_native_300_ru_wiki_lenta_lower_case"
    ref = 'http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.bin'
    download(ref, path)