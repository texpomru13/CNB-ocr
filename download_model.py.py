import argparse
import os
import sys
import tarfile
import warnings
import zipfile

import requests
import yaml


class GoogleDriveDownloader:
    ''' Minimal class to download shared files from Google Drive. Source: https://github.com/ndrplz/google-drive-downloader '''

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, unzip=True, showsize=False, untar=False):
        ''' Downloads a shared file from google drive into a given folder. Optionally unzips it.

        Parameters
        ----------
        file_id: str
            the file identifier. You can obtain it from the sharable link.
        dest_path: str
            the destination where to save the downloaded file. Must be a path (for example: './downloaded_file.txt')
        overwrite: bool
            optional, if True forces re-download and overwrite.
        unzip: bool
            optional, if True unzips a file. If the file is not a zip file, ignores it.
        showsize: bool
            optional, if True print the current download size.
        Returns
        -------
        None '''

        destination_directory = os.path.dirname(dest_path)
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        if not os.path.exists(dest_path) or overwrite:

            session = requests.Session()

            print('Downloading {} into {}... '.format(file_id, dest_path), end='')
            sys.stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader.__get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            if showsize:
                print()  # Skip to the next line

            current_download_size = [0]
            GoogleDriveDownloader.__save_response_content(response, dest_path, showsize, current_download_size)
            print('Done.')

            if unzip:
                try:
                    print('Unzipping... ', end='')
                    sys.stdout.flush()
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        z.extractall(destination_directory)
                    print('Done.')
                except zipfile.BadZipfile:
                    warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))
                finally:
                    os.remove(dest_path)

            if untar:
                file_tar = tarfile.open(dest_path)
                file_tar.extractall(destination_directory)  # specify which folder to extract to
                file_tar.close()


    @staticmethod
    def __get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        if 'Virus scan warning' in response.text:
            return 't'  # confirm=t - google itself gives this parameter when confirming the download through the browser

        return None

    @staticmethod
    def __save_response_content(response, destination, showsize, current_size):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    if showsize:
                        print('\r' + GoogleDriveDownloader.sizeof_fmt(current_size[0]), end=' ')
                        sys.stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE

    # From https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    @staticmethod
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return '{:.1f} {}{}'.format(num, unit, suffix)
            num /= 1024.0
        return '{:.1f} {}{}'.format(num, 'Yi', suffix)


# Copy from models_manager.py to speed up launch and get rid of dependencies
class ModelsConfig(dict):
    ''' Loading into dictionary .yaml config with the launch parameters of all known trained models.

    1. f_name_models_config - .yaml config file name (default is 'models_config.yaml') '''

    def __init__(self, f_name_models_config=None):
        f_name_models_config = 'models_config.yaml' if not f_name_models_config else f_name_models_config

        with open(f_name_models_config, 'r', encoding='utf-8') as f_models_config:
            models_config_dict = yaml.safe_load(f_models_config)

        super(ModelsConfig, self).__init__(models_config_dict)


MODEL_FILE_NAME_TO_FILE_ID = {
    'CNB_ner.zip': '10rgQlG6v2Bt4rovfv65h9B-7okTS429X',
}


def main():
    default_folder_for_models = 'models/'

    description = "A simple script for downloading the trained model files from Google Drive by its name. Information about models is taken " + \
                  "from 'models_config.yaml'"
    argparser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument('-ow', '--overwrite', dest='overwrite', default=False, action='store_true',
                           help='optional, if True forces re-download and overwrite (default False)')
    argparser.add_argument('-sz', '--showsize', dest='showsize', default=False, action='store_true',
                           help='optional, if True print the current download size (default False)')
    argparser.add_argument('-m', '--model', type=str, metavar='MODEL_NAME', required=True,
                           help='model name to download its files from Google Drive')
    argparser.add_argument('-fm', '--folder_for_models', type=str, default=default_folder_for_models,
                           metavar='FOLDER_NAME',
                           help="folder name to save downloaded model files (default is '{}')".format(
                               default_folder_for_models))
    args = argparser.parse_args()

    # models_config = ModelsConfig()
    model_name = args.model

    # if models_config.get(model_name):
    #     f_name_tacotron_model = models_config[model_name]['f_name_tacotron_model']
    #     f_name_vocoder_model = models_config[model_name]['f_name_vocoder_model']
    # else:
    #     print("[E] Unknown model name: '{}'! Supported: {}".format(model_name, ', '.join(["'{}'".format(name) for name in models_config])))
    #     return

    folder_for_models = args.folder_for_models
    if folder_for_models[-1] != '/':
        folder_for_models += '/'

    model_file_id = MODEL_FILE_NAME_TO_FILE_ID[model_name]

    GoogleDriveDownloader.download_file_from_google_drive(file_id=model_file_id, dest_path=folder_for_models+model_name,
                                                          overwrite=args.overwrite, showsize=args.showsize)


if __name__ == '__main__':
    main()
