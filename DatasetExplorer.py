import os
import sys
import urllib
import zipfile


class DatasetExplorer:
    """
    Yeri geldiğinde bir adet nesne oluşturulmalıdır.
    Lütfen bu sınıftan oluşturulacak nesneye dışarıdan müdahalede bulunmayın.
    """
    local_datasets = []
    download_queue = []
    install_queue = []

    def __init__(self, demanded_datasets, path_dict):
        self.to_be_used_datasets = demanded_datasets
        self.path_dict = path_dict

    def scan(self):
        """
        Scans the Datasets folder
        :return:
        """
        try:
            for dataset_folder in os.scandir(
                    self.path_dict['datasets_folder']):  # phase one -> scan local datasets dir
                if not dataset_folder.name.startswith('.') and dataset_folder.is_dir():
                    self.local_datasets.append(dataset_folder.name)
                    print("Local dataset found : ", dataset_folder.name, 'Folder size',
                          self.get_tree_size(
                              os.path.join(self.path_dict['datasets_folder'], dataset_folder.name)) / 10 ** 6,
                          'MB')
            for dataset in self.to_be_used_datasets:
                if dataset not in self.local_datasets:
                    print(dataset, ' verisetinin bilgisayarınızda yüklü olmadığı görüldü. İndirilecek.')
                    self.download_queue.append(dataset)
            print("Eğer bir verisetinin yanlış indirildiğini düşünüyorsanız, "
                  "verisetini silip programı tekrar çalıştırın.")
        except:
            print("Dataset Okuma sırasında bir hata oluşmuş olabilir.")

    def download_datasets(self):

        if len(self.download_queue) == 0:
            print("İstenen verisetleri bilgisayarınızda yüklü.. \nBir sonraki adıma geçiliyor..")
            return
        downloads_path = self.path_dict['downloads_folder']
        datasets_path = self.path_dict['datasets_folder']
        if 'Ravdess' in self.download_queue:
            print('Ravdess indiriliyor..')

            # perform download
            ravdess_downloaded_path = os.path.join(downloads_path, 'Ravdess.zip')
            url = 'https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1'
            urllib.request.urlretrieve(url, ravdess_downloaded_path)

            # perform unzip
            with zipfile.ZipFile(ravdess_downloaded_path, 'r') as zip_ref:
                zip_ref.extractall(datasets_path)

        # TODO-> implement install and unzip datasets

    def get_tree_size(self, path):
        """Verilen dizinin alt dizinleri ile birlikte byte
         cinsinden boyutunu döndürür.
        """
        total = 0
        for entry in os.scandir(path):
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError as error:
                print('Error calling is_dir():', error, file=sys.stderr)
                continue
            if is_dir:
                total += self.get_tree_size(entry.path)
            else:
                try:
                    total += entry.stat(follow_symlinks=False).st_size
                except OSError as error:
                    print('Error calling stat():', error, file=sys.stderr)
        return total



d = DatasetExplorer(demanded_datasets, path_dict)
d.scan()
d.download_datasets()
