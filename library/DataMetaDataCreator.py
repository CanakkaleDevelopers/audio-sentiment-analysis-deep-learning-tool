import pandas as pd
import os
from pathlib import Path


class MetaDataCreator:
    def __init__(self, path_dict):
        self.emoDB_path = path_dict['EMODB_FILES_PATH']
        self.ravdess_path = path_dict['RAVDESS_FILES_PATH']
        self.savee_path = path_dict['SAVEE_FILES_PATH']
        self.crema_d = path_dict['CREMA_D_FILES_PATH']
        self.data_metadata_file_path = path_dict['SAVE_RUNTIME_FEATURES']

    def save_to_metadata_table(self, df):
        save_path = os.path.join(self.data_metadata_file_path, 'metadata_table.csv')

        if os.path.exists(save_path):
            print("Veriseti metadata_table.csv dosyasına eklendi")
            datatable = pd.read_csv(save_path)
            datatable = pd.concat([df, datatable], ignore_index=True)
            datatable = datatable.loc[:, ~datatable.columns.str.contains('^Unnamed')]
            datatable.to_csv(save_path)
        else:
            print("Halihazirda olusmus metadata_table.csv dosyasının üzerine yazıldı")
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.to_csv(save_path)

    """Data Explorerin amacı farklı veritabanlarından istediğimiz kadar veriyi özellikleri ve dosya yolları ile
    bir dataframe'e dökmek ve verisetleri arasındaki uyuşmazlık ve farklıları ortadan kaldırmaktır."""

    def ravdess_to_datatable(self):
        print("RAVDESS")
        dir_path = self.ravdess_path
        dir_list = os.listdir(self.ravdess_path)
        emotion = []
        gender = []
        path = []
        for i in dir_list:
            fname = os.listdir(os.path.join(dir_path, i))
            for f in fname:
                part = f.split('.')[0].split('-')
                emotion.append(int(part[2]))
                temp = int(part[6])
                if temp % 2 == 0:
                    temp = "female"
                else:
                    temp = "male"
                gender.append(temp)
                path.append(self.ravdess_path + i + '/' + f)

            RAV_df = pd.DataFrame(emotion)
            RAV_df = RAV_df.replace(
                {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'})
            RAV_df = pd.concat([pd.DataFrame(gender), RAV_df], axis=1)
            RAV_df.columns = ['gender', 'emotion']
            RAV_df['gender'] = RAV_df.gender
            RAV_df['emotion'] = RAV_df.emotion
            RAV_df['source'] = 'Ravdess'
            RAV_df = pd.concat([RAV_df, pd.DataFrame(path, columns=['path'])], axis=1)

        self.save_to_metadata_table(RAV_df)

    def cremad_to_datatable(self):
        print("Crema-D")
        CREMA_D_F_PATH = self.crema_d
        dir_list = os.listdir(CREMA_D_F_PATH)
        dir_list.sort()
        print(dir_list[0:10])
        gender = []
        emotion = []
        path = []
        female = [1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018, 1020, 1021, 1024, 1025, 1028, 1029,
                  1030, 1037, 1043, 1046, 1047, 1049,
                  1052, 1053, 1054, 1055, 1056, 1058, 1060, 1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079, 1082,
                  1084, 1089, 1091]

        for i in dir_list:
            part = i.split('_')
            if int(part[0]) in female:
                temp = 'female'
            else:
                temp = 'male'
            gender.append(temp)
            if part[2] == 'SAD':
                emotion.append('sad')
            elif part[2] == 'ANG':
                emotion.append('angry')
            elif part[2] == 'DIS':
                emotion.append('disgust')
            elif part[2] == 'FEA':
                emotion.append('fear')
            elif part[2] == 'HAP':
                emotion.append('happy')
            elif part[2] == 'NEU':
                emotion.append('neutral')
            else:
                emotion.append('unknown')
            path.append(os.path.join(CREMA_D_F_PATH, i))
        crema_df = pd.DataFrame(gender, columns=['gender'])
        crema_df = pd.concat([crema_df, pd.DataFrame(emotion, columns=['emotion'])], axis=1)
        crema_df['source'] = 'Crema-D'
        crema_df = pd.concat([crema_df, pd.DataFrame(path, columns=['path'])], axis=1)
        self.save_to_metadata_table(crema_df)

    def savee_to_datatable(self):
        print("SAVEE")
        # Get the data location for SAVEE
        SAVEE_PATH = self.savee_path
        dir_list = os.listdir(SAVEE_PATH)

        # parse the filename to get the emotions
        emotion = []
        path = []
        for i in dir_list:
            if i[-8:-6] == '_a':
                emotion.append('angry')
            elif i[-8:-6] == '_d':
                emotion.append('disgust')
            elif i[-8:-6] == '_f':
                emotion.append('fear')
            elif i[-8:-6] == '_h':
                emotion.append('happy')
            elif i[-8:-6] == '_n':
                emotion.append('neutral')
            elif i[-8:-6] == 'sa':
                emotion.append('sad')
            elif i[-8:-6] == 'su':
                emotion.append('surprise')
            else:
                emotion.append('unknown')
            path.append(os.path.join(SAVEE_PATH, i))

        # Now check out the label count distribution
        savee_df = pd.DataFrame(emotion, columns=['emotion'])
        savee_df['gender'] = 'male'
        savee_df['source'] = 'SAVEE'
        savee_df = pd.concat([savee_df, pd.DataFrame(path, columns=['path'])], axis=1)

        self.save_to_metadata_table(savee_df)

    def emodb_to_datatable(self):
        print("EMODB")
        EMODB_PATH = self.emoDB_path
        gender = []
        emotion = []
        path = []

        for root, dirs, files in os.walk(EMODB_PATH):
            for name in files:
                if name[0:2] in '0310111215':  # o zaman bu bir erkek
                    gender.append("male")

                    if name[5] == 'W':  # Ärger (Wut) -> Angry
                        emotion.append('angry')
                    elif name[5] == 'L':  # Langeweile -> Boredom
                        emotion.append('bored')
                    elif name[5] == 'E':  # Ekel -> Disgusted
                        emotion.append('disgust')
                    elif name[5] == 'A':  # Angst -> Angry
                        emotion.append('fear')
                    elif name[5] == 'F':  # Freude -> Happiness
                        emotion.append('happy')
                    elif name[5] == 'T':  # Trauer -> Sadness
                        emotion.append('sad')
                    elif name[5] == 'N':
                        emotion.append('neutral')
                    else:
                        emotion.append('unknown')
                else:
                    gender.append("female")
                    if name[5] == 'W':  # Ärger (Wut) -> Angry
                        emotion.append('angry')
                    elif name[5] == 'L':  # Langeweile -> Boredom
                        emotion.append('bored')
                    elif name[5] == 'E':  # Ekel -> Disgusted
                        emotion.append('disgust')
                    elif name[5] == 'A':  # Angst -> Angry
                        emotion.append('fear')
                    elif name[5] == 'F':  # Freude -> Happiness
                        emotion.append('happy')
                    elif name[5] == 'T':  # Trauer -> Sadness
                        emotion.append('sad')
                    elif name[5] == 'N':
                        emotion.append('neutral')
                    else:
                        emotion.append('unknown')

                path.append(os.path.join(EMODB_PATH, name))

        emodb_df = pd.DataFrame(gender, columns=['gender'])
        emodb_df = pd.concat([emodb_df, pd.DataFrame(emotion, columns=['emotion'])], axis=1)
        emodb_df['source'] = 'emoDB'
        emodb_df = pd.concat([emodb_df, pd.DataFrame(path, columns=['path'])], axis=1)

        self.save_to_metadata_table(emodb_df)

    def create_csv(self, demanded_datasets):

        if 'Crema-D' in demanded_datasets:
            try:
                self.cremad_to_datatable()
            except:
                print("CremaD düzgün çıkartılamamış olabilir.")
        if 'emoDB' in demanded_datasets:
            try:
                self.emodb_to_datatable()
            except:
                print("emoDB düzgün çıkartılamamış olabilir.")
        if 'Ravdess' in demanded_datasets:
            try:
                self.ravdess_to_datatable()
            except:
                print("Ravdess düzgün çıkartılamamış olabilir.")

        if 'SAVEE' in demanded_datasets:
            try:
                self.savee_to_datatable()
            except:
                print("Savee düzgün çıkartılamamış olabilir.")
