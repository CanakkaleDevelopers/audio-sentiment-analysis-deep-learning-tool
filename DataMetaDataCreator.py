import pandas as pd
import Config as conf
import os
from pathlib import Path


class MetaDataCreator:

    @staticmethod
    def save_to_metadata_table(df):
        save_path = conf.Config.FilePathConfig.DATA_METADATA_DF_PATH

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

    @staticmethod
    def ravdess_to_datatable():
        print("RAVDESS")
        dir_list = os.listdir(conf.Config.FilePathConfig.RAVDESS_FILES_PATH)
        dir_list.sort()

        emotion = []
        gender = []
        path = []
        for i in dir_list:
            fname = os.listdir(os.path.join(conf.Config.FilePathConfig.RAVDESS_FILES_PATH, i))
            for f in fname:
                part = f.split('.')[0].split('-')
                emotion.append(int(part[2]))
                temp = int(part[6])
                if temp % 2 == 0:
                    temp = "female"
                else:
                    temp = "male"
                gender.append(temp)
                path.append(conf.Config.FilePathConfig.RAVDESS_FILES_PATH + '/' + i + '/' + f)

            RAV_df = pd.DataFrame(emotion)
            RAV_df = RAV_df.replace(
                {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'})
            RAV_df = pd.concat([pd.DataFrame(gender), RAV_df], axis=1)
            RAV_df.columns = ['gender', 'emotion']
            RAV_df['labels'] = RAV_df.gender + '_' + RAV_df.emotion
            RAV_df['source'] = 'RAVDESS'
            RAV_df = pd.concat([RAV_df, pd.DataFrame(path, columns=['path'])], axis=1)
            RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)

        MetaDataCreator.save_to_metadata_table(RAV_df)

    @staticmethod
    def cremad_to_datatable():
        print("Crema-D")
        CREMA_D_F_PATH = conf.Config.FilePathConfig.CREMA_D_FILES_PATH
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
            if part[2] == 'SAD' and temp == 'male':
                emotion.append('male_sad')
            elif part[2] == 'ANG' and temp == 'male':
                emotion.append('male_angry')
            elif part[2] == 'DIS' and temp == 'male':
                emotion.append('male_disgust')
            elif part[2] == 'FEA' and temp == 'male':
                emotion.append('male_fear')
            elif part[2] == 'HAP' and temp == 'male':
                emotion.append('male_happy')
            elif part[2] == 'NEU' and temp == 'male':
                emotion.append('male_neutral')
            elif part[2] == 'SAD' and temp == 'female':
                emotion.append('female_sad')
            elif part[2] == 'ANG' and temp == 'female':
                emotion.append('female_angry')
            elif part[2] == 'DIS' and temp == 'female':
                emotion.append('female_disgust')
            elif part[2] == 'FEA' and temp == 'female':
                emotion.append('female_fear')
            elif part[2] == 'HAP' and temp == 'female':
                emotion.append('female_happy')
            elif part[2] == 'NEU' and temp == 'female':
                emotion.append('female_neutral')
            else:
                emotion.append('Unknown')
            path.append(os.path.join(CREMA_D_F_PATH, i))

        crema_df = pd.DataFrame(emotion, columns=['labels'])
        crema_df['source'] = 'CREMA'
        crema_df = pd.concat([crema_df, pd.DataFrame(path, columns=['path'])], axis=1)

        MetaDataCreator.save_to_metadata_table(crema_df)

    @staticmethod
    def savee_to_datatable():
        print("SAVEE")
        # Get the data location for SAVEE
        SAVEE_PATH = conf.Config.FilePathConfig.SAVEE_FILES_PATH
        dir_list = os.listdir(SAVEE_PATH)

        # parse the filename to get the emotions
        emotion = []
        path = []
        for i in dir_list:
            if i[-8:-6] == '_a':
                emotion.append('male_angry')
            elif i[-8:-6] == '_d':
                emotion.append('male_disgust')
            elif i[-8:-6] == '_f':
                emotion.append('male_fear')
            elif i[-8:-6] == '_h':
                emotion.append('male_happy')
            elif i[-8:-6] == '_n':
                emotion.append('male_neutral')
            elif i[-8:-6] == 'sa':
                emotion.append('male_sad')
            elif i[-8:-6] == 'su':
                emotion.append('male_surprise')
            else:
                emotion.append('male_error')
            path.append(os.path.join(SAVEE_PATH, i))

        # Now check out the label count distribution
        savee_df = pd.DataFrame(emotion, columns=['labels'])
        savee_df['source'] = 'SAVEE'
        savee_df = pd.concat([savee_df, pd.DataFrame(path, columns=['path'])], axis=1)

        MetaDataCreator.save_to_metadata_table(savee_df)

    @staticmethod
    def emodb_to_datatable():
        print("EMODB")
        EMODB_PATH = conf.Config.FilePathConfig.EMODB_FILES_PATH
        emotion = []
        path = []

        for root, dirs, files in os.walk(EMODB_PATH):
            for name in files:
                if name[0:2] in '0310111215':  # o zaman bu bir erkek
                    if name[5] == 'W':  # Ärger (Wut) -> Angry
                        emotion.append('male_angry')
                    elif name[5] == 'L':  # Langeweile -> Boredom
                        emotion.append('male_bored')
                    elif name[5] == 'E':  # Ekel -> Disgusted
                        emotion.append('male_disgust')
                    elif name[5] == 'A':  # Angst -> Angry
                        emotion.append('male_fear')
                    elif name[5] == 'F':  # Freude -> Happiness
                        emotion.append('male_happy')
                    elif name[5] == 'T':  # Trauer -> Sadness
                        emotion.append('male_sad')
                    else:
                        emotion.append('unknown')
                else:
                    if name[5] == 'W':  # Ärger (Wut) -> Angry
                        emotion.append('female_angry')
                    elif name[5] == 'L':  # Langeweile -> Boredom
                        emotion.append('female_bored')
                    elif name[5] == 'E':  # Ekel -> Disgusted
                        emotion.append('female_disgust')
                    elif name[5] == 'A':  # Angst -> Angry
                        emotion.append('female_fear')
                    elif name[5] == 'F':  # Freude -> Happiness
                        emotion.append('female_happy')
                    elif name[5] == 'T':  # Trauer -> Sadness
                        emotion.append('female_sad')
                    else:
                        emotion.append('unknown')

                path.append(os.path.join(EMODB_PATH, name))

        emodb_df = pd.DataFrame(emotion, columns=['labels'])
        emodb_df['source'] = 'EMODB'
        emodb_df = pd.concat([emodb_df, pd.DataFrame(path, columns=['path'])], axis=1)

        MetaDataCreator.save_to_metadata_table(emodb_df)


