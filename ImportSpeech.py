import os
import pandas as pd
import tqdm

Ravdess = "actors/"
Crema = "AudioWAV/"
Tess = "TESS Toronto emotional speech set data/"
Savee = "ALL/"


def import_ravdess():
    ravdess_directory_list = os.listdir(Ravdess)

    file_emotion = []
    file_path = []
    for dir in tqdm.tqdm(ravdess_directory_list):
        # as their are 20 different actors in our previous directory we need to extract files for each actor.
        actor = os.listdir(Ravdess + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            try:
                file_emotion.append(int(part[2]))
                file_path.append(Ravdess + dir + '/' + file)
            except IndexError:
                pass

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    # changing integers to actual emotions.
    Ravdess_df.Emotions.replace(
        {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'},
        inplace=True)
    # Ravdess_df.head()
    print("Ravdess:\n", Ravdess_df.Emotions.value_counts())
    return Ravdess_df


def import_crema():
    crema_directory_list = os.listdir(Crema)

    file_emotion = []
    file_path = []

    for file in tqdm.tqdm(crema_directory_list):
        if '(' in file:
            continue
        # storing file paths
        file_path.append(Crema + file)
        # storing file emotions
        part = file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Crema_df = pd.concat([emotion_df, path_df], axis=1)
    # Crema_df
    print("Crema:\n", Crema_df.Emotions.value_counts())
    return Crema_df


def import_tess():
    tess_directory_list = os.listdir(Tess)

    file_emotion = []
    file_path = []

    for dr in tqdm.tqdm(tess_directory_list):
        directories = os.listdir(Tess + dr)
        for file in directories:
            if '(' in file:
                continue

            part = file.split('.')[0]
            try:
                part = part.split('_')[2]
                if part == 'ps':
                    file_emotion.append('surprise')
                else:
                    file_emotion.append(part)
                file_path.append(Tess + dr + '/' + file)
            except IndexError:
                pass

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    # Tess_df
    print("Tess:\n", Tess_df.Emotions.value_counts())
    return Tess_df


def import_savee():
    savee_directory_list = os.listdir(Savee)

    file_emotion = []
    file_path = []

    for file in tqdm.tqdm(savee_directory_list):
        if '(' in file:
            continue

        file_path.append(Savee + file)
        part = file.split('_')[1]
        ele = part[:-6]
        if ele == 'a':
            file_emotion.append('angry')
        elif ele == 'd':
            file_emotion.append('disgust')
        elif ele == 'f':
            file_emotion.append('fear')
        elif ele == 'h':
            file_emotion.append('happy')
        elif ele == 'n':
            file_emotion.append('neutral')
        elif ele == 'sa':
            file_emotion.append('sad')
        else:
            file_emotion.append('surprise')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Savee_df = pd.concat([emotion_df, path_df], axis=1)
    # Savee_df.head()
    print("Savee:\n", Savee_df.Emotions.value_counts())
    return Savee_df

