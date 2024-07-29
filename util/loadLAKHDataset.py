import os
import pypianoroll as ppr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import datetime
import random

#################### Settings ####################

# Get directories
rootPath = '/Users/jinghuang/Desktop/mgen'
relDatasetPath = 'datasets/LPD'
datasetPath = os.path.join(rootPath, relDatasetPath)

train_size = 10
test_size = 0

#################### Genre Sorting ####################

genre_file_dir = os.path.join(relDatasetPath, 'meta/msd_tagtraum_cd1.cls')
ids = []
genres = []
with open(genre_file_dir) as f:
    line = f.readline()
    while line:
        if line[0] != '#':
          split = line.strip().split("\t")
          if len(split) == 2:
            ids.append(split[0])
            genres.append(split[1])
          elif len(split) == 3:
            ids.append(split[0])
            ids.append(split[0])
            genres.append(split[1])
            genres.append(split[2])
        line = f.readline()
genre_df = pd.DataFrame(data={"TrackID": ids, "Genre": genres})
genre_dict = genre_df.groupby('TrackID')['Genre'].apply(lambda x: x.tolist()).to_dict()

print(f"{datetime.datetime.now()} Genres metadata loaded.")

#################### Load MIDI Files ####################

"""
MIDI Files (stored in NPZ archives) can be located by their MSD ID
"""

# Util functions to get npz file path from (MSD, NPZ)
def get_midi_npz_path(msd_id, npz_id):
    # MSD ID -> path w/ prefix
    path_prefix = os.path.join(
        msd_id[2], msd_id[3], msd_id[4], msd_id)
    # return path w/ prefix with absolute path
    return os.path.join(
        datasetPath, path_prefix, npz_id + '.npz')

# Cleansed ID file format: npz ID - MSD ID
msd_ids = pd.read_csv(os.path.join(relDatasetPath, 'meta/cleansed_ids.txt'), delimiter = '    ', header = None, engine='python')
msd_to_npz_ids = {a:b for a, b in zip(msd_ids[1], msd_ids[0])}

# Catalog MSD IDs to load
msd_ids_with_genre = [msd_id for msd_id, genres in genre_dict.items() if 'Pop_Rock' in genres and msd_id in msd_to_npz_ids.keys()]
used_ids = random.sample(msd_ids_with_genre, train_size+test_size)


print(f"{datetime.datetime.now()} Pianoroll Million Song Dataset IDS Loaded.")


#################### Create Data Tensors ####################

# Load track file into variables
all_pianorolls = []
program_to_category = {
    'drums': list(range(128, 129)),  # Channel 9 for drums
    'piano': list(range(0, 8)),
    'guitar': list(range(24, 32)),
    'bass': list(range(32, 40)),
    'strings': list(range(40, 52))  # Default for other instruments
}


for msd_id in used_ids:
    #lpd_file_name = msd_to_npz_ids[first_id]
    npz_path = get_midi_npz_path(msd_id, msd_to_npz_ids[msd_id])
    multitrack = ppr.load(npz_path)
    multitrack.set_resolution(2).pad_to_same()

    # Initialize parts
    parts = {name: None for name in ['piano', 'guitar', 'bass', 'strings', 'drums']}
    empty_array = None
    has_empty_parts = False
    print(len(multitrack.tracks))

    for track in multitrack.tracks:
        if track.pianoroll.shape[0] > 0:
            empty_array = np.zeros_like(track.pianoroll, dtype=np.float32)

        # Map program number to category
        for category, programs in program_to_category.items():
            if track.program in programs:
                parts[category] = track.pianoroll.astype(np.float32)

    for key, value in parts.items():
        if value is None:
            parts[key] = empty_array.copy()
            has_empty_parts = True

    # Combine parts into a single tensor
    pianoroll = torch.tensor([
        parts['piano'], parts['guitar'], parts['bass'],
        parts['strings'], parts['drums']
    ], dtype=torch.float32)

    all_pianorolls.append(pianoroll)
print(f"{datetime.datetime.now()} {train_size} + {test_size} pianorolls loaded.")

# Download pianorolls
all_pianorolls = torch.hstack(all_pianorolls) #


