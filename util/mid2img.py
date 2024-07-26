from music21 import converter, instrument, note, chord
import sys
import numpy as np
from imageio import imwrite
import os


def extractNote(element):
    return int(element.pitch.ps)

def extractDuration(element):
    return element.duration.quarterLength

def get_notes(notes_to_parse):

    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    durations = []
    notes = []
    start = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            if element.isRest:
                continue

            start.append(element.offset)
            notes.append(extractNote(element))
            durations.append(extractDuration(element))
                
        elif isinstance(element, chord.Chord):
            if element.isRest:
                continue
            for chord_note in element:
                start.append(element.offset)
                durations.append(extractDuration(element))
                notes.append(extractNote(chord_note))

    return {"start":start, "pitch":notes, "dur":durations}


def midi2image(midi_path, save_path, max_repetitions = float("inf"), resolution = 0.25, 
               lowerBoundNote = 21, upperBoundNote = 127, maxSongLength = 100):
    mid = converter.parse(midi_path)
    instruments = instrument.partitionByInstrument(mid)
    data = {}

    try:
        i=0
        for instrument_i in instruments.parts:
            notes_to_parse = instrument_i.recurse()

            notes_data = get_notes(notes_to_parse)
            if len(notes_data["start"]) == 0:
                continue

            if instrument_i.partName is None:
                data["instrument_{}".format(i)] = notes_data
                i+=1
            else:
                data[instrument_i.partName] = notes_data

    except:
        notes_to_parse = mid.flat.notes
        data["instrument_0"] = get_notes(notes_to_parse)

    # Create directory to put the file
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for instrument_name, values in data.items():
        # https://en.wikipedia.org/wiki/Scientific_pitch_notation#Similar_systems

        pitches = values["pitch"]
        durs = values["dur"]
        starts = values["start"]
    
        index = 0
        while index < max_repetitions:
            matrix = np.zeros((upperBoundNote-lowerBoundNote,maxSongLength))

            for dur, start, pitch in zip(durs, starts, pitches):
                dur = int(dur/resolution)
                start = int(start/resolution)

                if not start > index*(maxSongLength+1) or not dur+start < index*maxSongLength:
                    for j in range(start,start+dur):
                        if j - index*maxSongLength >= 0 and j - index*maxSongLength < maxSongLength:
                            matrix[pitch-lowerBoundNote,j - index*maxSongLength] = 255

            if matrix.any(): # If matrix contains no notes (only zeros) don't save it
                save_path = os.path.join(save_path,save_path+f"_{instrument_name}_{index}.png")
                imwrite(save_path,matrix.astype(np.uint8))
                index += 1
            else:
                break
            

if __name__ == "__main__":
    import pandas as pd
    print(os.path.dirname(os.path.abspath(__file__)))
    
    # settings
    maxSongLength = 1500
    rootPath = '/Users/jinghuang/Desktop/mgen'
    relDatasetPath = 'datasets/maestro-v3.0.0'
    os.chdir(os.path.join(rootPath, relDatasetPath)) 
    
    # catalog all midi files
    metadata_df = pd.read_csv('maestro-v3.0.0.csv')
    midi_titles = metadata_df.iloc[:, 1]
    midi_split = map(lambda s: 0 if s=='train' else 1 if s=='validation' else 2, metadata_df.iloc[:, 2])
    midi_filenames = metadata_df.iloc[:, 4]
    #print(*midi_split, sep='\n')
    
            
            
    
    
    # savePath = '/Users/jinghuang/Desktop/mgen/MusicGen/img/Mazurka06-3'

    # if len(sys.argv) >= 3:
    #     max_repetitions = int(sys.argv[2])
    #     midi2image(midi_path, savePath, max_repetitions, maxSongLength=maxSongLength)
    # else:
    #     midi2image(midi_path, savePath, maxSongLength=maxSongLength)
        

"""
MusicGen/util/mid2img.py
in mgen cwd
mid2img:
python MusicGen/util/mid2img.py '/Users/jinghuang/Downloads/mazurka06-3.mid'

img2mid:

python MusicGen/util/img2mid.py 'MusicGen/img/mazurka06-1/mazurka06-1_Piano_0.png' 
"""