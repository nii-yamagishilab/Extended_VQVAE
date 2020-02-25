import numpy as np
import glob
import os
files = glob.glob('/home/smg/zhaoyi/projects/neural_vocoder/data/nancy/mels-tacotron/mel-shinji/test100denorm/*.mfbsp')
save_path = '/home/smg/zhaoyi/projects/neural_vocoder/data/nancy/mels-tacotron/mel-shinji/test100denorm_npy/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

datatype = np.dtype(('<f4',(80,)))
for file in files:
    base = os.path.basename(file).split('.')[0]
    f = open(file)
    data = np.fromfile(f, dtype = datatype)
    f.close()
    save_file = os.path.join(save_path, base + '.npy')
    np.save(save_file, data.T)
