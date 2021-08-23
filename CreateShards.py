import os
import webdataset as wds

dataset_folder = "/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Dataset/"
MAXCOUNT = 10  # Maximum number of samples per shard

def yield_samples():
    for root, dirs, files in os.walk(dataset_folder + "tuples/", topdown=False):
        for fname in files:
            if 'afternoon' in fname:  # For each afternoon file
                afternoon_fpath = os.path.join(root, fname)  # Open the afternoon
                night_fpath = os.path.join(root, fname.replace('afternoon', 'night'))
                with open(afternoon_fpath, 'rb') as afternoon:
                    afternoon_binary = afternoon.read()  # Store the binary afternoon data
                with open(night_fpath, 'rb') as night:
                    night_binary = night.read()  # Store the binary night data
                sample = {
                    "__key__": os.path.splitext(fname)[0],
                    "afternoon": afternoon_binary,
                    "night": night_binary
                }
                yield sample


with wds.ShardWriter(dataset_folder + "shards/shard-%03d.tar", maxcount=MAXCOUNT) as sink:
    for sample in yield_samples():
        sink.write(sample)
