import netCDF4 as nc
import numpy as np
import pytorch_lightning as pl
import torch
import webdataset as wds
from skimage.measure import block_reduce
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def decode_netcdf(sample):
    '''
    Unpacks two netCDF4 files from a byte stream
    '''
    return nc.Dataset('in-mem-file', mode='r', memory=sample['afternoon.afternoon']), nc.Dataset('in-mem-file',
                                                                                                 mode='r',
                                                                                                 memory=sample[
                                                                                                     'afternoon.night'])


class NetcdfDataset(pl.LightningDataModule):
    def __init__(self, batch_size, bucket, num_workers=0, input_transforms=None, label_transforms=None):
        super().__init__(self)
        self.batch_size, self.num_workers, self.bucket = batch_size, num_workers, bucket
        self.input_transforms, self.label_transforms = input_transforms, label_transforms

        num_shards = len(bucket)
        shard_idx_list = np.arange(num_shards)
        np.random.shuffle(shard_idx_list)
        train_val_test_idcs = np.split(shard_idx_list, [int(num_shards * 0.75), int(
            num_shards * 0.85)])  # Split train (75%), val (10%) and test (15%) set
        self.train_urls, self.val_urls, self.test_urls = [[bucket[idx] for idx in idx_list] for idx_list in
                                                          (train_val_test_idcs)]

    def make_loader(self, urls):
        dataset = (wds.WebDataset(urls)
                   .map(decode_netcdf)
                   .map_tuple(self.input_transforms, self.label_transforms)
                   .batched(self.batch_size, partial=False)
                   )
        loader = wds.WebLoader(dataset, batch_size=None, shuffle=True, num_workers=self.num_workers)
        return loader

    def train_dataloader(self):
        return self.make_loader(urls=self.train_urls)

    def val_dataloader(self):
        return self.make_loader(urls=self.val_urls)

    def test_dataloader(self):
        return self.make_loader(urls=self.test_urls)

    def all_loader(self):
        return self.make_loader(urls=self.bucket)


class TensorDataset(pl.LightningDataModule):
    def __init__(self, batch_size, input_tensor, label_tensor, num_workers=0, input_transforms=None,
                 label_transforms=None, collate_fn=None):
        super().__init__(self)
        self.batch_size, self.collate_fn, self.num_workers = batch_size, collate_fn, num_workers
        self.input_transforms, self.label_transforms = input_transforms, label_transforms
        self.input_tensor, self.label_tensor = input_tensor, label_tensor

        num_idcs = len(label_tensor)
        idx_list = np.arange(num_idcs)
        np.random.shuffle(idx_list)
        train_val_test_idcs = np.split(idx_list, [int(num_idcs * 0.75), int(
            num_idcs * 0.85)])  # Split train (75%), val (10%) and test (15%) set
        self.train_idcs, self.val_idcs, self.test_idcs = [[idx_list[idx] for idx in idcs] for idcs in
                                                          (train_val_test_idcs)]
        self.all_idcs = idx_list

    def make_loader(self, idcs):
        data_set = My_Dataset(self.input_tensor[idcs], self.label_tensor[idcs], self.input_transforms,
                              self.label_transforms)
        return DataLoader(data_set, batch_size=self.batch_size, shuffle=True, drop_last=True,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def train_dataloader(self):
        return self.make_loader(idcs=self.train_idcs)

    def val_dataloader(self):
        return self.make_loader(idcs=self.val_idcs)

    def test_dataloader(self):
        return self.make_loader(idcs=self.test_idcs)

    def all_loader(self):
        return self.make_loader(idcs=self.all_idcs)


class My_Dataset(Dataset):
    def __init__(self, input_tensor, label_tensor, input_transforms=None, label_transforms=None):
        super().__init__()
        self.input_tensor, self.label_tensor = input_tensor, label_tensor
        self.input_transforms, self.label_transforms = input_transforms, label_transforms

    def __len__(self):
        return len(self.label_tensor)

    def __getitem__(self, idx):
        x, y = self.input_tensor[idx], self.label_tensor[idx]
        if self.input_transforms:
            x = self.input_transforms(x)
        if self.label_transforms:
            y = self.label_transforms(y)
        return x, y


class CropFieldsAndSize:
    """
    Need to perform both operations in one action as cropping in size requires the 'longitude' and 'latitude' attributes
    of the netcdf4 dataset, cropping by fields also requires the object to remain netcdf4 dataset.
    Alternatively, a new dataset could be created with the desired fields + longitude and latitude and then pass this
    to a separate crop function for size
    """

    def __init__(self, fields, max_lon=False, min_lat=False, height=840, width=840):
        if type(fields) == str:
            fields = [fields]
        self.fields, self.max_lon, self.min_lat, self.height, self.width = fields, max_lon, min_lat, height, width

    def __call__(self, netcdf4_dataset):
        check = all([type(x) in (int, float) for x in (self.min_lat, self.max_lon)])
        if check:
            lon_idx = np.argmin(np.abs(netcdf4_dataset.variables['longitude'][:] - self.max_lon))
            lat_idx = np.argmin(np.abs(netcdf4_dataset.variables['latitude'][:] - self.min_lat))
        result = torch.zeros((len(self.fields), netcdf4_dataset.dimensions['record'].size, self.height, self.width))
        for idx, field in enumerate(self.fields):
            masked_array = netcdf4_dataset[field][:].squeeze()
            if check:
                masked_array = masked_array[:, lat_idx:lat_idx + self.height, lon_idx:lon_idx + self.width]
            result[idx] = torch.FloatTensor(masked_array.filled())
        return result.squeeze()


class SumDim:
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, sample):
        return torch.sum(sample, dim=self.dim)


class MaxMapsIntoSingleMap:
    def __call__(self, sample):
        return torch.max(sample, dim=0)


class CombineProbabilityMapsIntoSingleMap:
    def __call__(self, sample):
        no_storm_prob = torch.ones_like(sample) - sample
        no_storm_prob = torch.prod(no_storm_prob, dim=0)
        return torch.ones_like(no_storm_prob) - no_storm_prob


class Pool:
    def __init__(self, pooled=0):
        self.pooled = pooled

    def __call__(self, sample):
        if self.pooled:
            if len(sample.shape) == 2:
                sample = torch.Tensor(block_reduce(sample, (2 ** self.pooled, 2 ** self.pooled), np.average))
            elif len(sample.shape) == 3:
                sample = torch.Tensor(block_reduce(sample, (1, 2 ** self.pooled, 2 ** self.pooled), np.average))
        return sample


class Mask:
    def __init__(self, mask, pooled=0):
        self.pooled = pooled
        self.mask = mask
        if self.pooled:
            self.mask = torch.Tensor(block_reduce(self.mask, (2 ** self.pooled, 2 ** self.pooled), np.max)).bool()

    def __call__(self, sample):
        if len(sample.shape) == 3:
            channels = sample.shape[0]
            return torch.masked_select(sample, self.mask).view(channels, -1)
        return sample[self.mask]


class Reconstruct2D:
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, sample):
        return torch.zeros_like(self.mask, dtype=sample.dtype).masked_scatter_(self.mask, sample)


class SumAll:
    def __call__(self, sample):
        return torch.sum(sample)


class BinarizeValues:
    def __call__(self, sample):
        return torch.where(sample > 1, torch.ones_like(sample), torch.zeros_like(sample))


class ToTensor:
    def __init__(self, field):
        self.field = field

    def __call__(self, sample):
        return torch.FloatTensor(sample[self.field][:].filled().squeeze())


class SelectTimes:
    def __init__(self, start, end=False):
        self.start, self.end = start, end

    def __call__(self, sample):
        if self.end:
            return sample[self.start:self.end]
        return sample[self.start::]


class ToExtreme:
    def __call__(self, sample):
        return torch.FloatTensor([int(float(sample) > 5379)])


class ToLong:
    def __call__(self, sample):
        return sample.long()


class ToFloat:
    def __call__(self, sample):
        return sample.float()


class ApplyFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, sample):
        return torch.Tensor(self.function(sample))
