import os
from multiprocessing import Pool
import nctoolkit as nc

nc.options(parallel=True)
nc.options(cores=10)

data_folder = "/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Dataset/tuples/"
result_folder = "/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Dataset/Products/OTsFull/"
# Sorting is not required but allows to estimate the progress during execution by viewing the result_folder
filenames = sorted(os.listdir(data_folder))

def getField(filename):
    ds = nc.open_data(data_folder + filename)  # Open the netCDF4 dataset
    ds.select(variables='ot_probability')  # Select the ot_probability variable
    ds.reduce_dims()  # Reduce redundant dimensions, equals .squeeze() in numpy/torch
    ds.to_nc(result_folder + filename)  # Save the result


if __name__ == '__main__':
    pool = Pool()
    pool.map(getField, filenames)
    pool.close()
