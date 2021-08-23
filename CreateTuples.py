import os
import subprocess
from multiprocessing import Pool
import netCDF4 as nc

timestamps = ['00', '15', '30', '45']

afternoon_timestamps = ['0' + str(x) + y for x in range(4, 10) for y in timestamps] \
                       + [str(x) + y for x in range(10, 18) for y in timestamps]

evening_timestamps = [str(x) + y for x in range(21, 24) for y in timestamps]
night_timestamps = ['0' + str(x) + y for x in range(9) for y in timestamps]
evening_night_timestamps = evening_timestamps + night_timestamps

DATAFOLDER = r"/theia/data/brussel/vo/000/bvo00012/data/dataset/nasalarc_otdetection_lakevictoria"
RESULTFOLDER = r"/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Dataset/tuples"
directory_list = sorted(os.listdir(f"{DATAFOLDER}/"))

def get_data(directory):
    year, day = int(directory[-8:-3]), int(directory[-3:])
    index = directory_list.index(directory)
    next_directory = directory_list[index + 1]
    next_year, next_day = int(next_directory[-8:-3]), int(next_directory[-3:])
    # If the next folder does not contain the data of the next day of the same year
    if day + 1 != next_day:
        # If (it is a non-leap year AND the day is not #365) OR (it is a leap year AND the day is not #366)
        # OR the next year is not the current + 1 OR the next day is not #1, then we'll skip the day
        if ((year % 4) and (day != 365)) or (not (year % 4) and (day != 366)) \
                or (year + 1 != next_year) or (next_day != 1):
            print(f'{year}{day} has no adjacent folder')
            return

    # Generate the names of the afternoon files
    afternoon_files = [f"{DATAFOLDER}/{directory}/{filename}" for filename in
                       sorted(os.listdir(f"{DATAFOLDER}/{directory}/")) if filename[-7:-3] in afternoon_timestamps]

    # Generate the names of the evening files
    night_files = [f"{DATAFOLDER}/{directory}/{filename}" for filename in
                   sorted(os.listdir(f"{DATAFOLDER}/{directory}/"))
                   if filename[-7:-3] in evening_timestamps]

    # Add the names of the night files of the next day to the evening files
    night_files.extend([f"{DATAFOLDER}/{next_directory}/{filename}" for filename in
                        sorted(os.listdir(f"{DATAFOLDER}/{next_directory}/")) if filename[-7:-3] in night_timestamps])

    for file_list in (afternoon_files, night_files):
        for idx, filename in enumerate(file_list):
            duplicate_list = [other_filename for other_filename in file_list[idx + 1::] if
                              other_filename[-7:-3] == filename[-7:-3]]
            for duplicate in duplicate_list:
                file_list.remove(duplicate)

    # Remove afternoon slices with an invalid shape (rare, ~180 slices in total)
    for f in afternoon_files:
        dataset = nc.Dataset(f)
        lon = dataset['longitude'].size
        lat = dataset['latitude'].size
        if (lon, lat) != (840, 840):
            afternoon_files.remove(f)

    # Find which time slices are missing: 
    afternoon_slices_present = [filename[-7:-3] for filename in afternoon_files]
    afternoon_slices_bool = [int(time_slice in afternoon_slices_present) for time_slice in afternoon_timestamps]

    if not any(afternoon_slices_bool[0:2]) or not any(afternoon_slices_bool[-2::]) or '000' in ''.join(
            str(x) for x in afternoon_slices_bool):
        print(f'{year}{day} has too many consecutively missing files in the afternoon')
        return

    # Remove night slices with an invalid shape (rare, ~180 slices in total)    
    for f in night_files:
        dataset = nc.Dataset(f)
        lon = dataset['longitude'].size
        lat = dataset['latitude'].size
        if (lon, lat) != (840, 840):
            night_files.remove(f)

    # Find which time slices are missing:
    night_slices_present = [filename[-7:-3] for filename in night_files]
    night_slices_bool = [int(time_slice in night_slices_present) for time_slice in evening_night_timestamps]

    if not any(night_slices_bool[0:2]) or not any(night_slices_bool[-2::]) or '000' in ''.join(
            str(x) for x in night_slices_bool):
        # If the first or last two are missing or there are three consecutive files missing then we won't use this date
        print(f'{year}{day} has too many consecutively missing files in the night')
        return

    # For the missing files: copy the slice before it if that file wasn't missing itself otherwise the slice after it is copied
    previous_val = False
    for idx, bool_val in enumerate(afternoon_slices_bool):
        if bool_val:
            previous_val = True
        else:
            # -int(previous_val) equals 0 if previous_val is False and equals -1 if previous_val is True
            afternoon_files.insert(idx, afternoon_files[idx - int(previous_val)])
            previous_val = False

    previous_val = False
    for idx, bool_val in enumerate(night_slices_bool):
        if bool_val:
            previous_val = True
        else:
            # Equals -1 (previous) if previous_val is True and equals +1 if previous_val is False
            night_files.insert(idx, night_files[idx - int(previous_val)])
            previous_val = False

    # Names for the two merged files
    afternoon_result = f"{RESULTFOLDER}/{directory[-8:]}.afternoon.nc"
    night_result = f"{RESULTFOLDER}/{directory[-8:]}.night.nc"

    # Concatenate the afternoon_files
    afternoon_code = subprocess.call(f"ncecat -O --no_tmp_fl {' '.join(afternoon_files)} {afternoon_result}",
                                     shell=True)

    # afternoon_code will be 0 (False) if everything went well, otherwise nco returned an error code
    if afternoon_code:
        print(f"{year}{day} returned an error for the afternoon")
        # Remove the erroneous afternoon
        subprocess.call(f"rm {afternoon_result}", shell=True)
        # No need to continue to the night files if the afternoon files failed
        return

    # Concatenate the night files
    night_code = subprocess.call(f"ncecat -O --no_tmp_fl {' '.join(night_files)} {night_result}", shell=True)

    # night_code will be 0 (False) if everything went well, otherwise nco returned an error code
    if night_code:
        print(f"{year}{day} returned an error for the night")
        # Remove the erroneous night result and its afternoon as well
        subprocess.call(f"rm {afternoon_result} {night_result}", shell=True)


if __name__ == '__main__':
    pool = Pool()
    pool.map(get_data, directory_list[:-1])
    pool.close()
