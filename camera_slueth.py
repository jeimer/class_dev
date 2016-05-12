#!/usr/local/bin/env python
# functions to work with site images

import glob, string, datetime, subprocess, images2gif, tempfile
from PIL import Image

def get_image_dirs(camera_num, start_date, end_date):
    '''returns a list of full paths to all images within the specified date range.
    dates should be specified as strings in format e.g. '2016-05-01' for May 1, 2016.'''
    start_parts = string.split(start_date, '-')
    end_parts = string.split(end_date, '-')
    start_date = datetime.date(int(start_parts[0]), int(start_parts[1]),int(start_parts[2]))
    end_date = datetime.date(int(end_parts[0]),int(end_parts[1]), int(end_parts[2]))
    delta = end_date - start_date
    photo_dates = []
    for i in range(delta.days + 1):
        photo_dates += [start_date + datetime.timedelta(days = i)]

    #generate list of directories spanning date range
    day_dirs = []
    for date in photo_dates:
        day_dirs += ['/data/field/{0:%Y}-{0:%m}/{0:%Y}-{0:%m}-{0:%d}/*'.format(date)]
    chunk_dirs = []
    for day_dir in day_dirs:
        chunk_dirs += sorted(glob.glob(day_dir))
    for index in range(len(chunk_dirs)):
        chunk_dirs[index] = chunk_dirs[index] + '/site/images/camera{0}_*'.format(camera_num)
    image_names = []
    for chunk in chunk_dirs:
        image_names += glob.glob(chunk)

    return image_names

def copy_images(camera_num, start_date, end_date, local_dir):
    ''' copies images from start_date to end_date inclusive from camera_num to local_dir
    dates are entered with directory name format, e.g. May 1, 2016 would be 2016-05-01'''
    image_names = get_image_dirs(camera_num, start_date, end_date)

    image_count = 0
    for image_name in image_names:
        subprocess.call('cp {0} {1}/{2}.jpg'.format(image_name, local_dir,image_count), shell=True)
        image_count += 1
    return

def make_mp4(camera_num, start_date, end_date, outfile, reduction_factor, fps):
    ''' creates a gif from the images from start_date to end_date'''
    temp_dir = tempfile.mkdtemp(dir='/home/eimer/')
    copy_images(camera_num, start_date, end_date, temp_dir)
    subprocess.call("ffmpeg -pattern_type glob -i '{0}/*.jpg' -r {1} -c:v libx264 {2}".format(temp_dir, fps, outfile),shell=True)
    return

def make_gif(camera_num, start_date, end_date, outfile, reduction_factor):
    ''' creates a gif from the images from start_date to end_date'''
    image_names = get_image_dirs(camera_num, start_date, end_date)

    #OS limits number of open files, use ulimit to find max number of open files.
    max_open_limit = subprocess.check_output('ulimit -n', shell=True)
    image_count = 0
    images = []
    if len(image_names) > max_open_limit:
        print('Too many images for ulimit. Decrease date range')
        return
    for image_path in image_names:
        full_image = Image.open(image_path)
        width, height = full_image.size
        width = width/reduction_factor
        height = height/reduction_factor
        smaller_image = full_image.resize((width,height), Image.ANTIALIAS)
        images += [smaller_image]
    images2gif.writeGif(outfile, images, duration=0.1)
    return
