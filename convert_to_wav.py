# coding: utf-8

import os
from config import *

if not os.path.exists(music_dir):
    raise ValueError('music pytdir "{}" not exists'.format(music_dir))

if not os.path.exists(data_dir):
    os.mkdir(data_dir)


def convert_to_wav():
    for in_file_name in os.listdir(music_dir):
        in_file_path = os.path.join(music_dir, in_file_name)
        out_file_name = os.path.splitext(in_file_name)[0] + '.wav'
        out_file_path = os.path.join(data_dir, out_file_name)
        comm = 'sox -b {bits} -r {rate} "{inp}" "{out}" remix {channels}'.format(
            bits=16,
            channels=1,
            rate=fs,
            inp=in_file_path,
            out=out_file_path,
        )
        if os.system(comm):
            print 'FAIL: file "{}"'.format(in_file_path)
        else:
            print 'OK: file "{}"'.format(in_file_path)


if __name__ == '__main__':
    convert_to_wav()
