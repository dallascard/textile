import os
import gzip
import json
import codecs
import pickle
import numpy as np
import pandas as pd
from scipy import sparse


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_basename_wo_ext(input_filename):
    """ Given a path to a file, return the filename with no path and no extension """
    parts = os.path.split(input_filename)
    # deal with the situation in which we're given a directory (ending with a pathsep)
    if parts[1] == '':
        parts = os.path.split(parts[0])
    basename = os.path.splitext(parts[1])[0]
    return basename


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)


def read_json(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file, encoding='utf-8')
    return data


def read_json_lines(input_filename):
    if input_filename[-3:] == '.gz':
        with gzip.open(input_filename, 'r') as input_file:
            data = {}
            for l_i, line in enumerate(input_file):
                data[l_i] = json.loads(line, encoding='utf-8')
    else:
        with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
            data = {}
            for l_i, line in enumerate(input_file):
                data[l_i] = json.loads(line, encoding='utf-8')
    return data


def pickle_data(data, output_filename):
    with open(output_filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


def unpickle_data(input_filename):
    with open(input_filename, 'rb') as infile:
        data = pickle.load(infile)
    return data


def read_text(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    return lines


def write_list_to_text(lines, output_filename, add_newlines=True, add_final_newline=False):
    if add_newlines:
        lines = '\n'.join(lines)
        if add_final_newline:
            lines += '\n'
    else:
        lines = ''.join(lines)
        if add_final_newline:
            lines[-1] += '\n'

    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.writelines(lines)


def read_csv_to_df(input_filename, header=0, index_col=0, sep=',', encoding='utf-8', parse_dates=False):
    """
    Use pandas to read in a .csv file into a dataframe.
    I should always use this method to ensure consistency in the index, for example
    :param input_filename: filename
    :param header: row of header (-1) if absent
    :param index_col: colum of index
    :param sep: colum separator
    :param encoding: encoding to use for the input file
    :param parse_dates: if true, pandas will attempt to parse the dates
    :return: pandas dataframe with an string-based index
    """
    df = pd.read_csv(input_filename, header=header, index_col=index_col, sep=sep, parse_dates=parse_dates, encoding=encoding)
    df.index = [str(i) for i in df.index]
    return df


def save_sparse(sparse_matrix, output_filename):
    assert sparse.issparse(sparse_matrix)
    if sparse.isspmatrix_coo(sparse_matrix):
        coo = sparse_matrix
    else:
        coo = sparse_matrix.tocoo()
    row = coo.row
    col = coo.col
    data = coo.data
    shape = coo.shape
    np.savez(output_filename, row=row, col=col, data=data, shape=shape)


def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
    return coo_matrix.tocsc()


def save_dense(array, output_filename):
    np.savez(output_filename, array=array)


def load_dense(input_filename):
    npy = np.load(input_filename)
    return npy['array']
