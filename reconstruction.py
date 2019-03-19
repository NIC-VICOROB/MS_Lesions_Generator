import itertools

import numpy as np

def perform_voting(patches, output_shape, expected_shape, extraction_step) :
    vote_img = np.zeros(expected_shape)
    vote_count = np.zeros(expected_shape)

    coordinates = generate_indexes(
        output_shape, extraction_step, expected_shape)

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
        vote_img[selection] += patches[count]
        vote_count[selection] += np.ones(vote_img[selection].shape)

    return np.divide(vote_img, vote_count)

def generate_indexes(output_shape, extraction_step, expected_shape) :
    ndims = len(output_shape)

    poss_shape = [output_shape[i] + extraction_step[i] * ((expected_shape[i] - output_shape[i]) // extraction_step[i]) for i in range(ndims)]

    idxs = [range(output_shape[i], poss_shape[i] + 1, extraction_step[i]) for i in range(ndims)]
    
    return itertools.product(*idxs)