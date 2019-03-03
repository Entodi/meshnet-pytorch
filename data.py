import progressbar
import nipy
import numpy as np
from scipy.stats import truncnorm
from torch.utils.data.dataset import Dataset
import torch
from joblib import Parallel, delayed


def unit_interval_normalization(x):
    return (x - x.min()) / (x.max() - x.min())


class Subject:
    def __init__(self, path, input_filenames, 
        target_filename, subvolume_shape, 
        preprocessing, extended):
        self._path = path
        self._input_filenames = input_filenames
        self._target_filename = target_filename
        self._preprocessing = preprocessing
        self._subvolume_shape = subvolume_shape
        self._half_subvolume_shape = subvolume_shape // 2
        self._extended = extended


    def _pad_volume(self, _input):
        new_shape = np.ceil(
            _input.shape / self._subvolume_shape) * self._subvolume_shape
        new_shape = tuple(new_shape.astype('int'))
        addition = new_shape - np.array(_input.shape)
        half_addition = addition // 2
        temp = torch.zeros(new_shape)
        temp[half_addition[0]:-half_addition[0],
            half_addition[1]:-half_addition[1],
            half_addition[2]:-half_addition[2]] = _input
        self._original = half_addition
        return temp


    def load_volume(self):
        self._inputs = {}
        self._target = None
        self._volume_shape = None
        # Load inputs
        for j, input_filename in enumerate(self._input_filenames):
            _input = torch.from_numpy(np.array(
                nipy.load_image(self._path + input_filename).get_data(), dtype='float'))
            if self._extended:
                _input = self._pad_volume(_input)
            if self._volume_shape is None:
                self._volume_shape = _input.shape
            else:
                assert self._volume_shape == _input.shape, \
                    'Input {} shape are not consisted with first input shape'.format(
                        self._path + input_filename)
            for f in self._preprocessing:
                self._inputs[j] = f(_input)
        # Load target if defined
        if self._target_filename:
            _target = torch.from_numpy(np.array(nipy.load_image(
                self._path + self._target_filename).get_data(), dtype='int'))
            if self._extended:
                _target = self._pad_volume(_target)
            # check for consistency with input shape
            assert _target.shape == self._volume_shape, \
                'Target shapes are not consistent with Input shapes for {}'.format(
                    self._path)
            self._target = _target
    
    def generate_nonoverlap_coordinates(self):

        def generate_centered_nonoverlap_1d_grid(length, step):
            # Calculation of the centered nonoverlap grid
            # Grid will not cover the whole volume if the multiplier 
            # of the volume shape is not equal to subvolume shape
            return [(c, c + step) for c in range((length % step) // 2, length - step, step)]
                
        z = generate_centered_nonoverlap_1d_grid(
            self._volume_shape[0], self._subvolume_shape[0])
        y = generate_centered_nonoverlap_1d_grid(
            self._volume_shape[1], self._subvolume_shape[1])
        x = generate_centered_nonoverlap_1d_grid(
            self._volume_shape[2], self._subvolume_shape[2])
        self._nonoverlap_coordinates = np.array([[i, j, l] for i in z for j in y for l in x])


    def set_truncated_gaussian_coordinate_generator(self, mus=None, sigmas=None):
        if mus is None:
            mus = np.array(
                [self._volume_shape[0] // 2, 
                self._volume_shape[0] // 2, 
                self._volume_shape[0] // 2]
            )
        if sigmas is None:
            sigmas = np.array(
                [self._volume_shape[0] // 3, 
                self._volume_shape[0] // 3, 
                self._volume_shape[0] // 3]
            )
        self._truncnorm_coordinates = truncnorm(
            (self._half_subvolume_shape - mus + 1) / sigmas, 
            (self._volume_shape - self._half_subvolume_shape - mus) / sigmas, 
            loc=mus, scale=sigmas
        )


    def generate_truncated_gaussian_coordinate(self):
        xyz = np.round(self._truncnorm_coordinates.rvs(size=(1, 3))[0]).astype('int')
        xyz_start = xyz - self._half_subvolume_shape
        xyz_end = xyz + self._half_subvolume_shape
        xyz_coords = np.vstack((xyz_start, xyz_end)).T
        return xyz_coords


    def get_input(self):
        return self._inputs

    
    def get_target(self):
        return self._target


    def get_nonoverlap_coordinates(self):
        return self._nonoverlap_coordinates

    
    def get_nonoverlap_coordinate(self, index):
        return self._nonoverlap_coordinates[index]

    
    def get_volume_shape(self):
        return self._volume_shape

    
    def get_target(self):
        return self._target


    def get_original(self):
        return self._original


class VolumetricDataset(Dataset):
    def __init__(self, 
        filename, n_subvolumes, subvolume_shape, mus=None, sigmas=None, 
        input_filenames=['T1.nii.gz'], target_filename='atlas_full_104.nii.gz', 
        preprocessing=[unit_interval_normalization], 
        evaluation=False, extended=False
    ):
        self._filename = filename
        self._n_subvolumes = n_subvolumes
        self._subvolume_shape = subvolume_shape
        self._mus = mus
        self._sigmas = sigmas
        self._isTest = False if target_filename else True
        self._input_filenames = input_filenames
        self._nInputs = len(input_filenames)
        self._target_filename = target_filename
        self._preprocessing = preprocessing
        self._evaluation = evaluation
        self._extended = extended

    
    def build(self):
        self._load_volumes()

        if self._isTest or self._evaluation:
            self._nonoverlap_coordinates()
        
        self._truncnorm_coordinates()


    def get_paths(self):
        return self._paths


    def get_element(self, index):
        return self._dataset[index]


    def get_all_data(self):
        return self._dataset


    def __getitem__(self, index):
        brain_id = index // self._n_subvolumes
        coords_id = index // len(self._dataset.keys())
        
        subject = self._dataset[brain_id]

        coords = np.array([])
        if (self._isTest or self._evaluation) and coords_id < len(
                subject.get_nonoverlap_coordinates()):
            coords = subject.get_nonoverlap_coordinate(coords_id)
        else:
            coords = subject.generate_truncated_gaussian_coordinate()

        data_tensor = torch.zeros((self._nInputs, self._subvolume_shape[0], 
            self._subvolume_shape[1], self._subvolume_shape[2]), dtype=torch.float32)
        subject_input = subject.get_input()
        for k in subject_input.keys():
            data_tensor[k, 
                :self._subvolume_shape[0], 
                :self._subvolume_shape[1], 
                :self._subvolume_shape[2]] = subject_input[k][
                    coords[0][0]:coords[0][1], 
                    coords[1][0]:coords[1][1], 
                    coords[2][0]:coords[2][1]]

        subject_target = subject.get_target()
        if not self._isTest:
            target_tensor = torch.zeros(
                (self._subvolume_shape[0], 
                self._subvolume_shape[1], 
                self._subvolume_shape[2]), dtype=torch.long)
            target_tensor[
                :self._subvolume_shape[0], 
                :self._subvolume_shape[1], 
                :self._subvolume_shape[2]] = subject_target[
                    coords[0][0]:coords[0][1], 
                    coords[1][0]:coords[1][1],  
                    coords[2][0]:coords[2][1]]
                    
        return data_tensor, target_tensor, coords


    def __len__(self):
        return self._n_subvolumes * len(self._dataset.keys())


    def _read_paths(self):
        if isinstance(self._filename, str):
            try:
                f = open(self._filename, 'r')
            except Exception as e:
                assert False, 'Error: {}'.format(e)
            self._paths = f.read().splitlines()
        elif isinstance(self._filename, list):
            self._paths = self._filename
        else:
            assert False, 'Filename isn\'t path or list'


    def _nonoverlap_coordinates(self):
        for k in self._dataset.keys():
            self._dataset[k].generate_nonoverlap_coordinates()
            

    def _truncnorm_coordinates(self):
        for k in self._dataset.keys():
            self._dataset[k].set_truncated_gaussian_coordinate_generator(
               mus=self._mus, sigmas=self._sigmas)


    def _load_volumes(self):

        def job(index):
            self._dataset[index] = Subject(self._paths[index], self._input_filenames, 
                self._target_filename, self._subvolume_shape, 
                self._preprocessing, self._extended)
            self._dataset[index].load_volume()     

        self._read_paths()
        self._dataset = {}
        Parallel(n_jobs=-1)(
            delayed(job)(i) for i in progressbar.progressbar(range(len(self._paths))))
