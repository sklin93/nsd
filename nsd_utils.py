import os
import h5py
import json
import yaml
import pickle
import numpy as np
import nibabel as nib
import scipy.io as spio
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

with open('nsd_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)
    print(config)

DATA_DIR = config['data']['data_dir']
voxel_size = config['data']['voxel_size']
ROI_FILE = config['data']['roi_file']
ROI_VOX = config['data']['roi_vox']

STIM_FILE = os.path.join(DATA_DIR, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
STIM_ORDER_FILE = os.path.join(DATA_DIR, 'nsddata/experiments/nsd/nsd_expdesign.mat')
STIM_INFO = os.path.join(DATA_DIR, 'nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
STIM_CAP = os.path.join(DATA_DIR, 'nsddata_stimuli/stimuli/nsd/annotations/nsd_captions.json')
STIM_CAT = os.path.join(DATA_DIR, 'nsddata_stimuli/stimuli/nsd/annotations/nsd_cat.json')

FMRI_DIR = os.path.join(DATA_DIR, 'nsddata_betas/ppdata/subj01/func'+voxel_size+'/betas_fithrf_GLMdenoise_RR/')
ROI_FILE = os.path.join(DATA_DIR, 'nsddata/ppdata/subj01/func'+voxel_size+'/roi/', ROI_FILE)
ROI_VOX = os.path.join(DATA_DIR, 'nsddata_betas/ppdata/subj01/func'+voxel_size+'/', ROI_VOX)

TRIAL_PER_SESS = 750
SESS_NUM = 37
MAX_IDX = TRIAL_PER_SESS * SESS_NUM
device = torch.device(config['device'])

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def extract_voxels(fmri_dir, roi_file, out_dir, regions=None, flatten=False):
    ''' Extract voxels defined by roi_file.
    Served as a preprocessing to save time during sample loading.
    Write extracted vectors into hdf5 files under output_dir.

    - fmri_dir: where the fmri betas are located.
                Each beta file should have shape (750, 83, 104, 81).
    - roi_file: the file defining ROI. Should have shape (81, 104, 83).
                Voxels values out of interest are -1.
                The else ranges from 0 to the (#region + 1).
    - regions: a list. If provided, the funciton will only care about the voxels
              in the given regions. If None, will care about all voxels not -1.
    - flatten: if True, return a flattened vector for each fmri sample.
               (only voxels in the input roi_file are kept)
    - out_dir: where to write the new fmri to.
    '''
    assert os.path.isdir(out_dir), 'mkdir for the output directory first!'
    mask = nib.load(roi_file).get_fdata()

    if regions:
        assert max(regions) <= mask.max(), 'region index out of range!'
        mask = np.stack([mask == r for r in regions]).sum(0)
        mask = (mask != 0)
    else:
        mask = (mask != -1)
    print('ROI voxel count:', np.count_nonzero(mask))

    # mask axis orders are different from fmri axis order, do transpose
    mask = mask.T

    # processing all fmri data
    fmri_files = [f for f in os.listdir(fmri_dir) if os.path.isfile(os.path.join(fmri_dir, f))]
    for fmri_file in tqdm(fmri_files):
        with h5py.File(os.path.join(fmri_dir, fmri_file), 'r') as f:
            fmri = f['betas'][()]
        if flatten:
            fmri = [fmri[trial][mask] for trial in range(len(fmri))]
            fmri = np.stack(fmri)
        else:
            fmri = fmri * mask[None,...]
        # save new fmri
        out_f = os.path.join(out_dir,
                             os.path.basename(roi_file)[:-7] + '_' + fmri_file[:-5] + '.hdf5')
        with h5py.File(out_f, 'w') as f:
            dset = f.create_dataset('roi_betas', data=fmri)


class NSDDataset(Dataset):
    def __init__(self, load_img=False, img_trans=None, load_fmri=False,
                 roi=None, load_caption=False, load_cat=False):
        '''
        Support loading one or more of: image, fmri betas, text captions/categories.

        - load_img, load_fmri, load_caption, load_cat: all bool,
          choose the modalities you need.
        - img_trnas: torchvision Transformations, optional
        - roi: string, the directory contains extracted ROI voxels.
               if None, return the 3d fmri activity: (83, 104, 81) for 1.8mm

        Note: there are multiple captions per image,
              the returned caption only keeps the first one.
        
        '''
        assert load_img or load_fmri or load_caption or load_caption, (
            'You must choose to load at least one modeality!'
            )
        
        if load_img or load_caption or load_cat:
            stim_order = loadmat(STIM_ORDER_FILE)
            self.subjectim = stim_order['subjectim']
            self.masterordering = stim_order['masterordering']
            if load_caption or load_cat:
                with open(STIM_INFO, 'rb') as f:
                    self.stim_info = pickle.load(f, encoding='latin1')
                if load_caption:
                    with open(STIM_CAP, 'r') as f:
                        self.nsd_captions = json.load(f)
                if load_cat:
                    with open(STIM_CAT, 'r') as f:
                        self.nsd_cat = json.load(f)                
       
        if load_img:
            self.stim_file = STIM_FILE
            self.img_trans = img_trans
       
        if load_fmri:
            self.fmri_dir = roi if roi else FMRI_DIR
            self.fmri_files = [f for f in os.listdir(self.fmri_dir) if
                               os.path.isfile(os.path.join(self.fmri_dir, f))]
            self.fmri_key = 'roi_betas' if roi else 'betas'
        
        self.load_img = load_img
        self.load_fmri = load_fmri
        self.load_caption = load_caption
        self.load_cat = load_cat

    def __len__(self):
        return MAX_IDX

    def get_fmri_shape(self):
        with h5py.File(os.path.join(self.fmri_dir,
                                    self.fmri_files[0]), 'r') as f:
            return f[self.fmri_key][0].shape

    def __getitem__(self, idx, verbose=False):
        sample = {}
        multi = False if type(idx).__name__ == 'int' else True
        if self.load_img or self.load_caption or self.load_cat:
            nsdId = self.subjectim[0, self.masterordering[idx] - 1] - 1
            if verbose:
                print(f'stim nsd id: {nsdId}, aka #{nsdId+1} in the tsv files.')

        if self.load_img:
            # dealing with multiple indexing,
            # to ensure the index maintains increasing order.
            if multi:
                nsdId_order = np.argsort(nsdId)
                _map = {nsdId_order[i] : i for i in range(len(nsdId_order))}
                nsdId_order = [_map[i] for i in range(len(nsdId_order))]
            with h5py.File(self.stim_file, 'r') as f:
                _nsdId = np.sort(nsdId) if multi else nsdId
                img_sample = f['imgBrick'][_nsdId]
            img_sample = img_sample[nsdId_order] if multi else img_sample
            if verbose:
                print('stim shape:', img_sample.shape)                        
            if self.img_trans:
                img_sample = self.img_trans(img_sample)
            sample['img'] = img_sample#.to(device)

        if self.load_fmri:
            idx_array = np.arange(default(idx.start, 0),
                                  default(idx.stop, self.__len__()), idx.step
                                  ) if multi else (idx)
            sess = idx_array // TRIAL_PER_SESS + 1
            if multi:
                fmri_sample = []
                for s in range(len(sess)):
                    fmri_file = os.path.join(
                        self.fmri_dir,
                        f'{self.fmri_files[0][:-7]}{sess[s]:02}.hdf5')
                    with h5py.File(fmri_file, 'r') as f:
                        fmri_sample.append(f[self.fmri_key]
                                           [idx_array[s] % TRIAL_PER_SESS])
                fmri_sample = np.stack(fmri_sample)
            else:
                fmri_file = os.path.join(
                    self.fmri_dir, f'{self.fmri_files[0][:-7]}{sess:02}.hdf5')
                with h5py.File(fmri_file, 'r') as f:
                    fmri_sample = f[self.fmri_key][idx_array % TRIAL_PER_SESS]                
                if verbose:
                    print('fmri loaded from', fmri_file)
                    print('fmri shape:', fmri_sample.shape)
                    print('beta min, mean, max:', fmri_sample.min(),
                        fmri_sample.mean(), fmri_sample.max())
            # # standardize
            # fmri_sample = zscore(fmri_sample)
            # fmri_sample = torch.FloatTensor(fmri_sample).to(device)
            sample['fmri'] = fmri_sample
        
        if self.load_caption:
            if multi:
                caption = []
                for id in nsdId:
                    caption.append(
                        self.nsd_captions[str(self.stim_info['cocoId'][id])][0])
            else:
                caption = self.nsd_captions[str(self.stim_info['cocoId']
                                                [nsdId])]
                if verbose:
                    print('Captions:')
                    for a in caption:
                        print(a)
                caption = [caption[0]]      
            sample['caption'] = caption

        if self.load_cat:
            if multi:
                cat = []
                for id in nsdId:
                    cat.append(self.nsd_cat[str(self.stim_info['cocoId'][id])])
            else:
                cat = [self.nsd_cat[str(self.stim_info['cocoId'][nsdId])]]
                if verbose:
                    print('Categories:', cat)
            sample['cat'] = cat
        return sample
