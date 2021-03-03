import numpy as np
import pandas as pd
import os

from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import load_img, new_img_like
from nilearn.signal import clean
from nilearn.glm.first_level import make_first_level_design_matrix

from .utils import create_task_confounders

def compute_edge_ts(roi_mat):

    """
    Function to compute the unwarped time
    from a matrix of time series

    """
    n_rois = roi_mat.shape[1]
    n_vols = roi_mat.shape[0]

    edge_mat = np.zeros((n_rois, n_rois, 1, n_vols ))

    for ii in range(n_rois):
        for jj in range(ii+1, n_rois):
            edge_mat[ii, jj, 0, :] = roi_mat[:,ii]*roi_mat[:,jj]
            edge_mat[jj, ii, 0,:] = edge_mat[ii, jj, 0, :]
        
    return edge_mat


class NiftiEdgeAtlas():
    
    
    def __init__(self, 
                 atlas_file,
                 detrend,
                 low_pass,
                 high_pass, 
                 t_r,
                 fir_delays=[0]
                ):
        
        self.atlas_file = atlas_file
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.fir_delays = fir_delays
        
    def fit_transform(self, 
                      run_img, 
                      events, 
                      confounds):
        
        run_img = load_img(run_img)
        n_scans = run_img.shape[3]
        # Load events
        if type(events)==str:
            assert os.path.exists(events)
            assert events.endswith(".csv")
            events_mat = pd.read_csv(events)
        else:
            #TODO: Function to check an input numpy array in the correct form
            events_mat = events
        
        start_time = 0
        end_time = (n_scans - 1)* self.t_r
        frame_times = np.linspace(start_time, end_time, n_scans)
        task_conf = create_task_confounders(frame_times, events_mat, fir_delays=self.fir_delays)
        
        label_masker = NiftiLabelsMasker(labels_img=self.atlas_file, 
                                         detrend=self.detrend,
                                         low_pass = self.low_pass,
                                         high_pass = self.high_pass,
                                         t_r = self.t_r, 
                                         standardize=False)
        atlas_ts_conf = label_masker.fit_transform(run_img, confounds=confounds)
        atlas_ts_conf_task = clean(atlas_ts_conf, confounds=task_conf, t_r=self.t_r, detrend=False, standardize='zscore')
    
        edge_ts = compute_edge_ts(atlas_ts_conf_task)
        edge_img = nib.Nifti1Image(edge_ts, affine=run_img.affine)
        edge_img.set_data_dtype(run_img.get_data_dtype()) # Use the same data type

        return edge_img
    
    