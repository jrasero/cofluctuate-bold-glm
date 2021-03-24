import numpy as np
import pandas as pd
import os

from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import load_img, new_img_like

from .utils import create_task_confounders, denoise_task, standardize

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
                 detrend = False,
                 low_pass = None,
                 high_pass= None, 
                 t_r = None,
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
                      events=None, 
                      confounds=None):
        
        run_img = load_img(run_img)
        n_scans = run_img.shape[3]
        
        #TODO: See if it makes sense to create a function for this
        # or a base class that has this method
        
        # 1- Load and compute FIR events
        task_conf = None
        if events is not None:
            if isinstance(events, str):
                assert os.path.exists(events)
                assert events.endswith("events.tsv")
                events_mat = pd.read_csv(events, sep="\t")
            
                start_time = 0
                end_time = (n_scans - 1)* self.t_r
                frame_times = np.linspace(start_time, end_time, n_scans)
                task_conf = create_task_confounders(frame_times, events_mat, 
                                                    fir_delays=self.fir_delays)
                
            elif isinstance(events, np.ndarray):
                # You can supply a given task matrix to denoise
                task_conf = events
            
            
        self.task_conf_ = task_conf
        
        # 2- Parcellate data
        label_masker = NiftiLabelsMasker(labels_img=self.atlas_file, 
                                         detrend=self.detrend,
                                         low_pass = self.low_pass,
                                         high_pass = self.high_pass,
                                         t_r = self.t_r, 
                                         standardize=False)
        atlas_ts_conf = label_masker.fit_transform(run_img, confounds=confounds)
        self.atlas_ts_conf_ = atlas_ts_conf.copy()
        
        # 3- Remove events if passed
        if events is not None:
            atlas_ts_conf_task = denoise_task(X=task_conf, Y = atlas_ts_conf)
        else:
            atlas_ts_conf_task = atlas_ts_conf.copy()
        self.atlas_ts_conf_task_ = atlas_ts_conf_task
        
        # 4-Standardize data 
        atlas_ts_clean =  standardize(atlas_ts_conf_task)
    
        edge_ts = compute_edge_ts(atlas_ts_clean)
        edge_img = new_img_like(run_img, edge_ts, affine = np.eye(4)) # Add fake affine (old was:run_img.affine)

        return edge_img
    
    
