import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor

def create_task_confounders(frame_times, events_df, fir_delays):
    
    trial_types = events_df.trial_type.unique()
    task_conf_reg = []
    for trial_name in trial_types:
        cond = events_df.trial_type==trial_name
        trial_events = events_df.loc[cond, ["onset", "duration"]].to_numpy()
        trial_events = np.column_stack((trial_events, np.ones(trial_events.shape[0]))).T # Add amplitudes
        trial_events_reg, _ = compute_regressor(trial_events, hrf_model="fir", fir_delays=fir_delays,frame_times=frame_times)
        task_conf_reg.append(trial_events_reg)
        
    return np.column_stack(task_conf_reg)
