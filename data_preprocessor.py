from __future__ import division
import pandas as pd
import numpy as np
import derivative_calculator

class DataPreprocessor:
    x_lim = 1920
    y_lim = 1080
    
    com_threshold_x = 50
    com_threshold_y = 100
    
    # to determine exact response intiation,
    # threshold for distance travelled by mouse cursor (in pixels) during single movement
    # it is needed when calculating initation time
    it_distance_threshold = 100
    # there are still some false alarms, especially in Exp 1, but they are still few (around 20)
    eye_v_threshold = 200
       
    index = ['subj_id', 'block_no', 'trial_no']
    
    def preprocess_data(self, choices, dynamics, resample=False):
        # originally, EyeLink data has -32768.0 values in place when data loss occurred
        # we replace it with np.nan to be able to use numpy functions properly
        dynamics = dynamics.replace(dynamics.eye_x.min(), np.nan)
        
        dynamics = self.set_origin_to_start(dynamics)                   
        dynamics = self.shift_timeframe(dynamics)
        
        dynamics = self.flip_left(choices, dynamics)
        
        if resample:
            dynamics = self.resample_trajectories(dynamics, n=resample)
        
        dc = derivative_calculator.DerivativeCalculator()
#        dynamics = dc.append_diff(dynamics)
        dynamics = dc.append_derivatives(dynamics)
        
#        dynamics['mouse_v'] = np.sqrt(dynamics.mouse_vx**2 + dynamics.mouse_vy**2 )
#        dynamics['eye_v'] = np.sqrt(dynamics.eye_vx**2 + dynamics.eye_vy**2 )        
                          
        return dynamics  
    
    def get_mouse_and_gaze_measures(self, choices, dynamics, stim_viewing):
        choices['is_correct'] = choices['direction'] == choices['response']
        choices.response_time /= 1000.0
        choices.gamble_time /= 1000.0
        choices['xflips'] = dynamics.groupby(level=self.index).\
                                    apply(lambda traj: self.zero_cross_count(traj.mouse_vx.values)) 
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_maxd))
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_midline_d))
        choices['is_com'] = ((choices.midline_d > self.com_threshold_x) & \
                                (choices.midline_d_y > self.com_threshold_y))
        
        # Here in gambling experiment, initiation times should be redefined
        # eye initiation time is the first passage time to the border between RDK and response area
        # maybe we can call it gaze latency
        # but mouse initiation time is basically undefined
        # what we have here is total time it takes until decision is executed

        # We can also z-score within participant AND coherence level, the results remain the same
        z_groups = ['subj_id']
        z_columns = ['response_time', 'gamble_value', 'gamble_time', 'max_d', 'xflips']
        for col in z_columns:
            choices[col + '_zscore'] = choices[col].groupby(level=z_groups). \
                                        apply(lambda c: (c-c.mean())/c.std())        
        return choices
    
    def set_origin_to_start(self, dynamics):
        # set origin to start button location
        dynamics.mouse_x -= self.x_lim/2
        dynamics.mouse_y = self.y_lim - dynamics.mouse_y
        dynamics.eye_x -= self.x_lim/2
        dynamics.eye_y = self.y_lim - dynamics.eye_y
        return dynamics
    
    def shift_timeframe(self, dynamics):
        # shift time to the timeframe beginning at 0 for each trajectory
        # also, express time in seconds rather than milliseconds
        dynamics.loc[:,'timestamp'] = dynamics.timestamp.groupby(by=self.index). \
                                        transform(lambda t: (t-t.min()))/1000.0
        return dynamics

    def flip_left(self, choices, dynamics):
        for col in ['mouse_x', 'eye_x']:
            dynamics.loc[choices.direction==180, ['mouse_x', 'eye_x']] *= -1
        return dynamics

    def resample_trajectories(self, dynamics, n_steps=100):
        resampled_dynamics = dynamics.groupby(level=self.index).\
                                    apply(lambda traj: self.resample_trajectory(traj, n_steps=n_steps))
        resampled_dynamics.index = resampled_dynamics.index.droplevel(4)
        return resampled_dynamics
            
    def get_maxd(self, traj):
        alpha = np.arctan((traj.mouse_y.iloc[-1]-traj.mouse_y.iloc[0])/ \
                            (traj.mouse_x.iloc[-1]-traj.mouse_x.iloc[0]))
        d = (traj.mouse_x.values-traj.mouse_x.values[0])*np.sin(-alpha) + \
            (traj.mouse_y.values-traj.mouse_y.values[0])*np.cos(-alpha)
        if abs(d.min())>abs(d.max()):
            return pd.Series({'max_d': d.min(), 'idx_max_d': d.argmin()})
        else:
            return pd.Series({'max_d': d.max(), 'idx_max_d': d.argmax()})
        
    def get_midline_d(self, traj):
        mouse_x = traj.mouse_x.values
        is_final_point_positive = (mouse_x[-1]>0)
        
        midline_d = mouse_x.min() if is_final_point_positive else mouse_x.max()

        idx_midline_d = (mouse_x == midline_d).nonzero()[0][-1]
        midline_d_y = traj.mouse_y.values[idx_midline_d]
        return pd.Series({'midline_d': abs(midline_d), 
                          'idx_midline_d': idx_midline_d,
                          'midline_d_y': midline_d_y})

    def zero_cross_count(self, x):
        return (abs(np.diff(np.sign(x)[np.nonzero(np.sign(x))]))>1).sum()
#        return (abs(np.diff(np.sign(x))) > 1).sum()
    
    def resample_trajectory(self, traj, n_steps):
        # Make the sampling time intervals regular
        n = np.arange(0, n_steps+1)
        t_regular = np.linspace(traj.timestamp.min(), traj.timestamp.max(), n_steps+1)
        mouse_x_interp = np.interp(t_regular, traj.timestamp.values, traj.mouse_x.values)
        mouse_y_interp = np.interp(t_regular, traj.timestamp.values, traj.mouse_y.values)
        eye_x_interp = np.interp(t_regular, traj.timestamp.values, traj.eye_x.values)
        eye_y_interp = np.interp(t_regular, traj.timestamp.values, traj.eye_y.values)
        pupil_size_interp = np.interp(t_regular, traj.timestamp.values, 
                                      traj.pupil_size.values)
        traj_interp = pd.DataFrame([n, t_regular, mouse_x_interp, mouse_y_interp, \
                                    eye_x_interp, eye_y_interp, pupil_size_interp]).transpose()
        traj_interp.columns = ['n', 'timestamp', 'mouse_x', 'mouse_y', 'eye_x', 'eye_y', 'pupil_size']
#        traj_interp.index = range(1,n_steps+1)
        return traj_interp
