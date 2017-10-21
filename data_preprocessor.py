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

    def drop_excluded_trials(self, choices, dynamics, trials):
        choices = choices.drop(trials, errors='ignore')
        
        # NB: this is a hack to deal with (supposedly) pandas bug.
        # When multiindex values are not unique (as in dynamics dataframes), 
        # drop doesn't really work, so we have to remove trials one-by-one
        dynamics = dynamics.reset_index()
        for trial in trials:
            dynamics = dynamics[~((dynamics.subj_id==trial[0]) & (dynamics.session_no==trial[1]) & 
                                  (dynamics.block_no==trial[2]) & (dynamics.trial_no==trial[3]))]
        dynamics = dynamics.set_index(self.index, drop=True)
        
        return choices, dynamics        
    
    def get_mouse_and_gaze_measures(self, choices, dynamics, stim_viewing):
        choices['is_correct'] = choices['direction'] == choices['response']
        choices.response_time /= 1000.0
        choices.gamble_time /= 1000.0
        choices['xflips'] = dynamics.groupby(level=self.index).\
                                    apply(lambda traj: self.zero_cross_count(traj.mouse_vx))    
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_maxd))
        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_midline_d))
        choices['is_com'] = ((choices.midline_d > self.com_threshold_x) & \
                                (choices.midline_d_y > self.com_threshold_y))
        
        # Here in gambling experiment, initiation times should be redefined
        # eye initiation time is the first passage time to the border between RDK and response area
        # maybe we can call it gaze latency
        # but mouse initiation time is basically undefined
        # what we have here is total time it takes until decision is executed
#        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_mouse_IT))
#        choices = choices.join(dynamics.groupby(level=self.index).apply(self.get_eye_IT))                       
#        choices['ID_lag'] = choices.mouse_IT - choices.eye_IT
        
        # We can also z-score within participant AND coherence level, the results remain the same
        # ['subj_id', 'coherence']
        z_columns = ['response_time', 'gamble_value', 'gamble_time', 'max_d', 'xflips']
        for col in z_columns:
            choices[col + '_zscore'] = choices[col].groupby(level='subj_id'). \
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
        
#        if is_final_point_positive:
#            midline_d = abs(mouse_x.min())
##            idx_midline_d = int(mouse_x.argmin())
#        else:
#            midline_d = abs(mouse_x.max())
##            idx_midline_d = int(mouse_x.argmax())
        midline_d = mouse_x.min() if is_final_point_positive else mouse_x.max()
#        print(traj)
#        print(midline_d)
#        print(abs(mouse_x.argmin()) if is_final_point_positive else abs(mouse_x.argmax()))
#        print((mouse_x == midline_d).nonzero())
        idx_midline_d = (mouse_x == midline_d).nonzero()[0][-1]
        midline_d_y = traj.mouse_y.values[idx_midline_d]
        return pd.Series({'midline_d': abs(midline_d), 
                          'idx_midline_d': idx_midline_d,
                          'midline_d_y': midline_d_y})

    def zero_cross_count(self, x):
        return (abs(np.diff(np.sign(x))) > 1).sum()

    def get_mouse_IT(self, traj):
        v = traj.mouse_v.values
    
        onsets = []
        offsets = []
        is_previous_v_zero = True
    
        for i in np.arange(0,len(v)):
            if v[i]!=0:
                if is_previous_v_zero:
                    is_previous_v_zero = False
                    onsets += [i]
                elif (i==len(v)-1):
                    offsets += [i]            
            elif (not is_previous_v_zero):
                offsets += [i]
                is_previous_v_zero = True
    
        submovements = pd.DataFrame([{'on': onsets[i], 
                 'off': offsets[i], 
                 'on_t': traj.timestamp.values[onsets[i]],
                 'distance':(traj.mouse_v[onsets[i]:offsets[i]]*
                             traj.timestamp.diff()[onsets[i]:offsets[i]]).sum()}
                for i in range(len(onsets))])
    
        it = submovements.loc[submovements.distance.ge(self.it_distance_threshold ).idxmax()].on_t
        return pd.Series({'mouse_IT': it, 'motion_time': traj.timestamp.max()-it})

    def get_eye_IT(self, traj):        
        v = traj.eye_v
        if ((v < self.eye_v_threshold) | (v.isnull())).all():
            # if eye stays at one location throughout the whole trial, 
            # it is supposedly fixated either at the center of the screen, or at the response location
            # in the former case, initation time is inf, in the latter case, initation time is 0
            # we detect which case is tru  by looking at first value of eye_x
            if abs(traj.eye_x.iloc[0]) < 100: 
                eye_IT = np.inf
                eye_initial_decision = 0
            else:
                eye_IT = 0
                eye_initial_decision = np.sign(traj.eye_x.iloc[0])
        else:
            eye_IT_idx = (v > self.eye_v_threshold).nonzero()[0][0]
            eye_IT = traj.timestamp.iloc[eye_IT_idx]
            eye_initial_decision = np.sign(traj.eye_x.iloc[eye_IT_idx+1])
        
        return pd.Series({'eye_IT': eye_IT, 'eye_initial_decision': eye_initial_decision})

    def get_stim_mouse_IT(self, stim_traj):
        t = stim_traj.timestamp.values
        v = stim_traj.mouse_v.values
        if v[-1]:
            idx = np.where(v==0)[0][-1]+1 if len(v[v==0]) else 0
            IT = (t[idx] - t.max())
        else:
            IT = 0
        return IT
    
    def get_stim_eye_IT(self, stim_traj):
        t = stim_traj.timestamp.values
        v = stim_traj.eye_v.values
    
        idx = np.where(v<self.eye_v_threshold)[0][-1] if len(v[v<self.eye_v_threshold]) else 0
        return (t[idx] - t.max())
    
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
