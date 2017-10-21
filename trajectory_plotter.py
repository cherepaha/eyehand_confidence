import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

class TrajectoryPlotter:
    x_lim = [-960-10, 960+10]
    y_lim = [0-10, 1080+10]
    left_resp_area_center = [-(960-150), (1080-170)]
    right_resp_area_center = [(960-150), (1080-170)]
    resp_area_radius = 90
    
    n_cells = 30
    legendFontSize = 16
    tickLabelFontSize = 20
    axisLabelFontSize = 24
    lw=2.0
    
    def __init__(self):
        pass
#        self.ax = self.init_xy_plot()
     
    def set_axis_params(self):
        self.ax.set_xlabel(r'x coordinate', fontsize=self.axisLabelFontSize)
        self.ax.set_ylabel(r'y coordinate', fontsize=self.axisLabelFontSize)
        self.ax.tick_params(axis='both', which='major', labelsize=self.tickLabelFontSize)
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        
    def plot_mouse_trajectories(self, trajectories, title='Mouse trajectories'):
        for idx, trajectory in trajectories.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']):
            self.ax.plot(trajectory.mouse_x.values, trajectory.mouse_y.values, 
                         alpha = 0.2, color = sns.color_palette()[0])
#        self.ax.legend()
        self.ax.set_title(title, fontsize=self.axisLabelFontSize)
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
#        plt.tight_layout()

    def plot_subject_eye_trajectories(self, data, subj_id, block_no=10):
        for trial_no, trajectory in data.loc[subj_id, block_no].groupby(level='trial_no'):
            self.ax.plot(trajectory.eye_x.values, trajectory.eye_y.values, marker='o', label=trial_no)
        self.ax.set_title('Participant '+ str(subj_id), fontsize=self.axisLabelFontSize)
#        self.ax.legend()
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        plt.tight_layout()
    
    def plot_pupil_size(self, trajectory):
        fig, ax = plt.subplots(1)
        ax.plot(trajectory.timestamp, trajectory.pupil_size_rel, marker='o')
        ax.set_xlabel('Time, s')
        ax.set_ylabel('Pupil size')
        plt.tight_layout()    
        
    def plot_pupil_size_hist(self, data):
        fig, ax = plt.subplots(1)
        for subj_id, subj_data in data.groupby(level=['subj_id']):
            sns.distplot(subj_data.pupil_size_rel, ax=ax)
        plt.tight_layout()
            
    def plot_all_subjects_trajectories(self, data, subj_ids):
        self.set_axis_params()        
        for subj_id in subj_ids:
            for trial_no, trajectory in data.loc[subj_id].groupby(level=['block_no', 'trial_no']):
                self.ax.plot(trajectory.mouse_x.values, trajectory.mouse_y.values, marker='o')
            self.ax.set_title('Participant '+ str(subj_id), fontsize=self.axisLabelFontSize)
    #        self.add_grid()
            plt.tight_layout()
            plt.savefig(str(subj_id)+'.png')
            plt.cla()
#        plt.ion()
    
    def init_xy_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)        
        plt.axis('scaled')
        self.set_axis_params()
        left_resp_area = plt.Circle(self.left_resp_area_center, self.resp_area_radius, 
                                    alpha = 0.3, color = 'red')
        right_resp_area = plt.Circle(self.right_resp_area_center, self.resp_area_radius, 
                                     alpha = 0.3, color = 'green')
        plt.gca().add_artist(left_resp_area)
        plt.gca().add_artist(right_resp_area)
        
        return self.ax
        
    def plot_trajectory_xy(self, trajectory, styles=['-', '-'], markers=['o', 'v'], lw=2, 
                           color='grey', ax=None):
        if not ax is None:        
            self.ax = ax
        else:
            self.ax = self.init_xy_plot()    
            
        self.ax.plot(trajectory.mouse_x, trajectory.mouse_y, ls=styles[0], marker=markers[0],
                     markersize = 7, label='Mouse', color=color, lw=lw)
#        self.ax.plot(trajectory.eye_x, trajectory.eye_y, ls=styles[1], marker=markers[1],
#                     markersize = 7, label='Eye')
#        plt.legend(loc='lower left', fontsize = self.legendFontSize)
        plt.tight_layout()        
        return self.ax
    
    def plot_trajectory_var(self, trajectory, var, lw=2, ax=None):
        if not ax is None:        
            self.ax = ax
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title(str(trajectory.index.unique().levels))
            
        self.ax.set_xlabel(r'time $t$', fontsize=self.axisLabelFontSize)        
        self.ax.set_ylabel(r'%s' % (var), fontsize=self.axisLabelFontSize)
        
        self.ax.plot(trajectory['timestamp'], trajectory[var], lw=lw,
                     ls='-', marker='v', markersize=5, label=var)
        
        return self.ax
    
    def plot_trajectory_x(self, trajectory, styles=['-', '-'], markers=['v', 'o'], lw=2, ax=None,
                          v=False, eye=True):
        if not ax is None:        
            self.ax = ax
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title(str(trajectory.index.unique()))
        
        self.ax.set_xlabel(r'time $t$', fontsize=self.axisLabelFontSize)        
        self.ax.set_ylabel(r'$x$ coordinate', fontsize=self.axisLabelFontSize)
        
        if v==True:
#            t = trajectory.timestamp
#            mouse_vx = trajectory.mouse_vx
            t = trajectory.timestamp[:-1]
            timestep = trajectory.timestamp.values[1] - trajectory.timestamp.values[0]
            mouse_vx = np.diff(trajectory.mouse_x)/timestep
            eye_vx = np.diff(trajectory.eye_x)/timestep
#
            self.ax.plot(t, mouse_vx, ls=styles[0], marker=markers[0],
                         markersize = 5, label='Mouse $v_{x}$')
            if eye:
                self.ax.plot(t, eye_vx, ls=styles[1], marker=markers[1],
                         markersize = 5, label='Eye $v_{x}$')
        else:
            self.ax.set_ylim(self.x_lim)
            self.ax.plot(trajectory.timestamp, trajectory.mouse_x, ls=styles[0], marker=markers[0],
                         markersize = 5, label='Mouse')
            if eye:
                self.ax.plot(trajectory.timestamp, trajectory.eye_x, ls=styles[1], marker=markers[1],
                         markersize = 5, label='Eye')
        self.ax.legend(fontsize=self.legendFontSize)
        plt.tight_layout()        
        return self.ax    
    
    def plot_trajectory(self, trajectory):
        plt.ion()
        fig, axes = plt.subplots(2, 3)
        axes[0][0].plot(trajectory.timestamp, trajectory.mouse_x, label='Mouse $x$')
        axes[0][1].plot(trajectory.timestamp, trajectory.mouse_y)
        axes[0][2].plot(trajectory.mouse_x, trajectory.mouse_y, marker='o') 

        axes[0][0].set_ylim(self.x_lim)
        axes[0][1].set_ylim(self.y_lim)
        axes[0][2].set_xlim(self.x_lim)
        axes[0][2].set_ylim(self.y_lim)

        axes[0][0].set_ylabel('Mouse x', fontsize=self.axisLabelFontSize)
        axes[0][1].set_ylabel('Mouse y', fontsize=self.axisLabelFontSize)
        axes[0][2].set_xlabel('Mouse x', fontsize=self.axisLabelFontSize)
        axes[0][2].set_ylabel('Mouse y', fontsize=self.axisLabelFontSize)
        
        axes[1][0].plot(trajectory.timestamp, trajectory.eye_x)
        axes[1][1].plot(trajectory.timestamp, trajectory.eye_y)        
        axes[1][2].plot(trajectory.eye_x, trajectory.eye_y, marker='o')
        
        axes[1][0].set_ylim(self.x_lim)
        axes[1][1].set_ylim(self.y_lim)
        axes[1][2].set_xlim(self.x_lim)
        axes[1][2].set_ylim(self.y_lim)
                
        axes[1][0].set_ylabel('Eye x', fontsize=self.axisLabelFontSize)
        axes[1][1].set_ylabel('Eye y', fontsize=self.axisLabelFontSize)
        axes[1][2].set_xlabel('Eye x', fontsize=self.axisLabelFontSize)
        axes[1][2].set_ylabel('Eye y', fontsize=self.axisLabelFontSize)
        
        axes[1][0].set_xlabel(r'time t', fontsize=self.axisLabelFontSize)
        axes[1][1].set_xlabel(r'time t', fontsize=self.axisLabelFontSize)
#        axes[0][0].set_ylabel(r'x coordinate', fontsize=self.axisLabelFontSize)
#        axes[1][0].set_ylabel(r'y coordinate', fontsize=self.axisLabelFontSize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()
        return axes
    
    def plot_com_eye_x(self, com_eye_x):
        fig, ax = plt.subplots(1)
        for idx, trajectory in com_eye_x.groupby(level=['subj_id', 'session_no', 'block_no', 'trial_no']):
            ax.plot(trajectory.eye_x.values, alpha = 0.2, color = sns.color_palette()[0])
#        self.ax.legend()
#        self.ax.set_xlim(self.x_lim)
#        self.ax.set_ylim(self.y_lim)
        plt.tight_layout()        

    def plot_dynamics_window(self, data, variables=['mouse_x', 'eye_x'], err_style='ci_band', 
                             estimator=np.nanmedian, ax=None, invert_errors=False):
        if invert_errors:
            data.loc[~data.is_correct, variables] *= -1
            
        data = pd.melt(data, id_vars=['id', 't'], value_vars=variables)
    
        ax = sns.tsplot(unit='id', time='t', data=data, value='value', condition='variable',
                       estimator=estimator, err_style=err_style, legend=False, ax=ax)
        return ax

    def plot_average_dynamics(self, data, variables=['mouse_x', 'eye_x'], condition=None, ax=None,
                              err_style='ci_band', estimator=np.nanmedian, invert_errors=False):
        if not ax is None:
            fig, ax = plt.subplots(1)
        palette = itertools.cycle(sns.color_palette())
        
        for variable in variables:
            if invert_errors:
                data.loc[~data.is_correct, variable] *= -1
            
            if condition is None:
                plot_data = data[[variable] + ['id', 'n']].reset_index(drop=True)
                plot_data = np.array(plot_data.pivot(index='id', columns='n', values=variable).values)
                ax=sns.tsplot(ax=ax, data = plot_data, estimator=estimator, color = next(palette),
                                   err_style=err_style
#                                   , label='%s' % (variable)
                                   )
                ax.set_xlabel('time t (rescaled)')
                ax.set_ylabel('x coordinate')
            else:
                for cond_value in sorted(data[condition].unique()):
                    plot_data = data[[variable] + ['id', 'n']][data[condition] == cond_value].reset_index(drop=True)
                    plot_data = np.array(plot_data.pivot(index='id', columns='n', values=variable).values)
                    ax=sns.tsplot(ax=ax, data = plot_data, estimator=estimator, 
                               color = next(palette), err_style=err_style
#                                ,label='%s, %s : %s' % (variable, condition, str(cond_value))
                               )

        plt.legend(loc='upper left')
        return ax
        
    def add_grid(self):
        x_ticks = np.linspace(self.x_lim[0], self.x_lim[1], self.n_cells)
        y_ticks = np.linspace(self.y_lim[0], self.y_lim[1], self.n_cells)
        
        self.ax.set_xticks(x_ticks, minor=True)
        self.ax.set_yticks(y_ticks, minor=True)

        self.ax.grid(b=False, which='major')
        self.ax.grid(b=True, which='minor', color='w', linewidth=0.5)
    
    def plot_surface(self, x, y, z):
        self.ax = self.fig.gca(projection='3d')
        x, y = np.meshgrid(x, y)
        
    def plot_p_right(self, p_right, choices, individual = False):
        if individual:    
            g = sns.factorplot(data = p_right, x = 'coherence', y = 'p_right', hue = 'direction', 
                               col = 'subj_id', col_wrap = 5, legend_out = False, ci=95, 
                               order=np.sort(choices.coherence.unique()), kind='point')
            for ax in g.axes:
                ax.set_ylim((0.0, 1.1))
                ax.axhline(0.5, ls='--', color = 'grey')
    #            g.savefig('../figures/p_right_individual_top9.png')
        else:
            g = sns.factorplot(data = p_right, x = 'coherence', y = 'p_right', hue = 'subj_id', 
                               legend_out = False, ci=None, kind='point')
            g.axes[0][0].set_ylim((0.0, 1.1))
            g.axes[0][0].axhline(0.5, ls='--', color = 'grey')
    #        g.savefig('../figures/p_right.png')