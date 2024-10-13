import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot
import plotly.express as px 
from collections import namedtuple

# default parameters
# number of features
N_INDEPENDENT = 10
N_CORRELATED = 20
N_RANDOM = 20   

# number of samples per step
N_SAMPLE_MIN = 50
N_SAMPLE_MAX = 200

# noise to be added 
NOISE_RATIO = 0.1

# names of series
INDI_SERIES_NAME = 'independent'
CORR_SERIES_NAME = 'correlated'
RAND_SERIES_NAME = 'random'
TARGET_SERIES_NAME = 'target'

# feature importance evolution
Importance_evo = namedtuple('Importance_evo', ['const', 'linear', 'seasonal'])
IMPORTANCE_EVO = Importance_evo(const=3, linear=3, seasonal=4)

# feature distribution drift
DIST_DRIFT = True

# number of disturbances per feature
N_DISTURBANCES = 3

# number of black swan events
N_BLACK_SWANS = 3
# impact of black swan events
BLACK_SWAN_IMPACT = 0.75


def target_fcn(x):
    return np.sin(x * 2 * np.pi)

class FeatureGenerator:
    def __init__(self, imp_evo=IMPORTANCE_EVO, n_correlated=N_CORRELATED, n_random=N_RANDOM, n_sample_min=N_SAMPLE_MIN, 
                 n_sample_max=N_SAMPLE_MAX, noise_ratio=NOISE_RATIO, dist_drift=DIST_DRIFT, 
                 n_disturbances=N_DISTURBANCES, n_black_swans=N_BLACK_SWANS, black_swan_impact=BLACK_SWAN_IMPACT):
        """
        initialize the feature generator
        @imp_evo: named tuple describing how feature importance changes over time
        @param n_correlated: number of correlated features
        @param n_random: number of random features
        @param n_sample_min: minimum number of samples per step
        @param n_sample_max: maximum number of samples per step
        @param noise_ratio: ratio of noise to be added to the target input
        @param dist_drift: flag to indicate if feature distribution drift should be applied
        @param n_disturbances: number of disturbances per feature 
        @param n_black_swans: number of black swan events
        @param black_swan_impact: impact of black swan events
        """      
        # number of independent features
        self.n_independent = imp_evo.const + imp_evo.linear + imp_evo.seasonal
        self.imp_evo = imp_evo
        self.n_correlated = n_correlated
        self.n_random = n_random
        self.n_sample_min = n_sample_min
        self.n_sample_max = n_sample_max
        self.noise_ratio = noise_ratio
        self.dist_drift = dist_drift
        self.n_disturbances = n_disturbances
        self.n_black_swans = n_black_swans
        self.black_swan_impact = black_swan_impact


    def generate(self, n_steps, target_fcn, seed=0):
        """
        generate features and target
        @param n_steps: number of steps
        @param target_fcn: target function
        @param seed: random seed
        """
        if self.n_disturbances > n_steps:
            raise ValueError('Number of steps must be at least number of disturbances')  
        if self.n_black_swans > n_steps:
            raise ValueError('Number of steps must be at least number of black swans')
        self.n_steps = n_steps

        # set random seed
        np.random.seed(seed)

        # number of samples per step
        n_samples = np.random.randint(self.n_sample_min, self.n_sample_max, n_steps)
        # total number of samples
        n_total = np.sum(n_samples)
        # sample index
        self.idx = np.array([i for i, n in enumerate(n_samples) for _ in range(n)]).reshape(-1, 1)

        # generate black swan event indices
        if self.n_black_swans > 0:
            self.black_swan_idx = np.sort(np.random.choice(range(self.n_steps), self.n_black_swans, replace=False))
        else:
            self.black_swan_idx = np.array([])

        # generate independent features
        self.independent = self.init_rand_feature(n_samples, self.n_independent)

        # generate correlated features
        corr_lst = []
        stable_intervals = self.get_stable_intervals()
        for _ in range(self.n_correlated):
            # select features
            select = np.random.choice([False, True], self.n_independent)
            n_select = np.sum(select)
            # create multipliers
            multipliers = np.random.uniform(low=-1, high=1, size=n_select)
            
            corr_feature_lst = []
            idx_start = 0
            for i in range(len(stable_intervals) - 1):
                idx_end = idx_start + np.sum(n_samples[stable_intervals[i]:stable_intervals[i+1]])
                
                # calculate correlated feature of this interval
                corr_feature_lst.append(np.matmul(self.independent[idx_start:idx_end, select], multipliers).reshape(-1, 1))
                
                # update multipliers
                multipliers *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, n_select)  
                # update start index
                idx_start = idx_end

            # concatenate all intervals
            corr = np.concatenate(corr_feature_lst, axis=0)

            # create noise
            noise_mean = np.mean(corr)
            noise_std = np.std(corr)
            noise = np.random.normal(noise_mean, noise_std, (n_total, 1))

            # add noise to correlated feature            
            corr_lst.append(corr + noise)

        # concatenate correlated features
        self.correlated = np.concatenate(corr_lst, axis=1)

        # generate random features
        self.random = self.init_rand_feature(n_samples, self.n_random)

        # create feature importance
        importance_lst = []
        idx_start = 0
        # constant importance parameter
        const = np.random.rand(self.imp_evo.const)
        # linear importance parameter
        start_imp = np.random.rand(self.imp_evo.linear)
        end_imp = np.random.rand(self.imp_evo.linear)    
        # seasonal importance parameter
        freq = np.random.uniform(low=1, high=10, size=self.imp_evo.seasonal)
        phase = np.random.uniform(low=0, high=2*np.pi, size=self.imp_evo.seasonal)
        for i in range(len(stable_intervals) - 1):            
            # number of samples in this interval
            n_sample = np.sum(n_samples[stable_intervals[i]:stable_intervals[i+1]]) 
            idx_end = idx_start + n_sample
            idx = self.idx[idx_start:idx_end]
        
            # constant importance
            const_imp = np.ones((n_sample, self.imp_evo.const)) * const

            # update constant importance parameter
            const *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, self.imp_evo.const)

            # linear importance
            linear_imp = ((1 - (idx + 1) / n_steps) * start_imp * np.ones((n_sample, self.imp_evo.linear)) + 
                          (idx + 1) / n_steps * end_imp * np.ones((n_sample, self.imp_evo.linear)))
            
            # update start and end importance
            start_imp *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, self.imp_evo.linear)
            end_imp *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, self.imp_evo.linear)

            # seasonal importance
            seasonal_imp = np.sin(idx * 2 * np.pi / n_steps * freq + phase) / 2 + 0.5

            # update frequency and phase
            freq *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, self.imp_evo.seasonal)
            phase *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, self.imp_evo.seasonal)

            # combine importances
            importance_lst.append(np.concatenate([const_imp, linear_imp, seasonal_imp], axis=1))

            idx_start = idx_end

        # concatenate importance
        self.importance = np.concatenate(importance_lst, axis=0)
        
        # calculate target input
        feature_sum = (self.independent * self.importance).sum(axis=1)
        # normalize
        feature_norm = (feature_sum - feature_sum.min()) / (feature_sum.max() - feature_sum.min())
        target_in = (1-self.noise_ratio) * feature_norm + self.noise_ratio * np.random.rand(n_total) 

        # generate target 
        self.target = target_fcn(target_in).reshape(-1, 1)

        # disturb features
        self.disturb_features(self.independent)
        self.disturb_features(self.correlated)      

               
    def init_rand_feature(self, n_samples, n_feature):
        if not self.dist_drift:
            feature_lst = []
            # initialize mean and std
            mean = np.random.randn(n_feature)
            std = np.random.rand(n_feature)

            stable_intervals = self.get_stable_intervals()

            for i in range(len(stable_intervals) - 1):
                interval_start = stable_intervals[i]
                interval_end = stable_intervals[i + 1]
                # number of samples in this interval
                n_total = np.sum(n_samples[interval_start:interval_end])
                feature_lst.append(np.random.normal(mean, std, (n_total, n_feature)))
                # update mean and std for the next interval
                mean *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, n_feature)
                std *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, n_feature)
        else:
            # generate features with distribution drift
            mean_start = np.random.randn(n_feature)
            mean_end = np.random.randn(n_feature)
            std_start = np.random.rand(n_feature)
            std_end = np.random.rand(n_feature)
            feature_lst = []
            for i, n_sample in enumerate(n_samples):
                # check if interval is black swan event
                if i in self.black_swan_idx:
                    # update mean and std
                    mean_start *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, n_feature)
                    mean_end *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, n_feature)
                    std_start *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, n_feature)
                    std_end *= np.random.uniform(1 - self.black_swan_impact, 1 + self.black_swan_impact, n_feature)
                # calculate mean and std for the step
                mean = (1 - (i + 1) / self.n_steps) * mean_start + (i + 1) / self.n_steps * mean_end
                std = (1 - (i + 1) / self.n_steps) * std_start + (i + 1) / self.n_steps * std_end
                feature_lst.append(np.random.normal(mean, std, (n_sample, n_feature)))
        
        return np.concatenate(feature_lst, axis=0)

    def get_stable_intervals(self):
        # generate intervals of stable distribution (in between black swan events)
        if self.n_black_swans > 0:
            stable_intervals = self.black_swan_idx
            if self.black_swan_idx[0] > 0:
                stable_intervals = np.insert(stable_intervals, 0, 0)
                stable_intervals = np.append(stable_intervals, self.n_steps)
        else:
            stable_intervals = np.array([0, self.n_steps])
        return stable_intervals


    def disturb_features(self, features):
        if self.n_disturbances == 0:
            return features
        
        # number of features
        n_feature = features.shape[1]
        # total number of disturbances
        total_disturbances = self.n_disturbances * n_feature
        # get number of disturbances per feature
        n_disturbe_feature = np.bincount(np.random.randint(0, n_feature, total_disturbances))
        # iterate through features
        for i_feature, n_disturbance in enumerate(n_disturbe_feature):
            if n_disturbance == 0: continue
            # sample disturbed indices
            disturb_idx = np.random.choice(range(self.n_steps), n_disturbance, replace=False)
            for i_step in disturb_idx:
                # number of samples
                n_sample = np.sum(self.idx == i_step)
                # generate mean and std
                mean = np.random.randn()
                std = np.random.rand()
                features[(self.idx == i_step).flatten(), i_feature] = np.random.normal(mean, std, n_sample)


    def to_df(self):
        # create data set
        data_df = pd.DataFrame({'step': self.idx.flatten()})
        # add features
        features = np.concatenate([self.independent, self.correlated, self.random], axis=1)
        n_features = features.shape[1]
        feature_cols = ['feature_' + str(i) for i in range(n_features)]
        feature_df = pd.DataFrame(features, columns=feature_cols)
        data_df = data_df.join(feature_df)

        # add target
        data_df['target'] = self.target

        return data_df
        

    def plot_series(self, series=[INDI_SERIES_NAME, CORR_SERIES_NAME, RAND_SERIES_NAME, TARGET_SERIES_NAME]):
        """
        create plotly subplots for the generated features and target
        @param series: list of series to be plotted
        return: plotly figure
        """
        # create subplots
        fig = make_subplots(rows=len(series), cols=1, subplot_titles=series)
        # iterate through series
        for i, s in enumerate(series):
            # get data
            data = self.get_series_data(s)
            # plot data
            for j in range(data.shape[1]):
                fig.add_trace(go.Scatter(y=data[:, j], mode='lines', name=f'{s}_{j}'), row=i+1, col=1)

        return fig    
    
    def distplot(self, series_name=INDI_SERIES_NAME):
        """
        create plotly histogram for the generated features and target
        @param series_name: series to be plotted
        return: plotly figure
        """
        data = self.get_series_data(series_name)

        labels = self.get_labels(series_name)
        hist_data = [data[:, i] for i in range(data.shape[1])]
        fig = create_distplot(hist_data, group_labels=labels, show_hist=False, show_rug=False)

        return fig
    
    def plot_agg_per_step(self, series_name=INDI_SERIES_NAME, agg_fcn=np.mean):
        """
        create plotly line chart for the average target per step
        @param series_name: series to be plotted
        @param agg_fcn: aggregation function
        return: plotly figure
        """
        # get data and labels
        data = self.get_series_data(series_name)
        labels = self.get_labels(series_name)

        # create dataframe and calcualte aggregate per step
        df = pd.DataFrame(data, columns=labels)
        df['idx'] = self.idx
        agg_df = df.groupby('idx').agg(agg_fcn).reset_index()
        
        # plot
        fig = px.line(agg_df, x='idx', y=labels, title=f'{agg_fcn.__name__} {series_name} per step')

        # add black swan events
        for i in self.black_swan_idx:
            fig.add_vline(x=i, line_dash='dash', line_color='red', annotation_text='Black Swan Event')

        return fig
    
    def plot_correlation(self):
        """
        create plotly heatmap for the correlation matrix
        return: plotly figure
        """
        # create dataframe
        labels = (self.get_labels(INDI_SERIES_NAME) + self.get_labels(CORR_SERIES_NAME) + 
                  self.get_labels(RAND_SERIES_NAME))
        df = pd.DataFrame(np.concatenate([self.independent, self.correlated, self.random], axis=1), 
                          columns=labels)
        # calculate correlation matrix
        corr = df.corr()
        # plot
        fig = px.imshow(corr, title='Correlation Matrix')
        
        return fig
    
    def plot_target_corr_series(self, series_name=INDI_SERIES_NAME):
        """
        plot correlation of series with target aggergated per step
        @param series_name: name of the series
        return: plotly figure
        """
        # get data and labels
        data = self.get_series_data(series_name)
        labels = self.get_labels(series_name)

        # create dataframe and calcualte aggregate per step
        df = pd.DataFrame(data, columns=labels)
        df['idx'] = self.idx
        target = pd.Series(self.target[:, 0], name='target', index=df.index)
        corr_df = df.groupby('idx').corrwith(target).reset_index()
        
        # plot
        fig = px.line(corr_df, x='idx', y=labels, title=f'Correlation with target {series_name} per step')

        # add black swan events
        for i in self.black_swan_idx:
            fig.add_vline(x=i, line_dash='dash', line_color='red', annotation_text='Black Swan Event')

        return fig

    
    def get_labels(self, series_name):
        """
        get the labels for the series
        @param series_name: name of the series
        return: list of labels
        """
        return [f'{series_name}_{i}' for i in range(self.get_series_data(series_name).shape[1])]
            
    
    def get_series_data(self, series_name):
        """
        get the data for a given series
        @param series_name: name of the series
        return: data for the series
        """
        if series_name == INDI_SERIES_NAME:
            return self.independent
        elif series_name == CORR_SERIES_NAME:
            return self.correlated
        elif series_name == RAND_SERIES_NAME:
            return self.random
        elif series_name == TARGET_SERIES_NAME:
            return self.target
        else:
            raise ValueError('Invalid series name')
 
    