import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error

# parameters
# data set selection ('all', 'window', 'weighted')
DATA_SELECTION_ALL = 'all'
DATA_SELECTION_WINDOW = 'window'
DATA_SELECTION_WEIGHTED = 'weighted'
DATA_SELECTION = DATA_SELECTION_ALL

# retrain trigger ('off', 'steps', 'performance')
RETRAIN_TRIGGER_OFF = 'off'
RETRAIN_TRIGGER_STEPS = 'steps'
RETRAIN_TRIGGER_PERF = 'performance'
RETRAIN_TRIGGER = RETRAIN_TRIGGER_OFF

# number of steps to retrain
RETRAIN_STEPS = 1
# performance drop to trigger retraining
RETRAIN_PERF = 0.01

# selection criterion of new model
MODEL_SELECTION_NEW = 'new'
MODEL_SELECTION_PERF = 'performance'
MODEL_SELECTION_P_VAL = 'p-value'
MODEL_SELECTION = MODEL_SELECTION_NEW

# weight decay factors
WEIGHT_DECAYS = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]


def neg_mse(y_true, y_pred):
    return -mean_squared_error(y_true, y_pred)

class ModelTrainer:
    def __init__(self, data_selection=DATA_SELECTION, retrain_trigger=RETRAIN_TRIGGER, retrain_steps=RETRAIN_STEPS,
                 retrain_perf=RETRAIN_PERF, model_selection=MODEL_SELECTION, weight_decays=WEIGHT_DECAYS):
        '''
        initizalize model trainer
        @param data_selection: data set selection ('all': all data, 
                               'window': data within window, 
                               'weighted': all data weighted per step)
        @param retrain_trigger: retrain trigger ('off': no retraining, 
                                'steps': retrain after retrain_steps, 
                                'performance': retrain after performance drop)
        @param retrain_steps: number of steps to retrain
        @param retrain_perf: performance drop to trigger retraining
        @param model_selection: selection criterion of new model ('new': always use new model, 
                                'performance': use new model if performance is better, 
                                'p-value': use new model if perforamnace is significantly better)
        @param weight_decays: weight decay factors for weighted data set selection
        '''
        self.data_selection = data_selection
        self.retrain_trigger = retrain_trigger
        self.retrain_steps = retrain_steps
        self.retrain_perf = retrain_perf
        self.model_selection = model_selection
        self.weight_decays = weight_decays

    def train_models(self, data_df, training_steps, model, metric):
        '''
        train models for each step
        @param data_df: training data set with step index, features, and target
        @param training_steps: number of training steps
        @param model: model to be trained
        @param metric: performance metric to evaluate model
        '''
        # initialize model properties
        self.n_models = 0
        self.init_scores = []
        self.last_training_steps = []
        self.best_model_id = None

        # copy dataframe
        data_df = data_df.copy()

        # number of steps
        n_steps = data_df['step'].max() + 1

        # get feature columns
        self.feature_cols = [col for col in data_df.columns if 'feature' in col]

        # initialize column of model predictions
        n_models = n_steps - training_steps
        model_cols = [self.model_id_to_col(i) for i in range(n_models)]
        model_df = pd.DataFrame(np.ones((len(data_df), n_models)) * np.nan, columns=model_cols)
        data_df = data_df.join(model_df)

        # add column with id of best model
        data_df['best_model_id'] = np.nan
        self.data_df = data_df

        # get validation and test data
        val_df = data_df[data_df['step'] == training_steps - 2]

        # get last step used for training data
        last_training_step = training_steps - 2 if self.model_selection != MODEL_SELECTION_NEW else training_steps - 1 

        # train initial model
        # use window to select training data
        if self.data_selection == DATA_SELECTION_WINDOW:
            # train model on different window sizes and select best model
            best_score = -np.inf
            best_ts = None
            for ts in range(training_steps - 3):
                train_df = data_df[data_df['step'] <= ts]
                model.fit(train_df[self.feature_cols], train_df['target'])
                score = metric(val_df['target'], model.predict(val_df[self.feature_cols]))
                if score > best_score:
                    best_score = score
                    best_ts = ts
            # set optimal window size
            self.window_size = best_ts + 1
            # select training data
            train_df = data_df[(data_df['step'] <= last_training_step) & 
                               (data_df['step'] > last_training_step - self.window_size)]

            model.fit(train_df[self.feature_cols], train_df['target'])
        # use sample weights to select training data
        elif self.data_selection == DATA_SELECTION_WEIGHTED:
            train_df = data_df[data_df['step'] <= last_training_step]
            # weigh training data with different sample weights and select best model
            best_score = -np.inf
            best_decay = None
            for decay in self.weight_decays:  
                # calculate sample weights
                sample_weights = self.get_sample_weights(train_df['step'], decay)            
                model.fit(train_df[self.feature_cols], train_df['target'], sample_weight=sample_weights)
                score = metric(val_df['target'], model.predict(val_df[self.feature_cols]))
                if score > best_score:
                    best_score = score
                    best_decay = decay
            self.weight_decay = best_decay
            # train model on optimal sample weights
            sample_weights = self.get_sample_weights(train_df['step'], best_decay)            
            model.fit(train_df[self.feature_cols], train_df['target'], sample_weight=sample_weights)
        elif self.data_selection == DATA_SELECTION_ALL:
            train_df = data_df[data_df['step'] <= last_training_step]
            model.fit(train_df[self.feature_cols], train_df['target'])
        else:
            raise ValueError('Invalid data selection method')
            
        # evaluate model on test data
        self.best_model_id = self.add_new_model(model, metric, last_training_step)
        # add index of best model
        self.data_df.loc[self.data_df['step'] < training_steps, 'best_model_id'] = self.best_model_id

        # iterate through all steps        
        for i, step in enumerate(range(training_steps, n_steps)):
            # decide if model training shall be triggered
            train_model = True
            if self.retrain_trigger == RETRAIN_TRIGGER_OFF: 
                train_model = False
            elif self.retrain_trigger == RETRAIN_TRIGGER_STEPS and i % self.retrain_steps != 0: 
                train_model = False
            elif self.retrain_trigger == RETRAIN_TRIGGER_PERF:
                # evaluate prediction performance at this step
                eval_df = data_df.loc[data_df['step'] == step]
                eval_score = metric(eval_df['target'], eval_df[self.model_id_to_col(self.best_model_id)])
                # check if eval score is within threshold of initial score
                if self.init_scores[self.best_model_id] - eval_score < self.retrain_perf: 
                    train_model = False
            # check if model shall be trained
            if not train_model:
                # add index of best model to current step
                self.data_df.loc[self.data_df['step'] == step, 'best_model_id'] = self.best_model_id
                continue

            # build new model
            # get last training step of new model
            last_training_step = step - 2 if self.model_selection != MODEL_SELECTION_NEW else step - 1
            if self.data_selection == DATA_SELECTION_WINDOW:
                # get training data
                train_df = data_df[(data_df['step'] <= last_training_step) & 
                                   (data_df['step'] > last_training_step - self.window_size)] 
                model.fit(train_df[self.feature_cols], train_df['target'])
            elif self.data_selection == DATA_SELECTION_WEIGHTED:
                # get training data
                train_df = data_df[data_df['step'] <= last_training_step]
                sample_weights = self.get_sample_weights(train_df['step'], self.weight_decay)            
                model.fit(train_df[self.feature_cols], train_df['target'], sample_weight=sample_weights)
            elif self.data_selection == DATA_SELECTION_ALL:
                train_df = data_df[data_df['step'] <= last_training_step]
                model.fit(train_df[self.feature_cols], train_df['target'])
            else:
                raise ValueError('Invalid data selection method')
            
            # add new model
            new_model_id = self.add_new_model(model, metric, last_training_step)

            # decide if new model shall replace current best model
            if self.model_selection == MODEL_SELECTION_NEW:
                # always use new model
                self.best_model_id = new_model_id
            elif self.model_selection == MODEL_SELECTION_PERF:
                # evaluate model performance of all models
                eval_df = data_df[data_df['step'] == step - 1]
                best_model_id = None
                best_score = -np.inf
                for model_id in range(self.n_models):
                    model_col = self.model_id_to_col(model_id)
                    score = metric(eval_df['target'], eval_df[model_col])
                    if score > best_score:
                        best_score = score
                        best_model_id = model_id
                self.best_model_id = best_model_id
            elif self.model_selection == MODEL_SELECTION_P_VAL:
                # iterate through all models
                for model_id in range(self.n_models):
                    # skip if model id is best model id
                    if model_id == self.best_model_id: continue
                    # get last training step
                    last_training_step = max(self.last_training_steps[model_id], 
                                             self.last_training_steps[self.best_model_id])
                    # get evaluation data
                    eval_df = data_df[(data_df['step'] > last_training_step) & (data_df['step'] < step)]

                    # calculate residuals
                    res_model = (eval_df['target'] - eval_df[self.model_id_to_col(model_id)]).abs()
                    res_best = (eval_df['target'] - eval_df[self.model_id_to_col(self.best_model_id)]).abs()

                    # check if residuals are significantly different
                    _, p_val = ttest_ind(res_model, res_best)
                    if p_val >= 0.05: continue

                    # check which model has better performance
                    score_model = metric(eval_df['target'], eval_df[self.model_id_to_col(model_id)])
                    score_best = metric(eval_df['target'], eval_df[self.model_id_to_col(self.best_model_id)])

                    if score_model > score_best: self.best_model_id = model_id
            else:
                raise ValueError('Invalid model selection method')
            
            # add index of best model to current step
            self.data_df.loc[self.data_df['step'] == step, 'best_model_id'] = self.best_model_id

        # convert best model id to int
        self.data_df['best_model_id'] = self.data_df['best_model_id'].astype(int)

        # add column with predictions of best model
        best_pred = []
        for step in range(n_steps):
            # get best model id
            model_id = self.data_df.loc[self.data_df['step'] == step, 'best_model_id'].values[0]
            # get predictions of best model
            best_pred.append(self.data_df.loc[self.data_df['step'] == step, self.model_id_to_col(model_id)])
        self.data_df['best_pred'] = np.concatenate(best_pred)

        return self.data_df


    def add_new_model(self, model, metric, last_training_step):
        model_id = self.n_models
        self.n_models += 1
        # add model predictions
        model_col = self.model_id_to_col(model_id)
        self.data_df[model_col] = model.predict(self.data_df[self.feature_cols])
        # evaluate model
        eval_df = self.data_df[self.data_df['step'] == last_training_step + 1]
        if not eval_df.empty:
            init_score = metric(eval_df['target'], eval_df[model_col])
        else:
            init_score = np.nan

        self.last_training_steps.append(last_training_step)
        self.init_scores.append(init_score)
        return model_id


    def get_sample_weights(self, steps, decay):
        return (1 - decay) ** (steps.max() - steps) 
        

    @staticmethod
    def model_id_to_col(model_id):
        return 'model_' + str(model_id)
