import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_feature_cols(df):
    return [col for col in df.columns if 'feature' in col]

def agg_feature(df, first_steps, second_steps):

    # get feature columns
    feature_cols = get_feature_cols(df)

    # select steps
    select_df = df.loc[df['step'].between(first_steps[0], second_steps[1])].copy()

    # normalize features
    scaler = MinMaxScaler()
    select_df[feature_cols] = scaler.fit_transform(select_df[feature_cols])

    # caluculate mean and std of each feature
    first_agg_df = (select_df
                    .loc[select_df['step'].between(first_steps[0], first_steps[1]), feature_cols]
                    .agg(['mean', 'std'])
                    .T
                    .reset_index()
                    .rename(columns={'index': 'feature'}))
    second_agg_df = (select_df
                     .loc[select_df['step'].between(second_steps[0], second_steps[1]), feature_cols]
                     .agg(['mean', 'std'])
                     .T
                     .reset_index()
                     .rename(columns={'index': 'feature'}))
    
    # combine dataframes
    agg_df = pd.merge(first_agg_df, second_agg_df, on='feature', suffixes=('_1', '_2'))

    # calculate mean difference
    agg_df['mean_diff'] = agg_df['mean_2'] - agg_df['mean_1']
    agg_df['std_diff'] = agg_df['std_2'] - agg_df['std_1']

    return agg_df


def get_feature_importance(model, train_df):
    # get feature columns
    feature_cols = get_feature_cols(train_df)

    # train model
    model.fit(train_df[feature_cols], train_df['target'])

    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})

    return imp_df
