####### NLP EXPLORE MODULE #######

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from scipy import stats
import nltk

def plot_stacked_all(word_counts):
    '''
    Takes in word_counts df with an 'All' column and produces stacked bar chart for each category for top num_top words in all classes
    '''
    (word_counts.sort_values('All', ascending=False)
     .head(20)
     .apply(lambda row: row/row['All'], axis = 1)
     .drop(columns = 'All')
     .sort_values(by = 'ham')
     .plot.barh(stacked = True, width = 1, ec = 'lightgrey')
    )
    plt.title('Proportion of Total Count for Each Class for 20 Most-Commonly Occuring Words')
    plt.legend(bbox_to_anchor= (1.03,1))
    plt.xlabel('Proportion')
    plt.ylabel('Word');
    
def plot_bar_all(word_counts):
    '''
    Takes in word_counts df with an 'All' column and produces bar chart for each category for top num_top words in all classes
    '''
    (word_counts.sort_values('All', ascending=False)
     .head(20)
     .apply(lambda row: row/row['All'], axis = 1)
     .drop(columns = 'All')
     .sort_values(by = 'ham')
     .plot.barh(width=0.75)
    )
    plt.title('Proportion of Total Count for Each Class for 20 Most-Commonly Occuring Words')
    plt.legend(bbox_to_anchor= (1.03,1))
    plt.xlabel('Proportion')
    plt.ylabel('Word');

def plot_stacked_bar(word_counts, category, num_top = 20, cmap = None):
    '''
    Takes in word_counts df with an 'All' column and produces stacked bar chart for each category for top num_top words specified by category argument
    '''
    plt.figure(figsize=(16, 9))
    plt.rc('font', size=16)
    (word_counts.sort_values(by=category, ascending=False)
     .head(num_top)
     .apply(lambda row: row / row['All'], axis=1)
     .drop(columns='All')
     .sort_values(by=category)
     .plot.barh(stacked=True, width=1, ec='lightgrey', cmap = cmap, alpha = 1))
    plt.legend(bbox_to_anchor= (1.03,1))
    plt.title(f'Proportions of Most Commonly-Occuring {num_top} {category} Readme Words\n')
    plt.xlabel('Proportion')
    plt.ylabel('Word');
    # make tick labels display as percentages
    # plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))
    plt.show();

def plot_horizontal_bar(word_counts, category, num_top = 20, cmap = None):
    '''
    Takes in word_counts df with an 'All' column and produces bar chart for each category for top num_top words specified by category argument
    '''
    plt.figure(figsize=(16, 9))
    plt.rc('font', size=16)
    (word_counts.sort_values(by=category, ascending=False)
     .head(num_top)
     .apply(lambda row: row / row['All'], axis=1)
     .drop(columns='All')
     .sort_values(by=category)
     .plot.barh(width=0.75))
    plt.title(f'Proportions of Most Commonly-Occuring {num_top} {category} Readme Words\n')
    plt.xlabel('Proportion')
    plt.ylabel('Word')
    # make tick labels display as percentages
    # plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))
    plt.show();

def bigram_count(words_list, top_num = 20, lang_name = None):
    '''
    This function takes in a words_list
    Creates bigrams
    Plots the counts on a bar chart 
    Optional arguements to change customization
    - top_num: default 20, shows most common number of bigrams
    '''

    # create bigrams
    ngrams = pd.Series(nltk.bigrams(words_list.split())).value_counts().head(top_num)

    
    # plot bigrams on left subplot
    ngrams.sort_values(ascending = True).plot.barh(alpha = .7, width = .9)
    plt.title(f'Top {top_num} Bigrams: {lang_name}')
    plt.show()

def trigram_count(words_list, top_num = 10, lang_name = None):
    '''
    This function takes in a words_list
    Creates trigrams
    Plots the counts on a bar chart 
    Optional arguements to change customization
    - top_num: default 10, shows most common number of trigrams
    '''

    # create bigrams
    ngrams = pd.Series(nltk.trigrams(words_list.split())).value_counts().head(top_num)

    
    # plot bigrams on left subplot
    ngrams.sort_values(ascending = True).plot.barh(alpha = .7, width = .9)
    plt.title(f'Top {top_num} Trigrams: {lang_name}')
    plt.show()    
    
def select_kbest(X, y, k):
    '''
    Takes in predictors, target, and number of features to select and returns that number of best features based on SelectKBest function
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    return X.columns[kbest.get_support()].tolist()

def rfe(X, y, n):
    '''
    Takes in predictors, target, and number of features to select and returns that number of best features based on RFE function
    '''
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=n)
    rfe.fit(X, y)
    return X.columns[rfe.get_support()].tolist()

def show_rfe_feature_ranking(X, y):
    '''
    Takes in predictors and target and returns feature ranking based on RFE function
    '''
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
    rfe.fit(X, y)
    rankings = pd.Series(rfe.ranking_, index=X.columns)
    return rankings.sort_values()
            
def group_stats(df, target_col, group_by_col):
    '''
    Returns 1-sample t test for groups and pop mean of target_col grouped by group_by_col
    '''
    for group, _ in df.groupby(group_by_col):
        t, p = stats.ttest_1samp(df[target_col][df[group_by_col] == group], df[target_col].mean())
        print(f'----------------\n{group}\n----------------\nT-Statistic: {t:.2f}\nP-value: {p:.3f}')
        if p < 0.01:
            print(f'We REJECT the null hypothesis, the mean {target_col} for the {group} subset is different than the overall population mean.\n')
        else:
            print(f'We FAIL TO REJECT the null hypothesis, the mean {target_col} for the {group} subset is not different than the overall population mean.\n')