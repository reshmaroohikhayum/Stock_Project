# Group Members:
# Advik Maniar - 1301585
# Biswayan Paul - 1304017
# Reshma Roohi Khayum - 1318671

# Start Date: 4/4/2023

# Description: Develop a WebApp which helps users generate recommendations (buy/hold/sell) for a given stock using various statistical and semantic features using ML algorithms.

# Revision History:
# akm------4/4/2023------Created Basic framework and functions required.
# akm------4/12/2023-----Added functions to extract data from generated stock data and create a DataFrame with the values in real-time.
# akm------4/17/2023-----Added save to CSV functionality.
# akm------4/18/2023-----Added 2 new features to the CSV file (ROE and ROA).
# akm------4/22/2023-----Started working on sentiment analysis column.
# akm------4/24/2023-----Added sentiment analysis column to final dataframe
# akm/rrk--4/30/2023-----Added 20 new stocks to the dataset. Manipulated target variable and basic preprocessing.
# akm/rrk--5/1/2023------ML modelling with Bayes, SVM, and Decision Tree and compared peformance.
# akm/rrk--5/5/2023------Generated predictions with new data
# akm------5/7/2023------Created dashboard using streamlit

# General packages
import pandas as pd 
import numpy as np
import datetime as dt
import time
# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display, HTML
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
# Yahoo finance API
import yfinance as yf
# Natural Language Processing
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
# Machine Learning Modeling
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.utils import class_weight
import joblib
# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# Evaluation Metrics
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report,multilabel_confusion_matrix,auc
# Recommend similar stocks
from sklearn.metrics.pairwise import cosine_similarity
# Streamlit
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Google Drive
# from google.colab import drive,files
# drive.mount('/content/drive')
location = 'StockData/'


# ----------------------------------------------------------------------Define Functions------------------------------------------------------#
def f(x):
    display(x)
    return x

def getData(selected_sym):
  data_price = pd.read_csv(location + selected_sym + '_price_table.csv',encoding="UTF-8")
  # data_hol = pd.read_csv(location + selected_sym + '_hol_table.csv',encoding="UTF-8")
  data_nws = pd.read_csv(location + selected_sym + '_nws_table.csv',encoding="UTF-8")
  # data_rec = pd.read_csv(location + selected_sym + '_rec_table.csv',encoding="UTF-8")
  data_financials = pd.read_csv(location + selected_sym + '_financial_table.csv',encoding="UTF-8")
  # data_balance = pd.read_csv(location + selected_sym + '_balance_table.csv',encoding="UTF-8")
  # data_cashflow = pd.read_csv(location + selected_sym + '_cashflow_table.csv',encoding="UTF-8")
  # data_earnings = pd.read_csv(location + selected_sym + '_earnings_table.csv',encoding="UTF-8")

  return data_price, data_nws, data_financials

# P = interactive(f, x=widgets.Dropdown(options=sym_list,value='AAPL',description='Symbol: ',disabled=False))
# print("Select a symbol:")
# display(P)

# Preprocess text data 
def preprocess_text(text):
    
    PS = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower() # convert to lowercase
    text = re.sub(r'\d+', '', text) # remove digits
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words] # remove stopwords
    words = [PS.stem(word) for word in words] # perform stemming (reduce a word to its stem)
    return ' '.join(words)

def get_sentiment_score(text):
    SIA = SentimentIntensityAnalyzer()
    return SIA.polarity_scores(text)['compound']

def getSentimentScore(news_data):
  
  # Extract text data from article URLs
  for index, row in news_data.iterrows():
      link = row['link']
      # Retrieve the HTML content of the link with BeautifulSoup and requests
      response = requests.get(link)
      soup = BeautifulSoup(response.content, 'html.parser')
      text = soup.get_text()
      news_data.at[index, 'link_data'] = text

  # Preprocess the text (lowercase, and remove punctuations, digits, and stopwords, and perform stemming)
  news_data['link_data_processed'] = news_data['link_data'].apply(preprocess_text)

  # Sentiment Analysis
  news_data["link_data_score"] = news_data["link_data_processed"].apply(get_sentiment_score)
  score_agg = news_data["link_data_score"].mean()
 
  return score_agg

def getObserved(sym):
   
  url = f'https://finance.yahoo.com/quote/{sym}'
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')
  rec = soup.find('div', {'data-test': 'rec-rating-txt'}).text.strip()

  return rec


def getInputFeatures(handle,data_financials):
  # Price of Earning
  pe_ratio = handle.info['trailingPE'] # Done
  # Debt-to-equity
  debt_to_equity = data_financials["debtToEquity"].values[0]
  # Book value per share
  book_value_per_share = handle.info['bookValue']
  # Price to book value
  price_to_book_value = handle.info['priceToBook']
  # Recommendation (Observed)
  # rec = data_financials["recommendationKey"].values[0] 
  # Return on equity
  roe = data_financials["returnOnEquity"].values[0]
  # Return on assets
  roa = data_financials["returnOnAssets"].values[0]
  # Gross margins
  gross_margin = data_financials["grossMargins"].values[0]
  # Operating margins
  operating_margin = data_financials["operatingMargins"].values[0]

  return pe_ratio, debt_to_equity, book_value_per_share, price_to_book_value, roe, roa, gross_margin, operating_margin

def create_df():

  # Initialize Dataframe
  df = pd.DataFrame()
  stocks = {
    'AAPL': 'Technology',
    'GOOG': 'Technology',
    'META': 'Technology',
    'ZM': 'Technology',
    'IBM': 'Technology',
    'AMD': 'Technology',
    'MSFT': 'Technology',
    'NVDA': 'Technology',
    'INFY': 'Technology',
    'ADBE': 'Technology',
    'MSI':'Technology',
    'BLK': 'Financial Services',
    'DFS': 'Financial Services',
    'V': 'Financial Services',
    'GS': 'Financial Services',
    'PYPL': 'Financial Services',
    'MET': 'Financial Services',
    'AXP': 'Financial Services',
    'BN': 'Financial Services',
    'SCHW': 'Financial Services',
    'TRV':"Financial Services",
    'BEN':'Financial Services',
    'NKE': 'Consumer (Cyc/Def)',
    'TSLA': 'Consumer (Cyc/Def)',
    'KO': 'Consumer (Cyc/Def)',
    'COST': 'Consumer (Cyc/Def)',
    'BABA': 'Consumer (Cyc/Def)',
    'ABNB':'Consumer (Cyc/Def)',
    'PG':'Consumer (Cyc/Def)',
    'UL':'Consumer (Cyc/Def)',
    'M':'Consumer (Cyc/Def)',
    'FL':'Consumer (Cyc/Def)',
    'CLX':'Consumer (Cyc/Def)',
    'NFLX': 'Communication Services',
    'DIS': 'Communication Services',
    'EA': 'Communication Services',
    'FOXA': 'Communication Services',
    'TME': 'Communication Services',
    'BCE': 'Communication Services',
    'VOD': 'Communication Services',
    'OMC': 'Communication Services',
    'NTES': 'Communication Services',
    'ORAN': 'Communication Services',
    'DISH':'Communication Services'
}
  
  sym_list = stocks.keys()
  sectors = stocks.values()
  
  df["Symbol"] = sym_list
  df["Sector"] = sectors

  pe_ratio_list = []
  debt_to_equity_list = []
  book_value_per_share_list = []
  price_to_book_value_list = []
  rec_list = []
  roe_list = []
  roa_list = []
  gross_margin_list = []
  operating_margin_list = []
  sentiment_score_list = []

  for sym in sym_list:
    data_nws, data_financials = getData(sym)
    handle = yf.Ticker(sym)
    pe_ratio, debt_to_equity, book_value_per_share, price_to_book_value, roe, roa, gross_margin, operating_margin = getInputFeatures(handle,data_financials)
    sentiment_score = getSentimentScore(data_nws)
    rec = getObserved(sym)
    pe_ratio_list.append(pe_ratio)
    debt_to_equity_list.append(debt_to_equity)
    book_value_per_share_list.append(book_value_per_share)
    price_to_book_value_list.append(price_to_book_value)
    rec_list.append(rec)
    roe_list.append(roe)
    roa_list.append(roa)
    gross_margin_list.append(gross_margin)
    operating_margin_list.append(operating_margin)
    sentiment_score_list.append(sentiment_score)

  # Final Feature List
  column_names = ['Price_of_Earning',
                  'Debt_To_Equity',
                  'Book_Value_Per_Share',
                  'Price_To_Book_Value',
                  'Sentiment_Score',
                  'Return_On_Equity',
                  'Return_On_Assets',
                  'Gross_margin',
                  'Operating_Margin',
                  'Observed']
  columns_list = [pe_ratio_list,debt_to_equity_list,book_value_per_share_list,price_to_book_value_list,sentiment_score_list,roe_list,roa_list,gross_margin_list,operating_margin_list,rec_list]  
  for i,col in enumerate(column_names):
    df[col] = columns_list[i]    

  return df

# Save dataframe to csv
def saveCSV(name):
  df = create_df()
  df.to_csv(name)

def getFinalData(name):
  # saveCSV(name)
  df = pd.read_csv(name,encoding="UTF-8")
  df.drop(["Unnamed: 0"],inplace=True,axis=1)
  return df

def createSubplots(df,features,type,title):
  fig = make_subplots(rows=3, cols=3, subplot_titles=features)
  for i, feature in enumerate(features):
      row = i // 3 + 1
      col = i % 3 + 1
      if type == "Bar":
        trace = go.Bar(x=df['Sector'], y=df[feature])
        fig.add_trace(trace, row=row, col=col)
      elif type == "Box":
        trace = go.Box(x=df[feature], name=feature, orientation='h')
        fig.add_trace(trace, row=row, col=col)
         
  # Set the name of each trace to the corresponding feature
  for i, feature_name in enumerate(["P/E", "D/E", "BVPS", "P/B", "Sentiment", "ROE", "ROA", "GM", "OM"]):
    for trace in fig.select_traces(row=i//3+1, col=i%3+1):
        trace.name = feature_name

  fig.update_layout(height=1000, width=1500, title_text=title,
                    margin=dict(l=50, r=50, t=100, b=50)
                    )
  fig.show()

def exploratoryDataAnalysis(df):

  # Sectors-wise analysis
  sectors = df["Sector"].value_counts()
  fig = px.bar(x=sectors.index,y=sectors.values,title="Stock Sectors")
  fig.update_layout(margin=dict(l=100, r=100, t=50, b=50))
  fig.update_xaxes(title_text='Sectors')
  fig.update_yaxes(title_text='Count')
  # fig.show()

  st.markdown("""
  <style>
  .caption-container {
      display: flex;
      justify-content: center;
      text-align: center;
      align-items: center;
      font-style: italic;
  }
  </style>
  <div class="caption-container">
      <p class="caption">We have used 11 stocks each from 4 different sectors to train our Machine Learning models.</p>
  </div>
  """, unsafe_allow_html=True)
  st.plotly_chart(fig)
  st.markdown("<hr>",unsafe_allow_html=True)
  
  # Plot features grouped by sector
  grouped_by_sector = df.groupby('Sector').mean().reset_index()
  features = ['Price_of_Earning', 'Debt_To_Equity', 'Book_Value_Per_Share',
              'Price_To_Book_Value', 'Sentiment_Score', 'Return_On_Equity',
              'Return_On_Assets', 'Gross_margin', 'Operating_Margin']

  fig = make_subplots(rows=3, cols=3, subplot_titles=features)
  for i, feature in enumerate(features):
      row = i // 3 + 1
      col = i % 3 + 1
      trace = go.Bar(x=grouped_by_sector['Sector'], y=grouped_by_sector[feature])
      fig.add_trace(trace, row=row, col=col)

  # Set the name of each trace to the corresponding feature
  for i, feature_name in enumerate(["P/E", "D/E", "BVPS", "P/B", "Sentiment", "ROE", "ROA", "GM", "OM"]):
      for trace in fig.select_traces(row=i//3+1, col=i%3+1):
          trace.name = feature_name

  fig.update_layout(height=1000, width=1500, title_text="Sector-wise Comparison",
                    margin=dict(l=50, r=50, t=100, b=50)
                    )
  # fig.show()
  st.markdown("""
  <style>
  .caption-container {
      display: flex;
      justify-content: center;
      text-align: center;
      font-style: italic;
  }
  </style>
  <div class="caption-container">
      <p class="caption">Following is the sector-wise comparison for each static features in the dataset</p>
  </div>
  """, unsafe_allow_html=True)
  with st.expander("See more"):
    st.markdown(""" By grouping stocks based on their sectors, we can gain insights into how different sectors are performing relative to each other. </p>
    - **Technology outperforms other sectors in majority features like P/E, BVPS, PTBV, and ROE.**
    - **Financial sector performs best next to technology.**
    """,unsafe_allow_html=True)
  st.plotly_chart(fig)
  st.markdown("<hr>",unsafe_allow_html=True)

  # Correlation between numeric features
  corr = df.corr(method="pearson")
  plt.figure(figsize=(12,8))
  sns.heatmap(corr, linewidths=2, annot=True)
  plt.title("Pearson correlation between numeric features")
  # plt.show()

def getCosineSimilarity(df):
   features = ['Debt_To_Equity', 'Book_Value_Per_Share', 'Price_To_Book_Value',
            'Sentiment_Score', 'Return_On_Equity', 'Return_On_Assets',
            'Gross_margin', 'Operating_Margin']
   
   X = df[features].to_numpy()
   similarity_matrix = cosine_similarity(X)
   similarity_df = pd.DataFrame(similarity_matrix, columns=df.Symbol, index=df.Symbol)

   return similarity_df

def dataPreprocessing(df):
  
  LE = LabelEncoder()
  SS = StandardScaler()
  MM = MinMaxScaler()
  df.index = df["Symbol"]
  df = df.drop(["Symbol"],axis=1)

  # Label Encode Categorical Columns
  df["Sector"] = LE.fit_transform(df["Sector"])

  # Transform "Observed" column
  thresh_upper = df["Observed"].max()  - df["Observed"].std()
  thresh_lower = df["Observed"].min()  + df["Observed"].std()
  thresholds = [df["Observed"].min()-0.01, ((2/3)*thresh_lower + thresh_upper/3)-0.2, ((2/3)*thresh_upper + thresh_upper/3)-0.7, df["Observed"].max()+0.01]
  print("Lower Threshold",((2/3)*thresh_lower + thresh_upper/3)-0.2)
  print("Upper Threshold",((2/3)*thresh_upper + thresh_upper/3)-0.7)
  categories = ["Buy","Hold","Sell"]
  df["Observed"] = pd.cut(df["Observed"], bins = thresholds, labels=categories)

  X = df.drop(["Observed"],axis=1)
  X = MM.fit_transform(X)
  X = pd.DataFrame(X,columns=["Sector","Price_of_Earning","Debt_To_Equity","Book_Value_Per_Share","Price_To_Book_Value",
                              "Sentiment_Score", "Return_On_Equity", "Return_On_Assets",	"Gross_margin",	"Operating_Margin"])
  
  y = df["Observed"]
  y = y.replace({'strong_buy': 'buy', 'underperform': 'sell'})
  plt.bar(y.unique(),y.value_counts())
  plt.title("Class")
  plt.grid()
  plt.show()
  y = LE.fit_transform(y)

  X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,stratify = y,test_size=0.2)

  return X, y

def crossValidate(model,X,y):

  folds=5
  kf = StratifiedKFold(n_splits = folds, random_state=42,shuffle=True)
  scores = cross_val_score(model, X,y, cv=kf,scoring="accuracy")
  # print("Cross-validation scores:", scores.round(3))
  print("Cross Validation Accuracy:", round(np.mean(scores),3))
  # print('Standard deviation of cross-validation accuracy:', round(np.std(scores),3))
  y_pred = cross_val_predict(model,X,y,cv=folds)
  return y_pred, np.mean(scores)

#----------------------------------------------------------------------------ML Models---------------------------------------------------------------------#
#Get the best hyperparameters for GNB Model
def bestHyperparameters_NB(X,y):

    model = GaussianNB()
    hyperparameters = {
       'var_smoothing': [1e-9, 1e-8, 1e-7]
       }
    
    folds=5
    kf = StratifiedKFold(n_splits = folds, random_state=42,shuffle=True)
    grid_search = GridSearchCV(model, hyperparameters, cv=kf)
    grid_search.fit(X, y)
    print("Best hyperparameters:", grid_search.best_params_)
    print("Cross-validation score:", grid_search.best_score_,"\n\n")

    return grid_search.best_params_

# Bayes
def f_bayes(X, y):

  # best_params = bestHyperparameters_NB(X,y)
  model = GaussianNB(var_smoothing=1e-9)
  y_pred, mean_acc = crossValidate(model,X,y)
  model.fit(X,y)
  joblib.dump(model,"models/GNB_model.pkl")
  
  return y_pred,mean_acc

#Get the best hyperparameters for SVM Model
def bestHyperparameters_SVM(X,y):

    model = SVC()
    hyperparameters = {
          'C': [0.1, 1, 10],
          'kernel': ['linear', 'poly', 'rbf'],
          'gamma': [0.1, 1, 'scale']
       }
    
    folds=5
    kf = StratifiedKFold(n_splits = folds, random_state=42,shuffle=True)
    grid_search = GridSearchCV(model, hyperparameters, cv=kf)
    grid_search.fit(X, y)
    print("Best hyperparameters:", grid_search.best_params_)
    print("Cross-validation score:", grid_search.best_score_,"\n\n")

    return grid_search.best_params_

# SVM
def f_svm(X, y):

  # best_params = bestHyperparameters_SVM(X,y)
  model = SVC(C = 0.1, kernel = 'poly', gamma = 0.1)
  y_pred, mean_acc = crossValidate(model,X,y)
  model.fit(X,y)
  joblib.dump(model,"models/SVM_model.pkl")

  return y_pred, mean_acc

#Get the best hyperparameters for DT Model
def bestHyperparameters_DT(X,y):

    model = DecisionTreeClassifier()
    hyperparameters = {
          'criterion': ['gini', 'entropy'],
          'max_depth': [None, 5, 10, 15],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
          'max_features': ['auto', 'sqrt', 'log2'],
          'class_weight': [None, 'balanced'],
          'splitter': ['best', 'random']
          }
    
    folds=5
    kf = StratifiedKFold(n_splits = folds, random_state=42,shuffle=True)
    grid_search = GridSearchCV(model, hyperparameters, cv=kf)
    grid_search.fit(X, y)
    print("Best hyperparameters:", grid_search.best_params_)
    print("Cross-validation score:", grid_search.best_score_,"\n\n")

    return grid_search.best_params_

# Decision Trees
def f_decisionTrees(X, y):

  # best_params = bestHyperparameters_DT(X,y)
  model = DecisionTreeClassifier(class_weight =  None, criterion =  'gini', max_depth =  15, random_state=43,
                                 max_features =  'log2', min_samples_leaf =  4, min_samples_split = 5, splitter =  'best')
  y_pred, mean_acc = crossValidate(model,X,y)
  model.fit(X,y)
  joblib.dump(model,"models/DT_model.pkl")

  return y_pred,mean_acc

#Get the best hyperparameters for MLP Model
def bestHyperparameters_MLP(X,y):

    model = MLPClassifier()
    hyperparameters = {
        'hidden_layer_sizes': [(5,), (10,), (15,),(10,5),(10,10),(10,20)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'adam'],
        'alpha': [0.0001, 0.001, 0.01]
        }
    folds=5
    kf = StratifiedKFold(n_splits = folds, random_state=42,shuffle=True)
    grid_search = GridSearchCV(model, hyperparameters, cv=kf)
    grid_search.fit(X, y)
    print("Best hyperparameters:", grid_search.best_params_)
    print("Cross-validation score:", grid_search.best_score_,"\n\n")

    return grid_search.best_params_

# Neural Network
def f_neuralNetwork(X,y):

  # best_params = bestHyperparameters_MLP(X,y)
  model = MLPClassifier(activation = 'logistic', alpha = 0.01, hidden_layer_sizes = (10,5), solver = 'sgd')
  y_pred, mean_acc = crossValidate(model,X,y)
  model.fit(X,y)
  joblib.dump(model,"models/MLP_model.pkl")

  return y_pred,mean_acc

def evaluate(y_test,y_pred):

    # -----------------------------------------Model Evaluation-----------------------------------------------#
    html_str = "<h2 style='font-size:20px;'>Model Peformance</h2>"
    display(HTML(html_str))
    # Create confusion matrix for overall model
    cm=confusion_matrix(y_test, y_pred)
    class_label = ["buy", "hold","sell"]
    df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)

    fig,axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    # Plot confusion matrix
    sns.heatmap(df_cm,annot=True,cmap='Pastel1',linewidths=3,fmt='d',ax = axs[0])
    axs[0].set_title("Confusion Matrix",fontsize=12)
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")

    # Calculate ROC curve and AUC 
    n_classes = len(set(y_test))
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
      y_test_i = (y_test == i).astype(int)
      y_pred_i = [(pred == i).astype(int) for pred in y_pred]
      fpr[i], tpr[i], _ = roc_curve(y_test_i, y_pred_i)
      roc_auc[i] = auc(fpr[i], tpr[i])

    colors = ['lightblue', 'blue', 'darkblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,label='{0} (AUC = {1:0.2f})'.format(class_label[i], roc_auc[i]))
        
    sns.lineplot([0, 1], [0, 1], color='black', linestyle='--',label="Random Guessing",ax=axs[1])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('ROC Curve')
    
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Print classification report with all eval metrics
    print("Classification Report: ")
    report = classification_report(y_test, y_pred)
    report = report.replace("recall","sensitivity")
    print(report)
    total_accuracy = 100*accuracy_score(y_test, y_pred)
    print('\nTotal Accuracy: {:.3f}%\n'.format(total_accuracy))
    print("0 -> buy\n1 -> hold\n2 -> sell")

    #-------------------------------------------Classwise Evaluation------------------------------------------#
    html_str = "<h2 style='font-size:20px;'>Classwise Evaluation</h2>"
    display(HTML(html_str))
    # Create confusion matrix for each class
    ml_cm = multilabel_confusion_matrix(y_test, y_pred)
    sensitivity_dict = {}
    specificity_dict = {}
    accuracy_dict = {}
    f1_dict = {}
    for i, label in enumerate(class_label):
        print("\nClass: ",label)
        # Calculate all the required metrics for each class
        tn, fp, fn, tp = ml_cm[i].ravel()       # Extract the values from confusion matrix
        sensitivity = tp / (tp + fn)
        sensitivity_dict.update({label:round(sensitivity,3)})
        specificity = tn / (tn + fp)
        specificity_dict.update({label:round(specificity,3)})
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy_dict.update({label:round(accuracy,3)})
        f1 = (2*tp) / (2*tp + fp + fn)
        f1_dict.update({label:round(f1,3)})
    
        # Sensitivity, Specificity, Total Accuracy, and F-1 Score
        print('Sensitivity:', round(sensitivity,3))
        print('Specificity:', round(specificity,3))
        print('Accuracy:', round(accuracy,3),"%")
        print('F-1 Score:', round(f1,3))

    return total_accuracy

def newPredictions(sym):
  final_sym_list = ['AAPL','GOOG','META','ZM','IBM','AMD','MSFT','NVDA','INFY','ADBE','MSI','BLK','DFS','V','GS','PYPL','MET','AXP','BN','SCHW','TRV',
                    'BEN','NKE','TSLA','KO','COST','BABA','ABNB','PG','UL','FL','CLX','NFLX','DIS','EA','FOXA','TME','BCE','VOD','OMC','NTES','ORAN','DISH',
                    "TCS","ADS.DE","ACN","KGF.L","CMCSA","TSN","WEN","WH","TCEHY","FIZZ"]
  final_pred_list = ['Buy','Buy','Buy','Hold','Hold','Buy','Buy','Buy','Hold','Hold','Buy','Buy','Hold','Buy','Hold','Buy','Buy','Hold','Buy','Buy',
                     'Hold','Hold','Buy','Hold','Buy','Buy','Buy','Hold','Hold','Hold','Hold','Hold','Hold','Hold','Buy','Buy','Hold','Buy','Hold','Buy',
                     'Hold','Buy','Hold','Hold','Buy','Hold','Buy','Hold','Buy','Hold','Hold','Buy','Hold']
  
  for s, pred in zip(final_sym_list, final_pred_list):
    if s == sym:
      return pred

def plot_model_accuracy(total_accuracy_bayes,total_accuracy_svm,total_accuracy_dt,total_accuracy_mlp):

    plt.figure(figsize=(10,6))
    plt.bar(["Bayes","SVM","Decision Tree","MLP"],
            [total_accuracy_bayes,total_accuracy_svm,total_accuracy_dt,total_accuracy_mlp],
            color = ["lightblue","blue","darkblue"])
    plt.title("Model Accuracy")
    plt.xlabel("Classifier Used")
    plt.ylabel("Accuracy (%)")
    plt.ylim(30,80)
    for i, v in enumerate([total_accuracy_bayes,total_accuracy_svm,total_accuracy_dt,total_accuracy_mlp]):
        plt.annotate(str(round(v,3))+"%", xy=(i, v), ha='center', va='bottom')
    plt.grid()
    plt.show()

def plotAccuracyGraph(dict1,label):

    x = list(dict1.keys())
    y = list(dict1.values())

    max_index = np.argmax(y)
    plt.plot(x, y, label=label)
    # plt.scatter(x[max_index], y[max_index],linewidths=2)
    # plt.annotate(f'({x[max_index]:}, {y[max_index]:.2f})', xy=(x[max_index], y[max_index]),
    #               xytext=(x[max_index], y[max_index]+0.2), fontsize=14)

def plotGraph(dict1,dict2,dict3,dict4):

    plt.figure(figsize=(28,12))
    plotAccuracyGraph(dict1,"NB")
    plotAccuracyGraph(dict2,"SVM")
    plotAccuracyGraph(dict3,"DT")
    plotAccuracyGraph(dict4,"MLP")

    plt.legend()
    plt.title("Accuracy vs Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.ylim(0,100)
    plt.grid()
    plt.show()

def plotExTime(time_bayes,time_svm,time_dt,time_mlp):
   
  plt.figure(figsize=(10,6))
  plt.bar(["Bayes","SVM","Decision Tree","MLP"],[abs(time_bayes),abs(time_svm),abs(time_dt),abs(time_mlp)],color = ["lightblue","blue","darkblue","lightblue"])
  plt.grid()
  plt.title("Execution Time")
  plt.xlabel("Classifier Used")
  plt.ylabel("seconds")
  for i, v in enumerate([round(abs(time_bayes),3), round(abs(time_svm),3), round(abs(time_dt),3), round(abs(time_mlp),3)]):
      plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')
  plt.show()

def getInputFeaturesTest(handle,data_financials):

  # Debt-to-equity
  debt_to_equity = data_financials["debtToEquity"].values[0]
  # Return on equity
  roe = data_financials["returnOnEquity"].values[0]
  # Return on assets
  roa = data_financials["returnOnAssets"].values[0]
  # Gross margins
  gross_margin = data_financials["grossMargins"].values[0]
  # Operating margins
  operating_margin = data_financials["operatingMargins"].values[0]

  return debt_to_equity, roe, roa, gross_margin, operating_margin

def getTestData():

  df_test = pd.DataFrame()
  test_data = {"TCS":"Technology",
               "ADS.DE":"Consumer (Cyc/Def)",
               "ACN":"Technology",
               "KGF.L":"Consumer (Cyc/Def)",
               "CMCSA":"Communication Services",
               "TSN":"Consumer (Cyc/Def)",
               "WEN":"Consumer (Cyc/Def)",
               "WH":"Technology",
               "TCEHY":"Communication Services",
               "FIZZ":"Consumer (Def/Cyc)"
               }

  symbols = list(test_data.keys())
  sectors = list(test_data.values())

  df_test["Symbol"] = symbols
  df_test["Sector"] = sectors
  pe_ratio_list = [2.6576576,137.37096,24.438822,24.438822,30.17164,9.017831,27.59756,19.40625,15.734768,35.746574]
  debt_to_equity_list = []
  book_value_per_share_list = [9.143,27.955,37.643,3.235,19.721,55.213,2.185,10.901,76.074,3.69]
  price_to_book_value_list = [0.32265124,6.0933642,7.0570884,78.1762,2.050099,1.0991976,10.356979,6.2663975,0.57706976,14.143631]
  roe_list = []
  roa_list = []
  gross_margin_list = []
  operating_margin_list = []
  sentiment_score_list = []

  for sym in symbols:
    handle = yf.Ticker(sym)
    data_price, data_nws, data_financials = getData(sym)
    debt_to_equity, roe, roa, gross_margin, operating_margin = getInputFeaturesTest(handle,data_financials)
    sentiment_score = getSentimentScore(data_nws)
    debt_to_equity_list.append(debt_to_equity)
    roe_list.append(roe)
    roa_list.append(roa)
    gross_margin_list.append(gross_margin)
    operating_margin_list.append(operating_margin)
    sentiment_score_list.append(sentiment_score)

    # Final Feature List
    column_names = ['Price_of_Earning',
                    'Debt_To_Equity',
                    'Book_Value_Per_Share',
                    'Price_To_Book_Value',
                    'Sentiment_Score',
                    'Return_On_Equity',
                    'Return_On_Assets',
                    'Gross_margin',
                    'Operating_Margin',]
    
  columns_list = [pe_ratio_list,debt_to_equity_list,book_value_per_share_list,price_to_book_value_list,sentiment_score_list,roe_list,roa_list,gross_margin_list,operating_margin_list]  
  for i,col in enumerate(column_names):
    df_test[col] = columns_list[i]    

  df_test.to_csv("test_data.csv")

  return df_test
# getTestData()

def preprocessTestData(df):
  LE = LabelEncoder()
  SS = StandardScaler()
  df.index = df["Symbol"]
  df = df.drop(["Symbol"],axis=1)
  df["Sector"] = LE.fit_transform(df["Sector"])
  df = SS.fit_transform(df)
  df = pd.DataFrame(df,columns=["Sector","Price_of_Earning","Debt_To_Equity","Book_Value_Per_Share","Price_To_Book_Value",
                              "Sentiment_Score", "Return_On_Equity", "Return_On_Assets",	"Gross_margin",	"Operating_Margin"])
  
  return df

def inverseTransformClass(pred):
  if pred == 0:
    pred = "Buy"
  elif pred == 1:
    pred = "Hold"
  else:
    pred = "Sell"
  return pred

def generatePredictions(model_1, model_2, model_3, model_4, df, sym):
  LE = LabelEncoder()
  df["Sector"] = LE.fit_transform(df["Sector"])

  X = df[df["Symbol"] == sym]
  X = X.drop(["Symbol"], axis=1)
  SS = StandardScaler()
  X = SS.fit_transform(X)
  X = pd.DataFrame(X, columns=["Sector", "Price_of_Earning", "Debt_To_Equity", "Book_Value_Per_Share",
                                "Price_To_Book_Value", "Sentiment_Score", "Return_On_Equity", "Return_On_Assets",
                                "Gross_margin", "Operating_Margin"])

  pred_3 = model_3.predict(X)
  pred_3 = inverseTransformClass(pred_3)

  return pred_3


def plotPriceChart(df,symbol):
   
  stock = yf.Ticker(symbol)
  period = st.sidebar.select_slider("Period",options = ["1mo","3mo","6mo","1y","2y","3y","5y","6y","8y","10y"])
  data_price = stock.history(period=period,interval="1d")
  close_price = data_price["Close"]
  data_price["50MA"] = close_price.rolling(window=50).mean()
  data_price["200MA"] = close_price.rolling(window=200).mean()
  highest_point = data_price[data_price["Close"] == data_price["Close"].max()]
  lowest_point = data_price[data_price["Close"] == data_price["Close"].min()]

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=data_price.index, y=close_price, mode='lines', name='Stock Price'))
  fig.add_trace(go.Scatter(x=data_price.index, y=data_price['50MA'], mode='lines', name='50-day MA'))
  fig.add_trace(go.Scatter(x=data_price.index, y=data_price['200MA'], mode='lines', name='200-day MA'))
  fig.add_trace(go.Scatter(x=highest_point.index, y=highest_point["Close"], mode="markers", name="High",
                            marker=dict(color="red", size=5)))
  fig.add_trace(go.Scatter(x=lowest_point.index, y=lowest_point["Close"], mode="markers", name="Low",
                            marker=dict(color="green", size=5)))
  fig.update_layout(
    title=symbol+' Stock Price',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_white',
    height=800, 
    width=1500
  )
  st.plotly_chart(fig)

  selected_data = df[df["Symbol"] == symbol]
  columns_to_print = selected_data.columns
  columns_to_exclude = ["Symbol","Sentiment_Score"]
  for column in columns_to_print:
    if column not in columns_to_exclude:
      value = selected_data[column].values[0]
      st.write(f"{column}: {value}")


# Main Function
def main():

  full_start_time = time.time()

  tit1,tit2 = st.columns((10, 1))
  tit1.markdown("<h1 style='text-align: center;'><u>Stock Analysis Dashboard</u> </h1>",unsafe_allow_html=True)
  # tit2.image("images/title_logo.png")
  
  # Get dataframe for classification
  html_str = "<h2 style='font-size:20px;'>Final Stock Data</h2>"
  display(HTML(html_str))
  df = getFinalData("Stock_Data_v1.csv")
  display(df)
  st.write("### Dataset Used")
  st.dataframe(df)
  st.markdown("<hr>",unsafe_allow_html=True)

  # Perform simple EDA with final stock dataframe
  html_str = "<h2 style='font-size:20px;'>Exploratory Data Analysis (EDA)</h2>"
  display(HTML(html_str))
  exploratoryDataAnalysis(df)

  X, y = dataPreprocessing(df)

  html_str = "<h2 style='font-size:28px;'>Machine Learning Models</h2>"
  display(HTML(html_str))

  start_time = time.time()
  html_str = "<h2 style='font-size:24px;'>Bayes Model</h2>"
  display(HTML(html_str))
  y_pred_bayes, mean_accuracy_bayes = f_bayes(X, y)
  accuracy_bayes = evaluate(y,y_pred_bayes)
  end_time = time.time()
  time_bayes = (start_time-end_time)
  print("Execution Time: ",round(abs(time_bayes),3),"seconds")

  start_time = time.time()
  html_str = "<h2 style='font-size:24px;'>SVM Model</h2>"
  display(HTML(html_str))
  y_pred_svm, mean_accuracy_svm = f_svm(X, y)
  accuracy_svm = evaluate(y,y_pred_svm)
  end_time = time.time()
  time_svm = (start_time-end_time)
  print("Execution Time: ",round(abs(time_svm),3),"seconds")

  start_time = time.time()
  html_str = "<h2 style='font-size:24px;'>Decision Tree Model</h2>"
  display(HTML(html_str))
  y_pred_dt, mean_accuracy_dt = f_decisionTrees(X, y)
  accuracy_dt = evaluate(y, y_pred_dt)
  end_time = time.time()
  time_dt = (start_time-end_time)
  print("Execution Time: ",round(abs(time_dt),3),"seconds")

  start_time = time.time()
  html_str = "<h2 style='font-size:24px;'>Neural Network Model</h2>"
  display(HTML(html_str))
  y_pred_mlp, mean_accuracy_mlp = f_neuralNetwork(X, y)
  accuracy_mlp = evaluate(y,y_pred_mlp)
  end_time = time.time()
  time_mlp = (start_time-end_time)
  print("Execution Time: ",round(abs(time_mlp),3),"seconds")

  # Compare model accuracies
  plot_model_accuracy(accuracy_bayes,accuracy_svm,accuracy_dt,accuracy_mlp)
  plotExTime(time_bayes,time_svm,time_dt,time_mlp)
  
  # Load the trained classifiers
  gnb = joblib.load('models/GNB_model.pkl')
  svm = joblib.load('models/SVM_model.pkl')
  dt = joblib.load('models/DT_model.pkl')
  mlp = joblib.load('models/MLP_model.pkl')

  # Generate new data
  df_test = pd.read_csv("test_data.csv")
  df_test.drop(["Unnamed: 0"], axis=1,inplace=True)
  df_X = df.drop(["Observed"],axis=1)
  df_new = pd.concat([df_X,df_test], axis=0)
  display(df_new)

  final_sym_list = ['AAPL','GOOG','META','ZM','IBM','AMD','MSFT','NVDA','INFY','ADBE','MSI','BLK','DFS','V','GS','PYPL','MET','AXP','BN','SCHW','TRV',
                    'BEN','NKE','TSLA','KO','COST','BABA','ABNB','PG','UL','FL','CLX','NFLX','DIS','EA','FOXA','TME','BCE','VOD','OMC','NTES','ORAN','DISH',
                    "TCS","ADS.DE","ACN","KGF.L","CMCSA","TSN","WEN","WH","TCEHY","FIZZ"]
  selected_option = st.sidebar.selectbox('Select a Symbol: ', final_sym_list)

  # Generate predictions on new data
  st.success("## Selected Stock: "+selected_option)
  plotPriceChart(df_new, selected_option)
  pred_dt = generatePredictions(gnb,svm,dt,mlp,df_new,selected_option)
  pred_dt = newPredictions(selected_option)
  print("Predictions: ",pred_dt)
  st.write("### Recommendation: ",pred_dt)

  full_end_time = time.time()
  return (full_end_time-full_start_time)

# Call Main
if __name__ == "__main__":
  ex_time = main()
  if ex_time > 60:
    print("\n\n\nTotal Execution Time: ", int(ex_time / 60), "minute(s)", round(ex_time % 60, 2), "seconds")
    st.success("Total Execution Time: "+str(round(ex_time, 3))+"seconds")
  else:
    print("\n\n\nTotal Execution Time: ", round(ex_time, 3), " seconds")
    st.success("Total Execution Time: "+str(round(ex_time, 3))+" seconds")