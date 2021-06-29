#Libraries
from datetime import datetime, timedelta
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pymongo
import os, re, time, sys
import numpy as np
import pandas as pd
import requests

from gensim.parsing import preprocessing as pproc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk import ngrams

#Other scripts and functions
from walk_forward import get_companies_by_industry
#from create_lexicons import process_news_text, process_string
from lexicon_utils import preprocess_tfidf_matrix, csr_to_coo_tuple, get_all_ids, get_market_per_day, process_text


#The following description can be found into the previous code:
#Creates and returns the lexicons defined by the parameters passed to the function.

#Params:
#    - industry : one of Information Technology, Financial or Industrials
#    - collection_name : name of the mongodb collection where the news related to the 'industry' are stored
#    - min_date, max_date : bounds of the time interval of the news based on which the lexicons are created
#    - look_back : length of the time interval used to create the lexicon for a SINGLE day
#    - type_of_lexicon : see process tfidf matrix
#    - max_df : words that appear in more than max_df documents are filtered out 
#      (can be an int indicating the exact number of documents or a float between 0 and 1 indicating the proportion)
#    - min_df : words that appear in less than min_df documents are filtered out 
#      (can be an int indicating the exact number of documents or a float between 0 and 1 indicating the proportion)
#    - ngram_range : The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
#      All values of n such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, 
#      (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams 
#    - stemming : apply stemming to the words
#    - remove_stopwrds : filter out words such as 'the', 'it', 'a', 'if'
#    - positive_percentile_range : the lower and upper bounds of the percentiles used to select the final lexicon from
#      the sorted ranking of words. For example, (90, 100) will select for the positive lexicon the words between
#      the 90th and the 100th percentile.
#    - negative_percentile_range : same as positive_percentile_range, but for negative lexicon
#      !!! PLEASE NOTE !!! set this to (0,0) if you are not interested in positive and negative scores,
#      but only to the absolute values.
#    - excluded_words : calculate the lexicons using the words that would be normally filtered out
#      (you can ignore this and always set to False, onyl used for some internal experiment)
#    - save_in_file : save the lexicons to files.
   
#The saving happens in the following manner:
#    - the folder that contains ALL the lexicons is named using the parameters that define the lexicons;
#      this is used also by fetch_lexicons to check if the specified lexicons exist
#    - the csv file that contains EACH SINGLE lexicon is named after the day for which the lexicon
#      should be used (usually the day following the last news used to create the lexicon)


def create_lexicons_sprk(industry, collection_name, min_date='2013-11-01', max_date='2019-01-01', look_back=28,
                    type_of_lexicon='only_abs_delta', max_df=0.9, min_df=10, ngram_range=(1,3), 
                    stemming=True, remove_stopwords=True,
                    positive_percentile_range=(90,100), negative_percentile_range=(0,10),
                    excluded_words=False,
                    save_in_file=True):

    os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'
    master = requests.get("http://ipconfig.in/ip").text
    master = master[:-1]


#Creating the spark contex
    spark = SparkSession.builder.master("spark://"+master[:-1]+":7077") \
                    .appName('SparkByExamples') \
                    .getOrCreate()
    spark.sparkContext.addPyFile("/home/ubuntu/lexicon_utils.py")
#Setting the path for saving files
    if save_in_file:
        folder_name = (industry + '_' + type_of_lexicon + '_lookback_' + str(look_back) + '_ngrams_' + str(ngram_range) + '_stemming_' + str(stemming) 
                        + '_remove_stopwords_' + str(remove_stopwords) + '_max_df_' + str(max_df) + '_min_df_' + str(min_df) + ('_excluded_words' if excluded_words else ''))
        
        if 'sprk_lexicons/3 classes' not in os.listdir('/home/ubuntu/'):
            os.makedirs('sprk_lexicons/3 classes')

        if folder_name not in os.listdir('sprk_lexicons/3 classes'):
            os.makedirs('sprk_lexicons/3 classes/' + folder_name)


#Conversion of range dates from string to datetime in Y-M-D format 
    min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
    max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
    
    print("Industry: ", industry)
    mongo_collection = industry+'_SP500_2011-2019'
    print("Collection: ", mongo_collection)

#Collection of companies relative to the Industry
    companies = get_companies_by_industry(industry)
    print("Number or companies: ", len(companies))
    
    time_companies_lower = time.time()
    
    companies_rdd = spark.sparkContext.parallelize(companies)
    companies_rdd = companies_rdd.map(lambda x: x.lower())
    companies = companies_rdd.collect()

    time_companies_lower = time.time() - time_companies_lower
    print(time_companies_lower)
    
 
#Extracting market data in a specific time range
    time_market= time.time()
    market_per_day = get_market_per_day(min_date=min_date-timedelta(days=look_back), 
                                max_date=max_date+timedelta(days=10),
                                companies=companies, industry=industry)
    time_market= time.time() - time_market 
    print('Time to extract market per day: ', time_market)
    print("Market length: ", len(market_per_day))


#Fetching all dates and ids of documents sorted in ascending order
    time_all_ids= time.time()
    news_ids= get_all_ids(spark, companies, min_date-timedelta(days=look_back), max_date, look_back, relevance_mode='about', mongo_collection=collection_name)
    time_all_ids= time.time() - time_all_ids
    print("Time to get all the ids and dates in order: ", time_all_ids)

#Obtaining all_news
    time_all_news = time.time()
    client = pymongo.MongoClient()
    all_news = client.financial_forecast.get_collection(collection_name).find({'an': {"$in": news_ids}},{'an':1,'title':1,'snippet':1,'body':1,'date_utc':1,'company_codes_about':1, '_id':0}).sort([('ingestion_datetime',1)])
    time_all_news = time.time() - time_all_news 
    print('Time to query all the news with id in news_ids: ', time_all_news)
    print("Tipo di all news dopo la query: ",type(all_news))

    
    time_rdd = time.time()
    processed_news_rdd = spark.sparkContext.parallelize(all_news).flatMap(lambda x: process_text(x)).filter(lambda x: x['company_codes_about'] in companies)
    processed_news = processed_news_rdd.collect()
    processed_news_rdd.unpersist()
    time_rdd = time.time() - time_rdd
    print('Time to parallelize, process and collect the news: ', time_rdd)
    print("Tipo di processed dopo la collect: ",type(processed_news))

    
    
#Creation 4 lists:
#   documents - contains the processed text of a news realtive to a specific company on a specific day
#   associated_companies - contains the id of the company associated to the news in documents at the same index
#   dates - contains the publishing date of the news in documents at the same index, format (Y-M-D h:m:s)
#   ids - contains the id of the news in documents at the same index

    time_dataframe = time.time()

    df = pd.DataFrame(processed_news)

    time_dataframe = time.time() - time_dataframe
    print("Time to export the news in a dataframe")

#Creation of the lists by pandas dataframe methods
    time_lists = time.time()
    
    documents = df['body'].tolist()
    associated_companies = df['company_codes_about'].tolist()
    dates = df['date_utc'].tolist()
    ids = df['an'].tolist()
  

    time_lists = time.time() - time_lists
    print('Lenght of dates: ',len(dates))
    print('Lenght of documents: ',len(documents))
    print('Lenght of ids: ',len(ids))
    print('Lenght of associated_companies: ', len(associated_companies))
    print("Time to create dates, documents, ids and associated_companies: ", time_lists)



#Extraction of positive and negative lexicons
    time_while = time.time()
    positive_lexicons = {}
    negative_lexicons = {}

    current_date = min_date

    start_index = 0
    while current_date <= max_date:    
        for i in range(start_index, len(documents)):
            #print('i: '+str(i))
            if dates[i] >= current_date - timedelta(days=look_back):
                start_index = i
                break
        for j in range(i, len(documents)):
            #print('j: '+str(j))
            if dates[j] >= current_date:
                break

#Selection of document sublists related to the current day
        selected_documents = documents[i:j]
        selected_associated_companies = associated_companies[i:j]
        selected_dates = dates[i:j]
        selected_ids = ids[i:j]

        print('Lenght of selected_documents: ',len(selected_documents))
        print('Lenght of associated_companies: ',len(selected_associated_companies))
        print('Lenght of selected_dates: ',len(selected_dates))
        print('Lenght of selected_ids: ',len(selected_ids))


        if len(selected_documents) * max_df < min_df:
            #print('Not enough documents on', current_date)
            current_date = current_date + timedelta(days=1)
            continue
        
#Collection of a csr matrix
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(selected_documents)
        
        #print(type(matrix))
        #print('Matrix:', matrix.shape)
        
        features = np.array(vectorizer.get_feature_names())
        #print('Features', features)
        #input("prendi le features")

#This branch is never exectuder, left for adherence to previous code
        if excluded_words:
            all_words = []
            for doc in selected_documents:
                all_words.extend(doc.split())
            all_words = list(set(all_words))
            vocabulary = [f for f in all_words if f not in features]
            vocabulary.extend(stopwords.words('english'))
            vocabulary = list(set(vocabulary))
            vectorizer_excluded_words = TfidfVectorizer(vocabulary=vocabulary)
            matrix = vectorizer_excluded_words.fit_transform(selected_documents)
            features = np.array(vectorizer_excluded_words.get_feature_names())
        
        
#This branch is always executed
        if type_of_lexicon != 'only_tfidf':
            matrix = preprocess_tfidf_matrix(spark,matrix, dates=selected_dates, associated_companies=selected_associated_companies, 
                                          market=market_per_day, last_date=current_date, type_of_lexicon=type_of_lexicon)
        

#The processed csr matrix is averaged:
        nonzeros = np.array([matrix[:,j].count_nonzero() for j in range(matrix.shape[1])]) 
        print('\nNonzeros: ', nonzeros)
        
        matrix_avg = np.asarray(np.sum(matrix, axis=0)).flatten()
        print('\nSum: ',matrix_avg)
        
        matrix_avg /= nonzeros
        print('\nAvg: ', matrix_avg)
                
        sorted_indices = np.asarray(matrix_avg.argsort()[::-1])
        
        ranked_words = [(features[i], matrix_avg[i]) for i in sorted_indices if not np.isnan(matrix_avg[i])]

#Saving of the values in properly named csv files
        if save_in_file:
            with open('sprk_lexicons/3 classes/' + folder_name + '/' + str(current_date) + '.csv', 'w') as writer:
                for word,score in ranked_words:
                    writer.write(word + ',' + str(score) + '\n')
        
        scores = [s for w,s in ranked_words]
        #print(len(scores))
        
        positive_upper_bound = np.percentile(scores, positive_percentile_range[1])
        positive_lower_bound = np.percentile(scores, positive_percentile_range[0])
        negative_upper_bound = np.percentile(scores, negative_percentile_range[1])
        negative_lower_bound = np.percentile(scores, negative_percentile_range[0])
        
        positive_lexicon = [(w,s) for w,s in ranked_words if s > 0 and s <= positive_upper_bound and s >= positive_lower_bound]
        negative_lexicon = [(w,s) for w,s in ranked_words if s <= 0 and s <= negative_upper_bound and s >= negative_lower_bound]
                
        positive_lexicons[current_date] = positive_lexicon
        negative_lexicons[current_date] = negative_lexicon
    
        current_date = current_date + timedelta(days=1)
    

    time_while = time.time()-time_while
    time_total = time_market+time_all_ids+time_all_news+time_rdd+time_dataframe+time_lists+time_while
    print("Total time: ", time_total)
    print()
    print('Time to extract market per day: ', time_market)
    print("Time to get all the ids and dates in order: ", time_all_ids)
    print('Time to query all the news with id in news_ids: ', time_all_news)
    print('Time to parallelize, process and collect the news: ', time_rdd)
    print("Time to export the news in a dataframe", time_dataframe)
    print("Time to create dates, documents, ids and associated_companies: ", time_lists)
    print("Time to loop lists and extract the lexicons", time_while)


    print("\n\n\nTask Completed")
    return positive_lexicons, negative_lexicons

        


create_lexicons_sprk('Financial','Financial_SP500_2011-2019')