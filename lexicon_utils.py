from datetime import datetime, timedelta
import pymongo
import os, re, time
import numpy as np
import pandas as pd
from gensim.parsing import preprocessing as pproc
from pyspark import SparkContext, SparkConf
from scipy.sparse import csr_matrix, coo_matrix
from nltk.corpus import stopwords
from nltk import ngrams

#The following description can be found into the previous code:
#Gets the price variations for the companies passed as parameter in the speficied time interval.
#
#Params:
#    - min_date: first date to start storing price variations
#    - max_date: last date to start storing price variations.
#    - companies: stocks we are interested in
#    - type_of_delta: can be:
#        - delta: the price variation is calculated as close - open
#        - delta_percentage: the price variation is calculated as close - open / open
#        - delta_percentage_previous_day: the price variation is calculated as close_t1 - close_t0 / close_t0
#    - forecast_horizon: number of days we want to look ahead for the forecast.
#      0 means daily prediction (close and open refer to the same day), 1 means the timespan of 2 days is
#      used for the prediction, so close is taken from the following day and open from the current day.
#      In cases where the market is shut, we consider the next useful date in the price dataset which is 
#      equal or bigger than current date + forecast horizon
     
#Returns:
#    market_per_day: a dict indexed by company (1st order key) and date (2nd order key).
#    At market_per_day[c][d] is the price variation wrt d for stock c.

def get_market_per_day(min_date, max_date, companies, industry, type_of_delta='delta_percentage_previous_day', forecast_horizon=0):
 
    #forecast_horizon=0
    #type_of_delta='delta_percentage_previous_day'
#dict that maps company names used in DNA to names used in SP500
    dna_to_sp500 = {}
    with open('companies_per_industry/'+industry+'.csv', 'r') as mapping:
        for x in mapping.readlines()[1:]:
            fields = x.strip().split('\t')
            sp500_name = fields[1]
            dna_names = fields[2].split(',')
            for n in dna_names:
                dna_to_sp500[n] = sp500_name
  
    market_per_day = {}
    for c in companies:
#        print(c)
        market_per_day[c.upper()] = {}
        prices = pd.read_csv('SP500_market_data/'+dna_to_sp500[c.upper()]+'.csv')
#        print('market_data/'+dna_to_sp500[c]+'_daily.csv')
        prices['Date'] = [datetime.strptime(d, '%Y-%m-%d').date() for d in prices['Date']]
      
        dates = prices['Date'].tolist()
        opens = prices['Open'].tolist()
        closes = prices['Adj Close'].tolist()
         #for iterate through dates first with index i and then with j to calculate the delta
        for i in range(len(dates)):
            if dates[i] < min_date:
                continue
            if dates[i] > max_date:
                break
#            if c == 'CCRED':
#                print()
#                print(dates[i])
            for j in range(i, len(dates)):
                #we need >= because, due to closing of the market, there are jumps in the dates.
                #the variation is always taken between the current date (at i) and
                #the first date that is equal or bigger than the current date plus the
                #forecast horizon
                if dates[j] >= dates[i] + timedelta(days=forecast_horizon):
                    if type_of_delta == 'delta':
                        value = closes[j] - opens[i]
#                        print(dates[j], closes[j], '-', opens[i])
                    elif type_of_delta == 'delta_percentage':
                        value = 100*((closes[j] - opens[i]) / opens[i])
                    elif type_of_delta == 'delta_percentage_previous_day':                    
                        value = 100*((closes[j] - closes[i-1]) / closes[i-1])
#                        if c == 'CCRED':
#                            print(dates[j], closes[j], '-', closes[i-1])
                    market_per_day[c.upper()][dates[i]] = value
                    break
    return market_per_day

#Queries the database for the news relative to the Industry companies, selects only the ones in the proper dates
#and orders them by increasing date. It returns two arrays: one with the ordered dates and one with the ids
#
#Params:
#	spark - the spark context necessary for RDD creation and manipulation
#	companies - list of companies relative to the Industry
#	min_date - temporal lower bound for the selection
#	max_date - temporal upper bound for the selection
#	look_back - temporal windown applied to min_date
#	relevance_mode - guides to query to check the record field 'company_codes_relevance'
#	mongo_collection - name of the collection upon which make the queries

def get_all_ids(spark, companies, min_date, max_date, look_back, relevance_mode, mongo_collection):
    time_all_ids = time.time()
    client = pymongo.MongoClient()
    
#Construction of a regular expression with all the companies
    regex= ''
    for a in companies:
        regex+=','+a+',|'
    regex = regex[:-1]
    
    time_query = time.time()

#Query on the collection upon the field 'company_codes_about', if requested also upon 'company_codes_relevance' 
    if relevance_mode == 'both':
        all_news = client.financial_forecast.get_collection(mongo_collection).find({'$or': [{'company_codes_relevance': {'$regex': regex, '$options':'i'}}, 
                                                                                                {'company_codes_about': {'$regex': regex, '$options':'i'}}]},
                                                                                         {'an':1, 'ingestion_datetime':1, 'date_utc':1,'_id':0})
    else:    
        all_news = client.financial_forecast.get_collection(mongo_collection).find({'company_codes_'+relevance_mode: {'$regex': regex, '$options':'i'}},
                                                                                            {'an':1, 'ingestion_datetime':1, 'date_utc':1, '_id':0})              

    time_query = time.time() - time_query
    print('Time to query the MongDB via regex: ', time_query)
    
    time_ndarray = time.time()
    
    all_news = list(all_news)
    print("Length of the list of news: ",len(all_news))
    all_news_rdd = spark.sparkContext.parallelize(all_news)
    time_ndarray = time.time() - time_ndarray
    print('Time necessary to convert to list and load on RDD: ', time_ndarray)

#Selecting the news in the proper time range via date_utc, sort by ingestion_datetime
    time_sorting=time.time()
    all_news_rdd = all_news_rdd.filter(lambda x: ((datetime.strptime(x['date_utc'], '%Y-%m-%d %H:%M:%S').date()>=min_date)\
                                                 and (datetime.strptime(x['date_utc'], '%Y-%m-%d %H:%M:%S').date()<=max_date)))\
                               .sortBy(keyfunc = lambda x: x['ingestion_datetime']).map(lambda x : x['an'])

    sorted_news=all_news_rdd.collect()
    print('tipo di sorted_news prima del cast a list?? ', type(sorted_news))
    #sorted_news = list(sorted_news)
    time_sorting = time.time() - time_sorting
    print('Time necessary to select and sort: ', time_sorting)
    print("Length of the sorted news: ", len(sorted_news))

#Conversion to ndarray and split into sorted_ids and sorted_dates
    time_split=time.time()
    #news_ids = np.array([x['an'] for x in sorted_news])
    time_split = time.time() - time_split

    print('Time to create the separate arrays of ids and dates: ', time_split)
    print("Length of news_ids: ", len(sorted_news))
    print('tipo di sorted_news? ', type(sorted_news))
    #input('quindi?')

    return sorted_news


#Takes a csr matrix and creates the list of related matrix entris in coordinate format (row, column, value)
#
#Params:
#	csr_mat - the original csr matrix

def csr_to_coo_tuple(csr_mat):
    
#Gather the values of row, column indexes and data in that coordinate
    csr_mat = csr_mat.tocoo()
    col = csr_mat.col
    row = csr_mat.row
    data = csr_mat.data
   
#Each triple (row,column,value) is added to the list of entries
    entries = []
    for i in range(len(data)):
        entries.append([row[i],col[i],data[i]])
    
    return entries


#Takes a csr matrix, the companies of a day and a market. Every non-zero element of the matrix is brought to value 1,
#and is multiplied by a coefficient delta. This is difference in stock opening and closing value of a company on a precise date.
#
#Params:
#	spark - the spark context necessary for RDD creation and manipulation
#	dates - sublist of dates
#	associated_companies - sublist of companies cited in the news on that date
#	market - dictionary that contains the variations of stock price of every company for all days
#	last_date - corresponds to the current date, it's necessary to avoid going too further in the future with the valus in dates
#	type_of_lexicons - regulates if nonzero elements are set to 1 or not

def preprocess_tfidf_matrix(spark,matrix, dates, associated_companies, market, last_date, type_of_lexicon='only_delta'):
#The csr format matrix is converterd in a list of coordinate entries, then stored in an RDD    
    [r,c] = matrix.shape
    print("Matrix shape: ",matrix.shape)
    entries= csr_to_coo_tuple(matrix)
    entries_rdd = spark.sparkContext.parallelize(entries)
    
#Every nonzero values is set to 1 for the proper value of type_of_lexicons
    if type_of_lexicon in ('only_delta', 'only_abs_delta'):
        entries_rdd = entries_rdd.map(lambda x: tuple((x[0],x[1],1)) if x[2] != 0 else tuple((x[0],x[1],x[2])))
    
    
    rows_to_remove = []
    coeffs = {}
    print('Length of dates: ',len(dates))
    


    for i in range(len(dates)):
        coeffs[i] = {}
        #print('\n***************\n')
        #print('associated_companies[i]: ', associated_companies[i])
        
        company = associated_companies[i].upper()
        
        #print('company not in market?',company.upper() not in market )
        #print('market[company]==\{\} ?', market[company.upper()] == {})

#If the current company has no market entry, the row is set to be deleted (set to zero)
        if (company not in market) or (market[company] == {}):
            rows_to_remove.append(i)
            continue

#Search for a delta into the market dictionary, if the delta is not found the row will be deleted       
        delta_found = False
        td = 1
        while not delta_found:
            next_date = dates[i] + timedelta(days=td)
            if next_date > last_date:
                break
            try:
                delta = market[company][next_date]
                delta_found = True
            except KeyError:
                td += 1
                if td > 4:  #the 4 is to account for the days in which the market is closed
                    break
        if not delta_found:
            rows_to_remove.append(i)
            continue
        
        
        #print(i,dates[i], company, delta)

#Creation of a dictinary (row: delta) that will be multiplied to the RDD        
        if coeffs[i] == {}:        	
            coeffs[i] = delta
            
        elif coeffs[i] != {}:
            coeffs[i] *= delta
            
        print('\n***************\n')
         
   
    print("Rows to remove: ", rows_to_remove)
    
#The coordinate entries in the RDD are multiplied by the proper coefficient, then filtered out if they are to erase
    entries_rdd =entries_rdd.map(lambda x: (x[0],x[1],x[2] * coeffs[x[0]]) if coeffs[x[0]]!={} else (x[0],x[1],x[2]))\
                            .filter(lambda x: x[2] not in rows_to_remove ) #x[2]!=1
    
    
#The entries are collected, then used to create a coordinate format matrix that will be converted to csr format 
    sel_entr = entries_rdd.collect()
    print("Number of tuples after discaring unnecessary rows: ", len(sel_entr))
    entries_rdd.unpersist()

    data=[]
    row=[]
    col=[]
    for j in sel_entr:
        if  j[0] in rows_to_remove:
            data.append(0)
            row.append(j[0])
            col.append(j[1])
        else:
            data.append(j[2])
            row.append(j[0])
            col.append(j[1])

    coo_mat = coo_matrix((data,(row,col)))
    csr_mat = coo_mat.tocsr()
    print("Matrix shape after removal: ",csr_mat.shape)

    return csr_mat

def process_text(entry, stemming=True, remove_stopwords=True):

    result=[]
    text=''

    if 'title' in entry and type(entry['title']) != type(None):
        text = text + entry['title'] + ' '
    if 'snippet' in entry and type(entry['snippet']) != type(None):
        text = text + entry['snippet'] + ' '
    if 'body' in entry and type(entry['body']) == type('str'):
        text = text + entry['body'] + ' '
    
    text=text.lower()

    for c in entry["company_codes_about"].split(','):
        

        abbreviations = re.findall(r'(?:[a-z]\.)+', text)
        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace('.',''))
            text = pproc.strip_punctuation(text)
#    print('*****************')
#    print(string)
        if remove_stopwords:
            text = pproc.remove_stopwords(text)
#    print()
#    print(string)
        if stemming:
            text = pproc.stem_text(text)
        text = text.strip()
        result.append({'company_codes_about':c,'body':text,'date_utc':datetime.strptime(entry["date_utc"], '%Y-%m-%d %H:%M:%S').date(),'an':entry['an']})
    return result



