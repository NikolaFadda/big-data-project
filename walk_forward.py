# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import numpy as np
import pymongo
from sklearn.model_selection import KFold
import time, os, pickle 
#import   sample_builder, specialized_lexicon #,neural_networks,evaluation
import warnings


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_companies_by_industry(industry):
                    
    sp500_to_dna = {}
    with open('companies_per_industry/'+industry+'.csv', 'r') as mapping:
        for x in mapping.readlines()[1:]:
#            print(repr(x))
            fields = x.strip().split('\t')
            sp500_name = fields[1]
            dna_names = fields[2].split(',')
            sp500_to_dna[sp500_name] = dna_names
            
    #for n in sp500_to_dna:
        #print('\n',n)
        #print(sp500_to_dna[n])
    cs = [sp500_to_dna[c] for c in sp500_to_dna]
    return [item for sublist in cs for item in sublist]
    

"""
Get the ids ('an' field in the json documents) and the dates of all the news included in the specified time interval, where
at least on of the companies passed as parameter is relevant.
'relevance_mode' determines whether we pick the articles where the company appears in relevant_companies or about_companies or in both.
"""
def get_all_ids(companies, min_date, max_date, relevance_mode, mongo_collection):
    print('Tipo di min_date in get_all_ids')
    print(type(min_date))
    print('min date?: ', min_date) #
    min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
    max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
    client = pymongo.MongoClient()
    #db=client['financial_forecast']
    #collection=db[mongo_collection]

    regex= ''
    for a in companies:
        regex+=','+a+',|'
    regex = regex[:-1]
    
    time_all_ids = time.time()
    try:
        if relevance_mode == 'both':
            all_news = client.financial_forecast.get_collection(mongo_collection).find({'$or': [{'company_codes_relevance': {'$regex': regex, '$options':'i'}}, 
                                                                                                {'company_codes_about': {'$regex': regex, '$options':'i'}}]},
                                                                                         {'an':1, 'ingestion_datetime':1, 'date_utc':1})
            
        

        else:    
            all_news = client.financial_forecast.get_collection(mongo_collection).find({'company_codes_'+relevance_mode: {'$regex': regex, '$options':'i'}},
                                                                                            {'an':1, 'ingestion_datetime':1, 'date_utc':1})              

        time_all_ids = time.time() - time_all_ids
        print('Il tempo necessario a fare le query con regex è: ', time_all_ids)
        time_all_ids=time.time()           
        sorted_news = sorted([(r['ingestion_datetime'], r['date_utc'], r['an']) 
                                
                                for r in all_news
                                    
                                    
                                    if (datetime.strptime(r['date_utc'], '%Y-%m-%d %H:%M:%S').date() >= min_date and 
                                      datetime.strptime(r['date_utc'], '%Y-%m-%d %H:%M:%S').date() <= max_date)], 
                                      key = lambda x : x[0])
        #print(len(sorted_news))
        time_all_ids = time.time() - time_all_ids
        print('Il tempo necessario per ordinare è: ', time_all_ids)
        print(str(len(sorted_news)))
        
        time_all_ids = time.time()
        sorted_dates = np.array([datetime.strptime(x[1], '%Y-%m-%d %H:%M:%S').date() for x in sorted_news])
        sorted_ids = np.array([x[2] for x in sorted_news])
        time_all_ids = time.time() - time_all_ids
        print('Il tempo necessario per rendere array con np è: ', time_all_ids)
        return sorted_ids, sorted_dates

    finally:
        client.close()



"""
Splits the data in a walk-forward manner (the samples in each training set are always older than those in the
corresponding test set).
Params:
    - dates: a sorted list of distinct dates
    - split_size: the number of dates contained in each test set
    - n_training_splits: the number of splits used for training (so the size of the training set is always 
      bigger than the test set by an integer factor)
          
Returns:
    two lists of lists, containing the dates included in the training and test set, respectively
"""
def split_dates(dates, split_size=100, n_training_splits=10):
    
    dates_indices = np.array(range(len(dates)))

    splits = []
    n_splits = (len(dates_indices))//split_size
    
    if n_splits < n_training_splits:
        print('Number of indices:', len(dates_indices))
        raise Exception('The number of total splits (%s) is smaller than n_training_splits (%s)' % (n_splits, n_training_splits))
    
    kf = KFold(n_splits=n_splits)
    for x,y in kf.split(dates_indices):
        splits.append(dates_indices[y])
    
    training_splits = []
    for i in range(len(splits)-n_training_splits):
        s = []
        for j in range(i,i+n_training_splits):
            s.extend(splits[j])
        training_splits.append(s)
        
    test_splits = splits[n_training_splits:]
    
    training_dates = []
    test_dates = []
    for training_indices, test_indices in zip(training_splits, test_splits):
        training_dates.append([dates[training_indices[0]], dates[training_indices[-1]]])
        test_dates.append([dates[test_indices[0]], dates[test_indices[-1]]])
    
    return training_dates, test_dates
       

        

def run_walk_forward(companies, mongo_collection='sp30_2013-2019', min_date='2014-01-01', max_date='2019-07-02', 
                     type_of_lexicon='tfidf_x_delta', pos_percentile_lexicon=80, experiment_lexicon=False,
                     relevance_mode='about', n_days_in_splits=50, n_training_splits=10, look_back=30, use_company_embedding=False,
                     cluster_alg='skmeans', n_clusters=3, cluster_max_iter=100, cluster_n_jobs=1, type_of_embedding='sentiment_we',
                     forecast_horizon=1, type_of_delta='delta_percentage_previous_day', delta_time_interval_for_lexicon=1,
                     use_temporal_discount_for_lexicon=False, delta_thresholds_per_class={(-100,-0.5):-1, (-0.5,0.5):0, (0.5,100):1}, 
                     nn_config={}, check_events=False, write_to_file=False, folder_name='experiment'):
    
#    with open('sp30_companies.txt', 'r') as f:
#        sp30_companies = [c.strip() for c in f]
#    for i in range(len(companies)):
#        if companies[i] not in sp30_companies:
#            raise Exception(companies[i], 'is not a company in SP30.')
    
    if mongo_collection not in ('sp30_2013-2019', 'sp30_2018-2019_prova', 'Financial_SP500_2011-2019', 'Information Technology_SP500_2011-2019', 'Industrials_SP500_2011-2019'):
        raise Exception('\'mongo_collection\' does not exist.')
    
    if relevance_mode not in ('relevant', 'about', 'both'):
        raise Exception('\'relevance_mode\' must be one of \'relevant\', \'about\' or \'both\'.')
    
    if cluster_alg not in ('kmeans', 'skmeans', 'kmeans_cosine'):
        raise Exception('\'cluster_alg\' must be one of kmeans, kmeans_cosine or skmeans.')
    
    if type_of_delta not in ('delta', 'delta_percentage', 'delta_percentage_previous_day'):
        raise Exception('\'type_of_delta\' must be one of delta, delta_percentage, delta_percentage_previous_day.')
    
    if type_of_embedding not in ('classic_we', 'sentiment_we', 'specialized_we', 'live_specialized_we'):
        raise Exception('\'type_of_embedding\' must be one of classic_we, sentiment_we, specialized_we or live_specialized_we.')
        
    if type_of_lexicon not in ('tfidf_x_delta', 'only_delta', 'only_tfidf', 'tfidf_x_abs_delta', 'only_abs_delta'):
        raise Exception('type_of_matrix must be one of tfidf_x_delta, only_delta, only_tfidf, tfidf_x_abs_delta or only_abs_delta')
    
    if forecast_horizon < 1:
        raise Exception('\'forecast_horizon\' must be >= 1. Use 1 for next-day-prediction.')
        
    if delta_time_interval_for_lexicon < 1:
        raise Exception('\'delta_time_interval\' must be >= 1. Use 1 for next-day.')
        
    if datetime.strptime(max_date, '%Y-%m-%d') + timedelta(days=forecast_horizon) > datetime.strptime('2019-07-02', '%Y-%m-%d'):
        raise Exception("""'max_date' is too recent with this forecast horizon (%s), because the most recent date on the
              market dataset is 2019-07-02. 'max_date' can be at most %s""" % (forecast_horizon, datetime.strptime('2019-07-02', '%Y-%m-%d')-timedelta(days=forecast_horizon)))
        
    n_classes = len(delta_thresholds_per_class.values())
    for v in delta_thresholds_per_class.values():
        if n_classes == 3 and v not in (-1,0,1):
            raise Exception('The only permissible labels for ternary classifications are -1, 0 and 1.')
        if n_classes == 2 and v not in (0,1):
            raise Exception('The only permissible labels for binary classification are 0 and 1.')
    
    sorted_keys = sorted(delta_thresholds_per_class, key = lambda x : x[0])
    if sorted_keys[0][0] != -100:
        raise Exception('The lower bound of the first interval must be -100.')
    if sorted_keys[-1][1] != 100:
        raise Exception('The upper bound of the last interval must be 100.')
    prev_max_d = -100
    for min_d, max_d in sorted_keys:
        if min_d >= max_d:
            raise Exception('In some delta interval of \'delta_thresholds_per_class\' the lower bound is bigger than or equal to the upper bound.')
        if min_d < prev_max_d:
            raise Exception('There is some overlapping in the delta intervals of \'delta_thresholds_per_class\'.')
        if min_d != prev_max_d:
            raise Exception('There cannot be holes in the specified delta intervals.')
        prev_max_d = max_d

    if write_to_file:
        folder_name = 'results/'+folder_name + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.mkdir(folder_name)
        os.mkdir(folder_name+'/nn_plots')
        os.mkdir(folder_name+'/iter_metrics')
        with open(folder_name + '/parameters.txt', 'w') as output_file:
            output_file.write('Companies: ' + ','.join(companies))
            output_file.write('\nIndustry: ' + industry)
            output_file.write('\nMongoDB collection: ' + mongo_collection)
            output_file.write('\nRelevance mode: ' + relevance_mode)
            output_file.write('\nTime interval: ' + min_date + ' - ' + max_date)
            output_file.write('\nDays in single split: ' + str(n_days_in_splits))
            output_file.write('\nTraining splits: ' + str(n_training_splits))
            output_file.write('\nLook-back: ' + str(look_back))
            output_file.write('\nForecast horizon: ' + str(forecast_horizon))
            output_file.write('\n\nCluster algorithm: ' + cluster_alg) 
            output_file.write('\nN. clusters: ' + str(n_clusters))
            output_file.write('\nCluster max iter: ' + str(cluster_max_iter))
            output_file.write('\n\nType of embedding: ' + type_of_embedding)
            output_file.write('\nType of lexicon: ' + type_of_lexicon)
            output_file.write('\nDelta time interval for lexicon: ' + str(delta_time_interval_for_lexicon))
            output_file.write('\nPositive percentile for lexicon: ' + str(pos_percentile_lexicon))
            output_file.write('\n\nType of delta: ' + type_of_delta)
            output_file.write('\nDelta threshold: ' + str(delta_thresholds_per_class))
            output_file.write('\n\nNeural Network configuration:')
            for param in nn_config:
                output_file.write(param + ': ' + str(nn_config[param]) + '\n')
            output_file.write('\n(Not specified parameters are defaulted to the values specified in the fit_nn method in neural_networks module)')
        
    t0 = time.time()
    
    #get all the news in the whole chosen time interval
    news_ids, news_dates = get_all_ids(companies=companies, min_date=min_date, max_date=max_date, relevance_mode=relevance_mode, mongo_collection=mongo_collection)
    if len(news_ids) == 0:
        raise Exception('No news were retrieved by get_all_ids.')
            
    global word_embedding_model
    
    #get splits of distinct dates of equal size
    distinct_dates = sorted(list(set(news_dates)))
    training_split_dates, test_split_dates = split_dates(distinct_dates[look_back:], split_size=n_days_in_splits, n_training_splits=n_training_splits)
    
    client = pymongo.MongoClient()
    try:
        
        #we'll store the training and test sets from the previous interation of walk-forward, 
        #to speed up the construction of the samples of the training set of the next iteration
        prev_training_centroids = None 
        prev_training_labels = None 
        prev_training_deltas = None 
        prev_training_clusters = None
        prev_training_dates = None
        prev_training_companies = None
               
        prev_test_centroids = None 
        prev_test_labels = None 
        prev_test_deltas = None 
        prev_test_clusters = None
        prev_test_dates = None
        prev_test_companies = None
        
        metrics_array = []
        
        if experiment_lexicon:
            ratios = []
            vol_percs = []
            non_vol_percs = []
            non_vol_zero_percs = 0
            print('\n\nOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n\n')
            print(type_of_lexicon.upper())
            print('Delta time interval:', delta_time_interval_for_lexicon)
            print('PERCENTILE:', pos_percentile_lexicon)
            print('N. training days:', n_training_splits)
            print('Temporal discount:', use_temporal_discount_for_lexicon)
            print('\n\n')
            print(min_date, max_date)
                    
        #the number of splits corresponds to the number of iterations of the walk-forward
        for i in range(len(training_split_dates)):
            #get minimum and maximum date of the news we need to fetch to build the training and test sets, respectively
            #we subtract the lookup from the min_date because we need to include also those news to build the first samples in the set
            min_training_date = training_split_dates[i][0] - timedelta(days=look_back) 
            max_training_date = training_split_dates[i][1]
            
            min_test_date = test_split_dates[i][0] - timedelta(days=look_back)
            max_test_date = test_split_dates[i][1]
            
            
#            print('\n***************\nWalk:', i, '/', len(training_split_dates)-1)
#            print('\n{\nTraining: news from', min_training_date, 'to', max_training_date)
#            print('Test: news from', min_test_date, 'to', max_test_date)
#            print('}')
#            continue
            
            #fetch the news from the db having a date between min and max dates of the split and sort them by date
            training_ids = list([nid for nid, d in zip(news_ids,news_dates) if d >= min_training_date and d <= max_training_date])
            training_news = client.financial_forecast.get_collection(mongo_collection).find({'an': {"$in": training_ids}}).sort([('ingestion_datetime',1)])
            training_news = np.array(list(training_news))
            
            test_ids = list([nid for nid, d in zip(news_ids,news_dates) if d >= min_test_date and d <= max_test_date])
            test_news = client.financial_forecast.get_collection(mongo_collection).find({'an': {"$in": test_ids}}).sort([('ingestion_datetime',1)])
            test_news = np.array(list(test_news))
            
            if experiment_lexicon:
                
#                print('\nTotal training news:', len(training_news))
#                print('Total test news:', len(test_news))
#                
#                volatility_delta_interval = (0.6,100) if delta_time_interval_for_lexicon == 1 else (1,100)
#                non_volatility_delta_interval = (0,0.5) if delta_time_interval_for_lexicon == 1 else (0,0.5)
#                
#                volatility_news = specialized_lexicon.sample_news_by_delta(industry=industry, companies=companies, news_array=test_news, delta_interval=volatility_delta_interval, 
#                                                                           sample_size=100, type_of_delta=type_of_delta, delta_time_interval=delta_time_interval_for_lexicon-1)
#                non_volatility_news = specialized_lexicon.sample_news_by_delta(industry=industry, companies=companies, news_array=test_news, delta_interval=non_volatility_delta_interval, 
#                                                                               sample_size=100, type_of_delta=type_of_delta, delta_time_interval=delta_time_interval_for_lexicon-1)
#                
#                if len(volatility_news) > len(non_volatility_news):
#                    volatility_news = volatility_news[:len(non_volatility_news)]
#                else:
#                    non_volatility_news = non_volatility_news[:len(volatility_news)]
#                
#                if len(volatility_news) == 0:
#                    continue
                
                lexicon = specialized_lexicon.create_lexicon(industry, news_array=training_news, max_df=0.9, min_df=10, pos_percentile=pos_percentile_lexicon,
                                                             type_of_delta=type_of_delta, type_of_lexicon=type_of_lexicon, delta_time_interval=delta_time_interval_for_lexicon-1,
                                                             use_temporal_discount=use_temporal_discount_for_lexicon, stemming_mode='stem', select_mode='about', mode='create_from_news_array')
                
                continue
#                volatility_perc, volatility_words_in_lexicon, volatility_total_words = specialized_lexicon.count_words_in_lexicon(volatility_news, lexicon)
#                non_volatility_perc, non_volatility_words_in_lexicon, non_volatility_total_words = specialized_lexicon.count_words_in_lexicon(non_volatility_news, lexicon)
#                if non_volatility_perc > 0:
#                    ratio = volatility_perc / non_volatility_perc
#                    ratios.append(ratio)
#                    vol_percs.append(volatility_perc)
#                    non_vol_percs.append(non_volatility_perc)
#                else:
#                    non_vol_zero_percs += 1
#                
#                continue
            
            
            if type_of_embedding == 'live_specialized_we':
                training_news, training_lexicon = sample_builder.build_live_specialized_embeddings(training_news, industry, type_of_delta=type_of_delta, delta_time_interval=delta_time_interval_for_lexicon-1,
                                                                                                   pos_percentile=pos_percentile_lexicon, use_temporal_discount=use_temporal_discount_for_lexicon,
                                                                                                   type_of_lexicon=type_of_lexicon, word_embedding_model=word_embedding_model)

            #if we have stored the training and test set from the previous iteration (so this is not the first iteration)
            if type(prev_training_centroids) != type(None):
                #find the index of the previous training set where the date is the same as the min_date of the current split
                for j in range(len(prev_training_dates)):
                    #we need to sum the look_back because here we want the first date of the SAMPLE, not of the news item
                    if prev_training_dates[j] >= (min_training_date + timedelta(days=look_back)):
                        break 
                #the current training set is given by a portion of the previous training set plus the whole previous test set
                training_centroids = np.concatenate([prev_training_centroids[j:], prev_test_centroids])
                training_labels = np.concatenate([prev_training_labels[j:], prev_test_labels])
                training_deltas = np.concatenate([prev_training_deltas[j:], prev_test_deltas])
                training_clusters = np.concatenate([prev_training_clusters[j:], prev_test_clusters])
                training_dates = np.concatenate([prev_training_dates[j:], prev_test_dates])
                training_companies = np.concatenate([prev_training_companies[j:], prev_test_companies])
            #if this is the first iteration, we need to fetch all the news of this split and build the whole training set
            else:                
                #at position i of each array we have the centroids, the clustered news, the date and the relevant company corresponding to
                #the i-th sample, respectively
                (training_centroids, training_labels, training_deltas,
                 training_clusters, training_dates, training_companies) = sample_builder.build_samples(news_items=training_news, companies=companies, industry=industry,
                                                                                        relevance_mode=relevance_mode, look_back=look_back, use_company_embeddings=use_company_embedding,
                                                                                        cluster_alg=cluster_alg, n_clusters=n_clusters, cluster_max_iter=cluster_max_iter, cluster_n_jobs=cluster_n_jobs,
                                                                                        type_of_delta=type_of_delta, delta_thresholds_per_class=delta_thresholds_per_class, 
                                                                                        type_of_embedding=type_of_embedding, forecast_horizon=forecast_horizon-1) 
                                                                                        #we decrease the forecast horizon by 1 so that for the next-day-prediction we set forecast_horizon = 1
            
            print('\n\n0:', len([x for x in training_labels if x == 0]))
            print('1:', len([x for x in training_labels if x == 1]))
#            raise Exception
            
            #store the training set from the previous interation of walk-forward, 
            #to speed up the construction of the samples of the training set of the next iteration
            prev_training_centroids = np.copy(training_centroids)
            prev_training_labels = np.copy(training_labels)
            prev_training_deltas = np.copy(training_deltas)
            prev_training_clusters = np.copy(training_clusters)
            prev_training_dates = np.copy(training_dates)
            prev_training_companies = np.copy(training_companies)
            
            #we always need to build the test set from scratch
            if type_of_embedding == 'live_specialized_we':
                test_news, test_lexicon = sample_builder.build_live_specialized_embeddings(test_news, industry, word_embedding_model=word_embedding_model, 
                                                                                           lexicon=training_lexicon, pos_percentile=pos_percentile_lexicon)

            (test_centroids, test_labels, test_deltas, 
             test_clusters, test_dates, test_companies) = sample_builder.build_samples(news_items=test_news, companies=companies, industry=industry,
                                                                        relevance_mode=relevance_mode, look_back=look_back, use_company_embeddings=use_company_embedding,
                                                                        cluster_alg=cluster_alg, n_clusters=n_clusters, cluster_max_iter=cluster_max_iter, cluster_n_jobs=cluster_n_jobs,
                                                                        type_of_delta=type_of_delta, delta_thresholds_per_class=delta_thresholds_per_class, 
                                                                        type_of_embedding=type_of_embedding, forecast_horizon=forecast_horizon-1)
                                                                        #we decrease the forecast horizon by 1 so that for the next-day-prediction we set forecast_horizon = 1
                        
            #store the test set from the previous interation of walk-forward, 
            #to speed up the construction of the samples of the training set of the next iteration
            prev_test_centroids = np.copy(test_centroids)
            prev_test_labels = np.copy(test_labels)
            prev_test_deltas = np.copy(test_deltas)
            prev_test_clusters = np.copy(test_clusters)
            prev_test_dates = np.copy(test_dates)
            prev_test_companies = np.copy(test_companies)
    
    
#            print('\nTraining set size:', len(training_centroids), '[', training_dates[0], '-', training_dates[-1], ']')
#            print('Test set size:', len(test_centroids), '[', test_dates[0], '-', test_dates[-1], ']')
            
#            print('\n\nTraining lexicon:\n', training_lexicon)
#            print('\n\nTest lexicon\n:', test_lexicon)
#            continue
        
        
            if 'validation_split' in nn_config:
                val_spl = nn_config['validation_split']
            else:
                val_spl = 0.3
            start_validation = int(len(training_centroids) * (1-val_spl))
            n_neg1 = len([x for x in training_labels[:start_validation+1] if x == -1])
            n_0 = len([x for x in training_labels[:start_validation+1] if x == 0])
            n_1 = len([x for x in training_labels[:start_validation+1] if x == 1])
                        
#            for d, comp, lab, delt, clust, cent in zip(training_dates, training_companies, training_labels, training_deltas, training_clusters, training_centroids):
#                print('\n\n*************************')
#                print(d, comp, lab, delt)
##                print('\nClusters:')
##                for k in clust:
##                    print(k)
##                    for n in clust[k]:
##                        print(n[1:])
##                print(cent)
#            raise Exception
            
            if write_to_file:
                nn_config['save_plots'] = True
                nn_config['plots_file_name'] = folder_name+'/nn_plots/iter_'+str(i)+'.jpg'
            
            if nn_config['regression']:
                training_targets = np.copy(training_deltas)
            else:
                training_targets = np.copy(training_labels)
            predictor, history = neural_networks.fit_nn(training_centroids, training_targets, config=nn_config)
            predicted_targets = predictor.predict(test_centroids)
            
            if nn_config['regression']:
                predicted_labels = []
                for pt in predicted_targets:
                    for min_delta, max_delta in delta_thresholds_per_class:
                        if pt >= min_delta and pt < max_delta:
                            predicted_labels.append(delta_thresholds_per_class[(min_delta, max_delta)])
            else:
                predicted_labels = [np.argmax(pt) if np.argmax(pt) != 2 else -1 for pt in predicted_targets]
            
            if write_to_file:
                with open(folder_name+'/predictions.csv', 'a+') as output_file:
                    for tdat, tcom, tdel, tlab, plab, ptar in zip(test_dates, test_companies, test_deltas, test_labels, predicted_labels, predicted_targets):
                        if nn_config['regression']:
                            ptar_string = str(ptar[0])
                        else:
                            ptar_string = ','.join([str(round(x,2)) for x in ptar])
                        output_file.write(tdat.strftime('%Y-%m-%d')+','+tcom+','+str(tdel)+','+str(tlab)+','+str(plab)+','+ptar_string+'\n')
            
            metrics_current_iteration = evaluation.compute_metrics(test_labels, predicted_labels, labels_to_include=sorted(delta_thresholds_per_class.values()), write_to_file=write_to_file, file_name=folder_name+'/iter_metrics/iter_'+str(i)+'.txt')
            if write_to_file:
                with open(folder_name+'/iter_metrics/iter_'+str(i)+'.txt', 'a') as output_file:
                    output_file.write('\n\n')
                    output_file.write('Training set size: ' + str(len(training_centroids)) + ' [' + training_dates[0].strftime('%Y-%m-%d') + ' - ' + training_dates[-1].strftime('%Y-%m-%d') + ']')
                    output_file.write('\nTest set size: ' + str(len(test_centroids)) + ' [' + test_dates[0].strftime('%Y-%m-%d') + ' - ' + test_dates[-1].strftime('%Y-%m-%d') + ']')
                    output_file.write('\n\nOriginal class sizes (without or before oversampling):')
                    output_file.write('\n-1: ' + str(n_neg1))
                    output_file.write('\n0: ' + str(n_0))
                    output_file.write('\n1: ' + str(n_1))


            print('\nMetrics from iter:')
            for m in metrics_current_iteration:
                print(m, metrics_current_iteration[m])
                
            metrics_array.append(metrics_current_iteration)
            
            if check_events:
               evaluation.check_events(test_companies, test_dates, test_labels, test_deltas, test_clusters, predicted_labels, delay=1)
            

        if not experiment_lexicon:
            if write_to_file:
                all_test_labels, all_predicted_labels, all_probabilities = evaluation.get_labels_from_file(folder_name+'/predictions.csv')
                global_metrics = evaluation.compute_metrics(all_test_labels, all_predicted_labels, labels_to_include=sorted(delta_thresholds_per_class.values()), write_to_file=write_to_file, file_name=folder_name+'/global_metrics.txt')
                print('\n\nGLOBAL METRICS:')
                for m in global_metrics:
                    print(m, global_metrics[m])
            
            evaluation.plot_metrics(metrics_array, save_plots=write_to_file, file_name=folder_name+'/metrics_plots.jpg')
        else:
            with open('results/lexicon creation/'+industry+'_percentile_'+str(pos_percentile_lexicon) + '_delta_time_interval_' + str(delta_time_interval_for_lexicon)+ '_n_training_days_' + str(n_training_splits) + '_' + type_of_lexicon + '.txt', 'w') as w:
                
                w.write('\nTime interval: ' + min_date + ' - ' + max_date)
                w.write('\nDays in single split: ' + str(n_days_in_splits))
                w.write('\nTraining splits: ' + str(n_training_splits))
                w.write('\nLook-back: ' + str(look_back))
                w.write('\nType of lexicon: ' + type_of_lexicon)
                w.write('\nDelta time interval for lexicon: ' + str(delta_time_interval_for_lexicon))
                w.write('\nPositive percentile for lexicon: ' + str(pos_percentile_lexicon))
                w.write('\nType of delta: ' + type_of_delta)
                w.write('\n\n')
                for x,y,z in zip(vol_percs, non_vol_percs, ratios):
                    w.write('\n% vol: ' + str(x) + '\t% non-vol: ' + str(y) + '\tratio: ' + str(z))
                w.write('\n\nAVG % vol: ' + str(np.average(vol_percs)))
                w.write('\nAVG % non-vol: ' + str(np.average(non_vol_percs)))
                w.write('\nAVG RATIO:' + str(np.average(ratios)))
                w.write('\n% non-vol was 0 in' + str(non_vol_zero_percs) + ' walks.')
                
            print()
#            for x,y,z in zip(vol_percs, non_vol_percs, ratios):
#                print('% vol:', x, '\t% non-vol:', y, '\tratio:', z)
            print('\nAVG % vol:', np.average(vol_percs))
            print('AVG % non-vol:', np.average(non_vol_percs))
            print('AVG RATIO:', np.average(ratios))
            print('% non-vol was 0 in', non_vol_zero_percs, 'walks.')
        
    finally:
        if 'predictor' in locals():
            del predictor
        if 'word_embedding_model' in locals():
            del word_embedding_model
        t1 = time.time()
        print()
        print(t1-t0, 'seconds')
        
        client.close()
    

if __name__ == "__main__":
    
    #companies = ['MCROST','AMZCOM','APPLC','JEFWUV','GOOG','BKHT','ONLNFR','VISA','JAJOHI','CNYC','EXXN',
    #             'WLMRT','PRGML','NCNBCO','MAINTU','MTCFDN','DSNYW','WALDWC','WDCOLT','PFIZ','PFZRI',
    #             'BELATT','CISCOS','UHELC','SBCATT','SOCAL','CCCCMT','COCA','HOMD','SCHPLO','DQUXRR','NWBC',
    #             'INTL','BOEING','COMCST','PEPSCO','ORCLE','NETFLI','BIGMAC']
    
    
    
#    if type_of_embedding == 'live_specialized_we':
#    print('Loading word2vec model...')
#    with open('word2vec_data/google_word2vec_sentiment.bin', 'rb') as f:
#        word_embedding_model = pickle.load(f)
#    print('Done')
#    else:
#        word_embedding_model = None
        
    
    word_embedding_model = None
#    for (nc, scale_factor) in [(4,1.5), (1,1.5), (4,1.7), (2,1.7), (1,1.7), (4,2), (2,2), (1,2)]:
        
    nn_config = {'layer_type' : 'Dense',
                 'validation_split' : 0.3,
                 'hidden_layers' : (264,64,16), 
                 'flatten_layer' : -1,
                 'activation' : 'relu', 
                 'optimizer' : 'adam', 
                 'dropout' : 0.3, 
                 'balance_method' : None,
                 'learn_rate' : 0.001,
                 'n_epochs' : 300,
                 'batch_size' : 32, 
                 'early_stopping' : True, 
                 'patience' : 50, 
                 'print_plots': True,
                 'save_plots' : True,
                 'regression': False,
                 'use_class_weights' : True,
                 'scale_minority_class_weight': 1,
                 'checkpoint_best_model' : False
                 }
        
#        if nn_config['n_epochs'] < 100 or len(nn_config['hidden_layers']) < 3:
#            print('*******************\n\n_epochs is below 100. Are you sure?\n\n*****************************')
            
#        for iteration in range(3):
    

    industry = 'Financial'
    companies = get_companies_by_industry(industry)
    run_walk_forward(mongo_collection=industry+'_SP500_2011-2019', companies=companies, relevance_mode='both',
                     min_date='2011-01-01', max_date='2019-01-01', look_back=14, forecast_horizon=1, use_company_embedding=False,
                     n_days_in_splits=1, n_training_splits=14, experiment_lexicon=True,
                     delta_thresholds_per_class={(-100,-2):1, (-2,2):0, (2,100):1}, delta_time_interval_for_lexicon=1,
                     cluster_alg='kmeans_cosine', n_clusters=2, cluster_max_iter=300, cluster_n_jobs=-1,
                     type_of_embedding='live_specialized_we', type_of_lexicon='only_delta', pos_percentile_lexicon=90,
                     type_of_delta='delta_percentage_previous_day', use_temporal_discount_for_lexicon=False,
                     nn_config=nn_config, check_events=False, write_to_file=True, 
                     folder_name='prova')
    
    
    industry = 'Industrials'
    companies = get_companies_by_industry(industry)
    run_walk_forward(mongo_collection=industry+'_SP500_2011-2019', companies=companies, relevance_mode='both',
                     min_date='2011-01-01', max_date='2019-01-01', look_back=14, forecast_horizon=1, use_company_embedding=False,
                     n_days_in_splits=1, n_training_splits=14, experiment_lexicon=True,
                     delta_thresholds_per_class={(-100,-2):1, (-2,2):0, (2,100):1}, delta_time_interval_for_lexicon=1,
                     cluster_alg='kmeans_cosine', n_clusters=2, cluster_max_iter=300, cluster_n_jobs=-1,
                     type_of_embedding='live_specialized_we', type_of_lexicon='only_delta', pos_percentile_lexicon=90,
                     type_of_delta='delta_percentage_previous_day', use_temporal_discount_for_lexicon=False,
                     nn_config=nn_config, check_events=False, write_to_file=True, 
                     folder_name='prova')
            

            
"""
ESPERIMENTI

0) about or both? MEGLIO BOTH
1) number of clusters: meglio 1,2 o 3?
2) type of embedding: meglio sentiment o specialized?
3) company embedding: meglio con o senza?
4) lookback: 3, 7, 10 o 14?
"""