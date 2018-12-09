import argparse
import pandas
import numpy.random as random
import sklearn.metrics
import time
pandas.options.mode.chained_assignment = None

from rmllib.data.load import BostonMedians
from rmllib.data.generate import BayesSampleDataset
from rmllib.data.generate import edge_rejection_generator
from rmllib.models.conditional import RelationalNaiveBayes
from rmllib.models.collective_inference import VariationalInference
from rmllib.models.semi_supervised import ExpectationMaximization

from loadData import AcademicPerformance


# Citation for Academic Dataset: P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. 
#In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

# Seed numpy
random.seed(16)

DATASETS = []

chosenFeatures = ["AGE", 'STUDYTIME_0', 'STUDYTIME_1', 'STUDYTIME_2','FAILURE_2', 'FAILURE_3', 'FAILURE_OTHER', "G2"]
DATASETS.append(AcademicPerformance(name='Academic Performance', subfeatures=None, sparse=True).node_sample_mask(.7))

#DATASETS.append(BostonMedians(name='Boston Medians', subfeatures=['RM', 'AGE'], sparse=True).node_sample_mask(.1))
#DATASETS.append(BayesSampleDataset(name='Sparse 1,000,000', n_rows=1000000, n_features=3, generator=edge_rejection_generator, density=.00002, sparse=False).node_sample_mask(.01))

MODELS = []

#Conditional
MODELS.append(RelationalNaiveBayes(name='NB', learn_method='iid', infer_method='iid', calibrate=False))
MODELS.append(RelationalNaiveBayes(name='RNB', learn_method='r_iid', infer_method='r_iid', calibrate=False))
#Collective inference
MODELS.append(VariationalInference(RelationalNaiveBayes)(name='RNB_VI', learn_method='r_iid', calibrate=True))
#Semi Supervised
MODELS.append(ExpectationMaximization(VariationalInference(RelationalNaiveBayes))(name='RNB_EM_VI', learn_iter=3, calibrate=True))

#Mine
#MODELS.append(RelationalNaiveBayes(name='RNB', learn_method='r_iid2', infer_method='r_iid', calibrate=False))

print('Begin Evaluation')
for dataset in DATASETS:
    TRAIN_DATA = dataset.create_training()

    for model in MODELS:
        print('\n' + "(" + dataset.name + ") " + model.name + ": Begin Train")
        train_data = TRAIN_DATA.copy()
        start_time = time.time()
        model.fit(train_data)
        model.listTopKfeatures(train_data, 10)
        print("(" + dataset.name + ") " + model.name, 'Training Time:', time.time() - start_time)
        model.predictions = model.predict_proba(train_data)
        print("(" + dataset.name + ") " + model.name, 'Total Time:', time.time() - start_time)            
        print("(" + dataset.name + ") " + model.name, 'Average Prediction:', model.predictions[:, 1].mean(), 'AUC:', sklearn.metrics.roc_auc_score(dataset.labels.Y[dataset.mask.Unlabeled][1], model.predictions[:, 1]))
        print("(" + dataset.name + ") " + model.name + ": End Train")

print('End Evaluation')