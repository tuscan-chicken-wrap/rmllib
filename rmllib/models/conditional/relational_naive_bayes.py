'''
Implements the relational naive bayes model
'''
import pandas
import numpy as np
import scipy
import time
import itertools

from .base import LocalModel

class RelationalNaiveBayes(LocalModel):
    '''
    Basic RNB implementation.  Can do iid learning to collective inference.
    '''
    def __init__(self, estimation_prior=.5, **kwargs):
        '''
        Sets up the NB specific parameters
        '''
        super().__init__(**kwargs)
        self.estimation_prior = estimation_prior
        self.class_log_prior_ = None
        self.feature_log_prob_x_ = None
        self.feature_log_prob_y_ = None
        self.feature_log_prob_y_given_neighbor_x_ = None


    def predict_proba(self, data, rel_update_only=False):
        '''
        Make predictions

        :param data: Network dataset to make predictions on
        '''
        index = data.labels[data.mask.Unlabeled].index

        if not rel_update_only:
            features = None
            if not data.is_sparse_features():
                features = data.features.values
            else:
                features = data.features.to_coo().tocsr()

            un_features = features[data.mask.Unlabeled.values.nonzero()[0]]
                
            self.base_logits = pandas.DataFrame(un_features.dot(self.feature_log_prob_x_.values.T) + self.class_log_prior_.values, index=index)
            base_logits = self.base_logits
        else:
            base_logits = self.base_logits.copy()

        # IID predictions
        if self.infer_method == 'iid':
            base_conditionals = np.exp(self.base_logits.values)

        # Relational IID Predictions
        elif self.infer_method == 'r_iid':
		
			#WHAT DOES THIS EVEN REPRESENT
            all_to_unlabeled_edges = data.edges[data.mask.Unlabeled.nonzero()[0], :]

            # Create basic Y | Y_N counts
            neighbor_counts = pandas.DataFrame(index=base_logits.index, columns=self.feature_log_prob_x_.index)

			#THIS LOOKS IMPORTANT WHAT DOES IT DO???
			#  For each label, 
            for neighbor_value in data.labels.Y.columns:
				# loc - Access a group of rows and columns by label(s) or a boolean array.
                neighbor_counts.loc[:, neighbor_value] = all_to_unlabeled_edges.dot(np.nan_to_num(data.mask.Labeled.values * data.labels.Y[neighbor_value].values))

            base_logits += neighbor_counts.values.dot(self.feature_log_prob_y_.T.values)
            base_conditionals = np.exp(base_logits.values)

        # Relational Join Predictions
        elif self.infer_method == 'r_joint' or self.infer_method == 'r_twohop':
            all_to_unlabeled_edges = data.edges[data.mask.Unlabeled.nonzero()[0], :]

            # Create basic Y | Y_N counts
            neighbor_counts = pandas.DataFrame(index=base_logits.index, columns=self.feature_log_prob_x_.index)

            for neighbor_value in data.labels.Y.columns:
                neighbor_counts.loc[:, neighbor_value] = all_to_unlabeled_edges.dot(data.labels.Y[neighbor_value].values)

            base_logits += neighbor_counts.values.dot(self.feature_log_prob_y_.T.values)
            base_conditionals = np.exp(base_logits.values)

        # Relational Joint Predictions
        predictions = base_conditionals / base_conditionals.sum(axis=1)[:, np.newaxis]

        if self.calibrate:
            logits = scipy.special.logit(predictions[:, 1])
            logits -= np.percentile(logits, data.labels.Y[1].loc[data.mask.Labeled].mean()*100)
            predictions[:, 1] = scipy.special.expit(logits)
            predictions[:, 0] = 1 - predictions[:, 1]

        return predictions

    def predict(self, data):
        '''
        Returns the predicted labels on the dataset

        :param data: Network dataset to make predictions on
        '''
        return np.argmax(self.predict_proba(data), axis=1)

    def fit(self, data, rel_update_only=False):
        if not rel_update_only:
            self.class_log_prior_ = np.log(data.labels.Y[data.mask.Labeled].mean())
            features = None
            if not data.is_sparse_features():
                features = data.features.values
            else:
                features = data.features.to_coo().tocsr()

            
            self.feature_log_prob_x_ = pandas.DataFrame(columns=data.features.columns, dtype=np.float64)
            for ycl in data.labels.Y.columns:
                #print(ycl)
                idx = np.nan_to_num(data.labels.Y[ycl].values * data.mask.Labeled.values).nonzero()[0]
                #print(data.labels.Y[ycl].values.shape)
                #print(np.log(features[idx,:].mean(axis=0)))
                #print(features[idx,:].mean(axis=0))
                currentMeanCalc = features[idx,:].mean(axis=0)
                #print(currentMeanCalc.shape)
                for i in range(0, len(currentMeanCalc)):
                    if currentMeanCalc[i] == 0.0:
                        currentMeanCalc[i] += 0.01
                #print(currentMeanCalc)
				
                self.feature_log_prob_x_.loc[ycl, :] = np.log(currentMeanCalc)
				
            #print(self.feature_log_prob_x_)
				
            #Here you can iterate through each labelled point, and use each feature from its neighbors (edges=1) to get a probability of 
            # Y given X_N.  You can average the probability of these points to get an overall probability of Y given its neighbors
            #End dimensions should be same as this, but the method of acquiring the values will require iterating through each labelled point, its edges, and
            # finally its attributes
            self.feature_log_prob_y_given_neighbor_x_ = pandas.DataFrame(columns=data.features.columns, dtype=np.float64)
            #print(data.mask.Labeled.nonzero()[0])
            edge_matrix = data.edges.todense()

            #print(data.mask.Labeled.values)
            for ycl in data.labels.Y.columns:

                #Initialize the result array for this y-value to all zeroes
                self.feature_log_prob_y_given_neighbor_x_.loc[ycl,:] = np.full_like(features[0,:], 0.0)
                #print(self.feature_log_prob_y_given_neighbor_x_)
                #Iterate through each unlabeled node
                total_nodes = data.mask.Labeled.nonzero()[0].shape[0]
                #print(total_nodes)
                for node in data.mask.Labeled.nonzero()[0]:

                    #This gives the starting node and its neighbors
                    #print(str(node) + "," + str(edge_matrix[node].nonzero()[1]))
                
                    neighbors = edge_matrix[node].nonzero()[1] #Array containing all the neighboring indexes
                    neighbors_idx = np.nan_to_num(data.labels.Y[ycl][neighbors]).nonzero()[0]  #This is all neighbor indexes that have Y = ycl (0 or 1)
                    #TODO: neighbors might also include unlabelled points - maybe remove them from this set.
                    
                    
                    #Calculate the mean of the features for this set of neighbors
                    #print(features.shape)
                    currentMeanCalc = features[neighbors_idx,:].mean(axis=0)
                    #print(currentMeanCalc.shape)
                    for i in range(0, len(currentMeanCalc)):
                        if currentMeanCalc[i] == 0.0:
                            currentMeanCalc[i] += 0.01
                    self.feature_log_prob_y_given_neighbor_x_.loc[ycl, :] += currentMeanCalc
                    
                self.feature_log_prob_y_given_neighbor_x_.loc[ycl,:] = np.log(self.feature_log_prob_y_given_neighbor_x_.loc[ycl,:].divide(total_nodes))
                #print(self.feature_log_prob_y_given_neighbor_x_)
			


        if self.learn_method == 'r_iid':
            lab_to_all_edges = data.edges[data.mask.Labeled.nonzero()[0], :]

            # Create basic Y | Y_N counts
            neighbor_counts = pandas.DataFrame(0, index=self.feature_log_prob_x_.index, columns=self.feature_log_prob_x_.index)
            #print(neighbor_counts)
            for neighbor_value in data.labels.Y.columns:
                neighbor_product = lab_to_all_edges.dot(np.nan_to_num(data.labels.Y[neighbor_value].values) * data.mask.Labeled)
                #print(neighbor_value)
                for labeled_value in data.labels.Y.columns:
                    neighbor_counts.loc[labeled_value, neighbor_value] = np.dot(neighbor_product, data.labels.Y[labeled_value][data.mask.Labeled])
                    #print(labeled_value)
            neighbor_counts = neighbor_counts.div(neighbor_counts.sum(axis=1), axis=0)
			
            self.feature_log_prob_y_ = np.log(neighbor_counts)
			
        # elif self.learn_method == "r_iid2":
            # lab_to_all_edges = data.edges[data.mask.Labeled.nonzero()[0], :]
            # print(lab_to_all_edges)
			
            # self.feature_log_prob_y_given_neighbor_x_ = pandas.DataFrame(columns=data.features.columns, dtype=np.float64)
            # for ycl in data.labels.Y.columns:
                # idx = np.nan_to_num(data.labels.Y[ycl].values * data.mask.Labeled.values).nonzero()[0]
                # #print(np.log(features[idx,:].mean(axis=0)))
                # #print(features[idx,:].mean(axis=0))
                # currentMeanCalc = features[idx,:].mean(axis=0)
                # for i in range(0, len(currentMeanCalc)):
                    # if currentMeanCalc[i] == 0.0:
                        # currentMeanCalc[i] += 0.01
                # #print(currentMeanCalc)
				
                # self.feature_log_prob_x_.loc[ycl, :] = np.log(currentMeanCalc)

            # # Create basic Y | Y_N counts
            # neighbor_counts = pandas.DataFrame(0, index=self.feature_log_prob_x_.index, columns=self.feature_log_prob_x_.index)
            
            # for neighbor_value in data.labels.Y.columns:
                # neighbor_product = lab_to_all_edges.dot(np.nan_to_num(data.labels.Y[neighbor_value].values) * data.mask.Labeled)
                # print(neighbor_product)
                # for labeled_value in data.labels.Y.columns:
                    # neighbor_counts.loc[labeled_value, neighbor_value] = np.dot(neighbor_product, data.labels.Y[labeled_value][data.mask.Labeled])
                    # #print(labeled_value)
            # neighbor_counts = neighbor_counts.div(neighbor_counts.sum(axis=1), axis=0)
			
            # self.feature_log_prob_y_ = np.log(neighbor_counts)

        elif self.learn_method == 'r_joint' or self.learn_method == 'r_twohop':
            lab_to_all_edges = data.edges[data.mask.Labeled.nonzero()[0], :]

            # Create basic Y | Y_N counts
            neighbor_counts = pandas.DataFrame(0, index=self.feature_log_prob_x_.index, columns=self.feature_log_prob_x_.index)

            for neighbor_value in data.labels.Y.columns:
                neighbor_product = lab_to_all_edges.dot(data.labels.Y[neighbor_value].values)

                for labeled_value in data.labels.Y.columns:
                    neighbor_counts.loc[labeled_value, neighbor_value] = np.sum(neighbor_product * data.labels.Y[labeled_value][data.mask.Labeled])

            neighbor_counts = neighbor_counts.div(neighbor_counts.sum(axis=1), axis=0)
            self.feature_log_prob_y_ = np.log(neighbor_counts)
            
        return self
