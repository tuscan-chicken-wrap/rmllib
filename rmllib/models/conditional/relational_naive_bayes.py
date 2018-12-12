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
        self.top_k_own_features = None
        self.top_k_neighbor_features = None


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
            #print(self.base_logits)
            base_logits = self.base_logits
        else:
            base_logits = self.base_logits.copy()

        # IID predictions
        if self.infer_method == 'iid':
            base_conditionals = np.exp(self.base_logits.values)

        # Relational IID Predictions
        elif self.infer_method == 'r_iid':
		
            all_to_unlabeled_edges = data.edges[data.mask.Unlabeled.nonzero()[0], :]

            # Create basic Y | Y_N counts
            neighbor_counts = pandas.DataFrame(index=base_logits.index, columns=self.feature_log_prob_x_.index)

			#  For each label (0 or 1), 
			# Get the number of neighbors for each label for each data point (i.e. data point 0 has 2 neighbors with label 0, and 3 neighbors with label 1)
            for neighbor_value in data.labels.Y.columns:
				# loc - Access a group of rows and columns by label(s) or a boolean array.
                neighbor_counts.loc[:, neighbor_value] = all_to_unlabeled_edges.dot(np.nan_to_num(data.mask.Labeled.values * data.labels.Y[neighbor_value].values))

            
            #EXTENSION TO INFERENCE - add Pr(X_neighbors | Y)Pr(Y) to the inference
		    
            edge_matrix = data.edges.todense()
            temp = []  # Each entry is the dot product of a data point's summed neighboring features with the Pr(X_neighbors) and then adding Pr(Y)
            for node in data.mask.Unlabeled.nonzero()[0]:
                neighbors = edge_matrix[node].nonzero()[1] #Array containing all the neighboring indexes
                temp.append(np.sum(data.features.values[neighbors], axis=0).dot(self.feature_log_prob_y_given_neighbor_x_.values.T) + (len(neighbors)) * self.class_log_prior_.values)
            temp = np.array(temp)

            base_logits += neighbor_counts.values.dot(self.feature_log_prob_y_.T.values)  #Add Pr(Y | Y_neighbors)

            base_logits += temp

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
        base_conditionals += 5e-13
        predictions = base_conditionals / base_conditionals.sum(axis=1)[:, np.newaxis]
        #print(predictions)

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
		

	
    #Lists the top K features for each output label (0 or 1)
    def listTopKfeatures(self, data, k):
	
        featuresArray = np.full_like(data.features.values[0,:], 0.0)
		
		#Basically sum the Pr(X | Y) of each data point for each label 0 or 1
		# This allows us to iterate through the featuresArray to figure out what the max k values are.
        for ycl in data.labels.Y.columns:
            featuresArray = np.add(featuresArray, self.feature_log_prob_x_.loc[ycl,:].values)
        ind = np.argpartition(featuresArray, -k)[-k:]  #Get the indices of the k largest values for prob x

        #print(data.features.columns.levels[0])  #This is the names of the features
        #print(data.features.columns.labels[0][ind])  #This gives the index of the feature list (since features are true or false, this is double the size of the feature list)
        #print(data.features.columns.labels[1][ind])  #This gives whether the feature is present or not
        print("\nPROB Y GIVEN OWN FEATURES")
        self.top_k_own_features = data.features.columns.levels[0][data.features.columns.labels[0][ind]]
        print(self.top_k_own_features)  #This gives us the highest valued K features for this output (0 or 1)
        print(data.features.columns.labels[1][ind])  #This gives whether the feature is present or not
		
        featuresArray = np.full_like(data.features.values[0,:], 0.0)
		
		#This is the same as before, only summing over Pr(X_neighbors | Y) instead of Pr(X | Y)
        for ycl in data.labels.Y.columns:
            featuresArray = np.add(featuresArray, self.feature_log_prob_y_given_neighbor_x_.loc[ycl,:].values)
        ind = np.argpartition(featuresArray, -k)[-k:]  #Get the indices of the k largest values for prob y given x neighbors
        print("\nPROB Y GIVEN NEIGHBORING FEATURES")
        self.top_k_neighbor_features = data.features.columns.levels[0][data.features.columns.labels[0][ind]]
        print(self.top_k_neighbor_features)  #This gives us the highest valued K features for this output (0 or 1)
        print(data.features.columns.labels[1][ind])  #This gives whether the feature is present or not
			

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

                idx = np.nan_to_num(data.labels.Y[ycl].values * data.mask.Labeled.values).nonzero()[0]

                currentMeanCalc = features[idx,:].mean(axis=0)

                for i in range(0, len(currentMeanCalc)):
                    if currentMeanCalc[i] == 0.0:
                        currentMeanCalc[i] += 0.01
				
                self.feature_log_prob_x_.loc[ycl, :] = np.log(currentMeanCalc)
				
            #print(self.feature_log_prob_x_)
				
            #Here we begin computing Pr( X_neighbors | Y)
			# In general, we are iterating over each output label Y = 0 and Y = 1, and producing a distribution for each one.
			# We iterate through each labelled node and sum the features of its neighbors (currentMeanCalc)
			# If the label of this node matches the current output label, we can add currentMeanCalc to the overall sum
			#    Otherwise, we just ignore it because it does not contribute to the Pr(X_neighbors | Y)
			#    For example, if the current node's label is Y = 1, it does not make sense to add this node's distribution to Pr(X_neighbors | Y = 0)
			# Finally, we divide by the total number of nodes for this output label Y and logify it
            self.feature_log_prob_y_given_neighbor_x_ = pandas.DataFrame(columns=data.features.columns, dtype=np.float64)

            edge_matrix = data.edges.todense()

            #Iterate through each label Y = 0 and Y = 1
            for ycl in data.labels.Y.columns:

                #Initialize the result array for this y-value to all zeroes
                self.feature_log_prob_y_given_neighbor_x_.loc[ycl,:] = np.full_like(features[0,:], 0.0)

                
                total_nodes = 0
                #Iterate through each unlabeled node
                for node in data.mask.Labeled.nonzero()[0]:

                
                    neighbors = edge_matrix[node].nonzero()[1] #Array containing all the neighboring indexes
                    neighbors_idx = np.nan_to_num(data.labels.Y[ycl][neighbors]).nonzero()[0]  #This is all neighbor indexes that have Y = ycl (0 or 1)

                    
                    node_label = data.labels.Y.values[node][ycl]  #This will be 0 or 1.  It is 0 if the label of this node does not match the current label Y
                    total_nodes += node_label
					#Calculate the mean of the features for this set of neighbors
                    currentMeanCalc = features[neighbors_idx,:].mean(axis=0)

                    for i in range(0, len(currentMeanCalc)):
                        if currentMeanCalc[i] == 0.0:
                            currentMeanCalc[i] += 0.01
                    #If the label of this node matches the current label Y, we add the mean of the neighboring features to the global distribution
                    if(node_label > 0.0):
                        self.feature_log_prob_y_given_neighbor_x_.loc[ycl, :] += currentMeanCalc
                    
				# We have to divide the distribution for this Y by the total number of nodes to get an average, and then calculate the log probability distribution
                self.feature_log_prob_y_given_neighbor_x_.loc[ycl,:] = np.log(self.feature_log_prob_y_given_neighbor_x_.loc[ycl,:].divide(total_nodes))
			


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
            #print(neighbor_counts)
            neighbor_counts = neighbor_counts.div(neighbor_counts.sum(axis=1), axis=0)
			
            self.feature_log_prob_y_ = np.log(neighbor_counts)
            #print(self.feature_log_prob_y)
			
        elif self.learn_method == "r_iid2":

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
            #print(neighbor_counts)
            neighbor_counts = neighbor_counts.div(neighbor_counts.sum(axis=1), axis=0)
			
            self.feature_log_prob_y_ = np.log(neighbor_counts)

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
