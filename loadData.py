import csv
import numpy as np
import pandas
from copy import deepcopy


def oneHotEncodeCSV():
	data_file_name = ".\student\student-mat.csv"
	output_file_name = ".\student\student-mat-boolean.csv"
	
	output_f = open(output_file_name, "w")
	
	with open(data_file_name) as f:
			data_file = csv.reader(f, delimiter =';')

			#We are picking and choosing our features.
			#These booleanized features are taken from each of the below attributes
			feature_names = "SCHOOL_IS_GP; SCHOOL_IS_MS; SEX_F; SEX_M; AGE; ADDR_U; ADDR_R; FAM_LE3; FAM_GT3; PSTAT_T; PSTAT_A;" + \
			"MEDU_0; MEDU_1; MEDU_2; MEDU_3; MEDU_4; FEDU_0; FEDU_1; FEDU_2; FEDU_3; FEDU_4; MJOB_0; MJOB_1; MJOB_2;MJOB_3;MJOB_4;" + \
			"FJOB_0; FJOB_1; FJOB_2;FJOB_3;FJOB_4; REASON_HOME;REASON_REP;REASON_COURSE;REASON_OTHER;" + \
			"GUARDIAN_M; GUARDIAN_F; GUARDIAN_O; TRAVEL_T0; TRAVEL_T1; TRAVEL_T2;TRAVEL_T3; STUDYTIME_0;STUDYTIME_1;STUDYTIME_2;STUDYTIME_3;" + \
			"FAILURE_1; FAILURE_2; FAILURE_3; FAILURE_OTHER; SCHOOL_SUPPORT; FAMILY_SUPPORT; EXTRA_CLASSES; EXTRA_ACTIVITIES; ATTENDED_NURSERY;" + \
			"HIGHER_ED; INTERNET; ROMANTIC; FAMILY_RELATION_1; FAMILY_RELATION_2; FAMILY_RELATION_3; FAMILY_RELATION_4; FAMILY_RELATION_5;" + \
			"FREE_TIME_1;FREE_TIME_2;FREE_TIME_3;FREE_TIME_4;FREE_TIME_5; GO_OUT_1;GO_OUT_2;GO_OUT_3;GO_OUT_4;GO_OUT_5;" + \
			"WDAY_ALC_1;WDAY_ALC_2;WDAY_ALC_3;WDAY_ALC_4;WDAY_ALC_5; WEND_ALC_1;WEND_ALC_2;WEND_ALC_3;WEND_ALC_4;WEND_ALC_5;" + \
			"HEALTH_1;HEALTH_2;HEALTH_3;HEALTH_4;HEALTH_5; ABSENCES; G1; G2; G3;";
			output_f.write(feature_names + "\n");
			
			feature_list_cnt = len(feature_names.split(";"))
			print("We have " + str(feature_list_cnt) + " features(including label)")
			next(data_file)
			
			#print(feature_names)
			for i, d in enumerate(data_file):
				
				#print(d)
				lineToWrite = ""
				lineToWrite += return_school(d[0])  #Get School Data
				lineToWrite += return_sex(d[1]) #Get Sex information
				lineToWrite += return_val(d[2]) #get age information
				lineToWrite += return_addr(d[3]) #get addr information
				lineToWrite += return_famsize(d[4]) #get family size information
				lineToWrite += return_pstatus(d[5]) #get parent relationship information
				lineToWrite += return_edu(d[6]) #get education information of mother
				lineToWrite += return_edu(d[7]) #get education information of father
				lineToWrite += return_job(d[8]) #Get job info of mother
				lineToWrite += return_job(d[9]) #Get job info of father
				lineToWrite += return_reason(d[10]) #Get reason for school
				lineToWrite += return_guardian(d[11]) #Get guardian info of student
				lineToWrite += return_time(d[12]) #Get travel time of student
				lineToWrite += return_time(d[13]) #Get study time of student
				lineToWrite += return_time(d[14]) #Get failure count of student
				
				lineToWrite += return_binary(d[15]) #Get school support status
				lineToWrite += return_binary(d[16]) #Get family support status
				lineToWrite += return_binary(d[17]) #Get paid classes status
				lineToWrite += return_binary(d[18]) #Get activities status
				lineToWrite += return_binary(d[19]) #Get nursery status
				lineToWrite += return_binary(d[20]) #Get higher education status
				lineToWrite += return_binary(d[21]) #Get internet status
				lineToWrite += return_binary(d[22]) #Get romance status
				
				lineToWrite += return_5s(d[23]) #Get family rel quality
				lineToWrite += return_5s(d[24]) #Get free time
				lineToWrite += return_5s(d[25]) #Get go out time
				lineToWrite += return_5s(d[26]) #Get weekday alc consumption
				lineToWrite += return_5s(d[27]) #Get weekend alc consumption
				lineToWrite += return_5s(d[28]) #Get health status
				
				lineToWrite += return_val(d[29]) #Get absences
				lineToWrite += return_val(d[30]) #Get G1 Grades
				lineToWrite += return_val(d[31]) #Get G2 grades
				lineToWrite += return_val(d[32]) #Get G3 grades
				
				#Write the data
				output_f.write(lineToWrite + "\n")
				
				lineList_cnt = len(lineToWrite.split(";"))
				
				if lineList_cnt != feature_list_cnt:
					print("Something does not add up!")
					break
				
				#for i in range(0, len(lineList)):
				#	print("Feature: " + feature_list[i] + " val: " + lineList[i])
				
				#print(feature_list)
				#break
	
	output_f.close()


#1 school - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
def return_school(val):
	if val == "GP":
		return "1;0;"
	else:
		return "0;1;"


# 2 sex - student's sex (binary: "F" - female or "M" - male)
#  Return 0 for Female, 1 for Male
def return_sex(val):
	if val == "F":
		return "1;0;"
	else:
		return "0;1;"

# 3 age - student's age (numeric: from 15 to 22)

# 4 address - student's home address type (binary: "U" - urban or "R" - rural)
# Return 0 for Urban, 1 for Rural
def return_addr(val):
	if val == "U":
		return "1;0;"
	else:
		return "0;1;"

# 5 famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
def return_famsize(val):
	if val == "LE3":
		return "1;0;"
	else:
		return "0;1;"

# 6 Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
def return_pstatus(val):
	if val == "T":
		return "1;0;"
	else:
		return "0;1;"

# 7 Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# 8 Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
def return_edu(val):
	val = int(val)
	if val == 0:
		return "1;0;0;0;0;"
	elif val == 1:
		return "0;1;0;0;0;"
	elif val == 2:
		return "0;0;1;0;0;"
	elif val == 3:
		return "0;0;0;1;0;"
	elif val == 4:
		return "0;0;0;0;1;"
	else:
		return "0;0;0;0;0;"



# 9 Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
# 10 Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
def return_job(val):
	if val == "teacher":
		return "1;0;0;0;0;"
	elif val == "health":
		return "0;1;0;0;0;"
	elif val == "services":
		return "0;0;1;0;0;"
	elif val == "at_home":
		return "0;0;0;1;0;"
	elif val == "other":
		return "0;0;0;0;1;"
	else:
		return "0;0;0;0;0;"

# 11 reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
def return_reason(val):
	if val == "home":
		return "1;0;0;0;"
	elif val == "reputation":
		return "0;1;0;0;"
	elif val == "course":
		return "0;0;1;0;"
	elif val == "other":
		return "0;0;0;1;"

# 12 guardian - student's guardian (nominal: "mother", "father" or "other")
def return_guardian(val):
	if val == "mother":
		return "1;0;0;"
	elif val == "father":
		return "0;1;0;"
	elif val == "other":
		return "0;0;1;"

# 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# 15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
def return_time(val):
	val = int(val)
	if val == 1:
		return "1;0;0;0;"
	elif val == 2:
		return "0;1;0;0;"
	elif val == 3:
		return "0;0;1;0;"
	elif val == 4:
		return "0;0;0;1;"
	elif val == 0:
		return "0;0;0;0;"


# 16 schoolsup - extra educational support (binary: yes or no)
# 17 famsup - family educational support (binary: yes or no)
# 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# 19 activities - extra-curricular activities (binary: yes or no)
# 20 nursery - attended nursery school (binary: yes or no)
# 21 higher - wants to take higher education (binary: yes or no)
# 22 internet - Internet access at home (binary: yes or no)
# 23 romantic - with a romantic relationship (binary: yes or no)
# Return 0 for no, 1 for yes
def return_binary(val):
	if val == "yes":
		return "1;"
	else:
		return "0;"

# 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 29 health - current health status (numeric: from 1 - very bad to 5 - very good)
def return_5s(val):
	val = int(val)
	if val == 1:
		return "1;0;0;0;0;"
	elif val == 2:
		return "0;1;0;0;0;"
	elif val == 3:
		return "0;0;1;0;0;"
	elif val == 4:
		return "0;0;0;1;0;"
	elif val == 5:
		return "0;0;0;0;1;"
	else:
		return "0;0;0;0;0;"

# 30 absences - number of school absences (numeric: from 0 to 93)

# these grades are related with the course subject, Math or Portuguese:
# 31 G1 - first period grade (numeric: from 0 to 20)
# 31 G2 - second period grade (numeric: from 0 to 20)
# 32 G3 - final grade (numeric: from 0 to 20, output target)
def return_val(val):
	return str(val) + ";"


def load_data(n_samples, n_features):
	data_file_name = ".\student\student-mat-boolean.csv"

	with open(data_file_name) as f:
			data_file = csv.reader(f, delimiter =';')
			#temp = next(data_file)
			#n_samples = int(temp[0])
			#n_features = int(temp[1])
			data = np.empty((n_samples, n_features))
			target = np.empty((n_samples,))
			temp = next(data_file)  # names of features
			#feature_names = np.array(temp)
			
			#We are picking and choosing our features.
			feature_names = np.array([x.strip() for x in temp])
			#print(feature_names)
			for i, d in enumerate(data_file):
				#print(str(i) + ":  " + str(d[:-1]))
				#print("D size: " + str(len(d)))
				parsed_d = [int(x) for x in d[:-1]]
				# parsed_d.append(return_sex(d[1]))
				# parsed_d.append(return_addr(d[3]))
				# parsed_d.append(d[6])
				# parsed_d.append(d[7])
				# parsed_d.append(d[13])
				# parsed_d.append(return_higherEd(d[20]))
				# parsed_d.append(d[29])
				# print(parsed_d)
				
				data[i] = np.asarray(parsed_d, dtype=np.uint8)
				target[i] = np.asarray(d[-2], dtype=np.uint8)
				
	return data, target, feature_names

	
def booleanizeViaMedian(init_features_old, listOfFeatures):

	init_features = deepcopy(init_features_old)
	for feature in listOfFeatures:
		if feature in init_features:
			#print(init_features[feature])
			median = init_features[feature].median()
			print("Feature: " + feature + " Median: " + str(median))
			init_features[feature] = init_features[feature] - median
			
			init_features[feature][init_features[feature] <= 0] = 0
			init_features[feature][init_features[feature] > 0] = 1
			#init_features[init_features[feature] < 0] = 0
			#init_features[init_features[feature] > 0] = 1
			#print(init_features[feature])
	
	return init_features

from rmllib.data.base import Dataset
from rmllib.data.base import class_transform_to_dataframe
from rmllib.data.generate import matched_edge_generator

#Instead of discretized values, try using present/notpresent
# Test data with custom links - just absences, just father's education and see if it can predict anything.
class AcademicPerformance(Dataset):
	'''
	Simple boston dataset with randomized edge data
	'''
	def __init__(self, subfeatures=None, **kwargs):
		'''
		Builds our dataset by
		(a) loading sklearn Boston dataset
		(b) binarizing it via the median of the feature values
		(c) generating random edges

		:subfeatures: Subsets of features available in the academic performance dataset.  Primarily for simulating weakened feature signals.
		:kwargs: Arguments for matched_edge_generator
		'''
		super().__init__(**kwargs)
		data, target, feature_names = load_data(395, 92)  #We have 395 students, and 92 features (+1 label)
		#print("Features: " + str(feature_names))
		
		init_features = pandas.DataFrame(data, columns=feature_names[:-1])
		#print(feature_names[:-1])

		if subfeatures:
			init_features = init_features[subfeatures]

		#This is binarizing the labels (G3)
		init_labels = pandas.DataFrame(target, columns=['Y'])
		init_labels = init_labels - init_labels.median(axis=0)
		init_labels.Y = np.where(init_labels.Y > 0, 1, 0).astype(int)

		
		#We only need the medians for column AGE,  ABSENCES, G1, G2, G3

		# Booleanize feat by medians
		toBooleanize = ["AGE", "ABSENCES", "G1", "G2", "G3"]
		init_features = booleanizeViaMedian(init_features, toBooleanize)
		#print(init_features)
		
		# init_features = init_features - init_features.median(axis=0)

		#Is there a column that is entirely zeros?
		for col in init_features:
			if np.max(init_features[col]) == 0:
				init_features[col] = 1
				print("ZERO COLUMN " + col)
				#print(init_features[col])
		
		
		#init_features[init_features < 0] = 1
		#init_features[init_features > 0] = 1
		#init_features[:] = 1
		#print(init_features)
		init_features = init_features.astype(int)
		
		#print("Minimum:" + str(np.min(init_features)))
		#print("Maximium:" + str(np.max(init_features)))
		# Create dataframe
		self.labels = class_transform_to_dataframe(init_labels.Y.values, islabel=True)
		self.features = class_transform_to_dataframe(init_features.values, islabel=False, classes=init_features.columns.values)

		# Simple correlation for edges       
		#self.edges = matched_edge_generator(self.labels, **kwargs)
		self.edges = matched_edge_generator(self.labels, mu_match= 0.5, mu_nomatch = 0.5, **kwargs)
		#self.edges = 
		
		return
