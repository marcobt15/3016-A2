import pandas as pd
import statistics as stats #from standard library
import math #from standard library

def read_data(filepath):
  df = pd.read_csv(filepath, header=None)

  healthy_temp = []
  healthy_heart_rate = []
  diseased_temp = []
  diseased_heart_rate = []
  healthy_count = 0

  for _, row in df.iterrows():
    if row[0] == "healthy":
      healthy_temp.append(row[1])
      healthy_heart_rate.append(row[2])
      healthy_count +=1

    else:
      diseased_temp.append(row[1])
      diseased_heart_rate.append(row[2])

  total_patients = df.shape[0]

  diseased_count = total_patients - healthy_count

  return healthy_temp, healthy_heart_rate, diseased_temp, diseased_heart_rate, healthy_count, diseased_count, total_patients

def get_stats(curr_list):
  return stats.mean(curr_list), stats.stdev(curr_list)

def probability_of_feature_given_class(mean, stddev, x):
  term1 = 1/math.sqrt(2*math.pi*(stddev**2))
  term2 = -1/2 * (((x-mean)/stddev)**2)
  term3 = math.e**term2

  return term1 * term3

def naive_bayes_classifier(dataset_filepath, patient_measurements):
  # dataset_filepath is the full file path to a CSV file containing the dataset
  # patient_measurements is a list of [temperature, heart rate] measurements for a patient
  healthy_temp, healthy_heart_rate, diseased_temp, diseased_heart_rate, healthy_count, diseased_count, total_patients = read_data(dataset_filepath)

  #get all means and standard deviations for every feature and class
  healthy_temp_mean, healthy_temp_stddev = get_stats(healthy_temp)
  healthy_hr_mean, healthy_hr_stddev = get_stats(healthy_heart_rate)
  diseased_temp_mean, diseased_temp_stddev = get_stats(diseased_temp)
  diseased_hr_mean, diseased_hr_stddev = get_stats(diseased_heart_rate)

  patient_temp = patient_measurements[0]
  patient_hr = patient_measurements[1]

  #P(temp|healthy)
  p_of_temp_given_healthy = probability_of_feature_given_class(healthy_temp_mean, healthy_temp_stddev, patient_temp)
  #P(hr|healthy)
  p_of_hr_given_healthy = probability_of_feature_given_class(healthy_hr_mean, healthy_hr_stddev, patient_hr)
  #P(temp|diseased)
  p_of_temp_given_diseased = probability_of_feature_given_class(diseased_temp_mean, diseased_temp_stddev, patient_temp)
  #P(hr|diseased)
  p_of_hr_given_diseased = probability_of_feature_given_class(diseased_hr_mean, diseased_hr_stddev, patient_hr)

  # P(healthy)
  p_of_healthy = healthy_count / total_patients

  # P(diseased)
  p_of_diseased = diseased_count / total_patients

  # P(hr = x | healthy) * P(temp = y | healthy) * P(healthy)
  p_of_hr_and_temp_healthy = p_of_hr_given_healthy * p_of_temp_given_healthy * p_of_healthy

  # P(hr = x | diseased) * P(temp = y | diseased) * P(diseased)
  p_of_hr_and_temp_diseased = p_of_hr_given_diseased * p_of_temp_given_diseased * p_of_diseased

  # P(hr = x , temp = y) = P(hr = x | healthy) * P(temp = y | healthy) * P(healthy) + P(hr = x | diseased) * P(temp = y | diseased) * P(diseased)
  p_of_hr_and_temp = p_of_hr_and_temp_healthy + p_of_hr_and_temp_diseased

  # P(diseased | hr = x , temp = y) = P(hr = x | diseased) * P(temp = y | diseased) * P(diseased) / P(hr = x , temp = y)
  p_of_diseased_given_hr_and_temp = p_of_hr_and_temp_diseased / p_of_hr_and_temp

  # P(healthy | hr = x, temp = y) = P(hr = x | healthy) * P(temp = y | healthy) * P(healthy) / P(hr = x , temp = y)
  p_of_healthy_given_hr_and_temp = p_of_hr_and_temp_healthy / p_of_hr_and_temp

  if p_of_diseased_given_hr_and_temp > p_of_healthy_given_hr_and_temp:
    most_likely_class = "diseased"
  else:
    most_likely_class = "healthy"
  
  class_probabilities = [p_of_healthy_given_hr_and_temp, p_of_diseased_given_hr_and_temp]

  # most_likely_class is a string indicating the most likely class, either "healthy", "diseased"
  # class_probabilities is a two element list indicating the probability of each class in the order [healthy probability, diseased probability]
  return most_likely_class, class_probabilities