import pandas as pd
import statistics as stats

def read_data(filepath):
  df = pd.read_csv(filepath, header=None)
  # print(df)

  healthy_temp = []
  healthy_heart_rate = []
  diseased_temp = []
  diseased_heart_rate = []

  for _, row in df.iterrows():
    if row[0] == "healthy":
      healthy_temp.append(row[1])
      healthy_heart_rate.append(row[2])
    else:
      diseased_temp.append(row[1])
      diseased_heart_rate.append(row[2])

  return healthy_temp, healthy_heart_rate, diseased_temp, diseased_heart_rate

def get_stats(curr_list):
  return stats.mean(curr_list), stats.stdev(curr_list)



def naive_bayes_classifier(dataset_filepath, patient_measurements):
  # dataset_filepath is the full file path to a CSV file containing the dataset
  # patient_measurements is a list of [temperature, heart rate] measurements for a patient
  healthy_temp, healthy_heart_rate, diseased_temp, diseased_heart_rate = read_data(dataset_filepath)

  healthy_temp_mean, healthy_temp_stddev = get_stats(healthy_temp)
  healthy_hr_mean, healthy_hr_stddev = get_stats(healthy_heart_rate)
  diseased_temp_mean, diseased_temp_stddev = get_stats(diseased_temp)
  diseased_hr_mean, diseased_hr_stddev = get_stats(diseased_heart_rate)

  #P(diseased| hr=)
  
  most_likely_class = "none"
  class_probabilities = [0,0]

  # most_likely_class is a string indicating the most likely class, either "healthy", "diseased"
  # class_probabilities is a two element list indicating the probability of each class in the order [healthy probability, diseased probability]
  return most_likely_class, class_probabilities

naive_bayes_classifier('Examples\\Example0\\dataset.csv', 0)