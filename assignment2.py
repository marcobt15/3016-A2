import pandas as pd
def read_data(filepath):
  df = pd.read_csv(filepath, header=None)
  print(df)

  healthy_set = set()
  diseased_set = set()

  for row in df:
    if row[0] == "healthy":
      healthy_set.add(row[1:3])
    else:
      diseased_set.add(row[1:3])

  return healthy_set, diseased_set

def naive_bayes_classifier(dataset_filepath, patient_measurements):
  # dataset_filepath is the full file path to a CSV file containing the dataset
  # patient_measurements is a list of [temperature, heart rate] measurements for a patient
  healthy_set, deseased_set = read_data(dataset_filepath)
  
  
  most_likely_class = "none"
  class_probabilities = [0,0]

  # most_likely_class is a string indicating the most likely class, either "healthy", "diseased"
  # class_probabilities is a two element list indicating the probability of each class in the order [healthy probability, diseased probability]
  return most_likely_class, class_probabilities

naive_bayes_classifier('Examples\\Example0\\dataset.csv', 0)