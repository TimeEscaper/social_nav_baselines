import json

first_file_path = "evaluation/studies/study_1/results_3_6/datasets/stats_MD-MPC-EDC_W2000.json"
second_file_path = "evaluation/studies/study_1/results_7_8/datasets/stats_MD-MPC-EDC_W2000.json"
merged_file_path = "evaluation/studies/study_1/results/datasets/stats_MD-MPC-EDC_W2000.json"

with open(first_file_path) as file:
     first_file = json.load(file)

with open(second_file_path) as file:
     second_file = json.load(file)

for scene in first_file.keys(): # ключи - сцены
    for controller in first_file[scene].keys(): # ключи - контроллеры
         first_file[scene][controller].update(second_file[scene][controller])

with open(merged_file_path, 'w') as outfile:
            json.dump(first_file, outfile, indent=4)
     