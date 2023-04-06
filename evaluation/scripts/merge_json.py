import json

first_file_path = "evaluation/studies/study_0/results/datasets/stats_ED-MPC.json"
second_file_path = "evaluation/studies/study_2/merged/stats_ED-MPC.json"
merged_file_path = "evaluation/studies/study_0//results/merged/stats_ED-MPC.json"

with open(first_file_path) as file:
     first_file = json.load(file)

with open(second_file_path) as file:
     second_file = json.load(file)

for scene in first_file.keys(): # ключи - сцены
    for controller in first_file[scene].keys(): # ключи - контроллеры
         first_file[scene][controller].update(second_file[scene][controller])

with open(merged_file_path, 'w') as outfile:
            json.dump(first_file, outfile, indent=4)
     