from data.scaffold import scaffold_split
from data.data import MoleculeDatapoint, MoleculeDataset
from argparse import ArgumentParser, Namespace


parser = ArgumentParser()

parser.add_argument('--train_data_path', type=str, default='datachem/ZhangDDI_train.csv', choices=['datachem/ZhangDDI_train.csv', 'datachem/ChChMiner_train.csv', 'datachem/DeepDDI_train.csv'])
parser.add_argument('--valid_data_path', type=str, default='datachem/ZhangDDI_valid.csv', choices=['datachem/ZhangDDI_valid.csv', 'datachem/ChChMiner_valid.csv', 'datachem/DeepDDI_valid.csv'])
parser.add_argument('--test_data_path', type=str, default='datachem/ZhangDDI_test.csv', choices=['datachem/ZhangDDI_test.csv', 'datachem/ChChMiner_test.csv', 'datachem/DeepDDI_test.csv'])


args = parser.parse_args()

data_train_path = args.train_data_path
data_valid_path = args.valid_data_path
data_test_path = args.test_data_path


if data_train_path == 'datachem/DeepDDI_train.csv':

	druglist = set()
	with open('datachem/drug_list_deep.csv') as f:
		firstline = f.readline()
		while True:
			line = f.readline().strip()
			if not line:
				break
			drug_smile = line.split(',')[1]
			druglist.add(drug_smile)



	interaction_relation = {}
	smiles_set = set()


	with open(data_train_path, 'r') as f:
	    tile_line = f.readline()
	    while True:
	        line = f.readline().strip()
	        if not line:
	            break
	            
	        line_list = line.split(',')
	        
	        smile1 = line_list[0]
	        smile2 = line_list[1]
	        inter_relation = line_list[2]

	        smiles_set.add(smile1)
	        smiles_set.add(smile2)
	        if interaction_relation.get(smile1) is None:
	            interaction_relation[smile1] = {}
	        interaction_relation[smile1][smile2] = inter_relation
	        
	        if interaction_relation.get(smile2) is None:
	            interaction_relation[smile2] = {}
	        interaction_relation[smile2][smile1] = inter_relation
	        
	with open(data_valid_path, 'r') as f:
	    tile_line = f.readline()
	    while True:
	        line = f.readline().strip()
	        if not line:
	            break
	            
	        line_list = line.split(',')
	        smile1 = line_list[0]
	        smile2 = line_list[1]
	        inter_relation = line_list[2]

	        smiles_set.add(smile1)
	        smiles_set.add(smile2)
	        if interaction_relation.get(smile1) is None:
	            interaction_relation[smile1] = {}
	        interaction_relation[smile1][smile2] = inter_relation
	        
	        if interaction_relation.get(smile2) is None:
	            interaction_relation[smile2] = {}
	        interaction_relation[smile2][smile1] = inter_relation

	with open(data_test_path, 'r') as f:
	    tile_line = f.readline()
	    while True:
	        line = f.readline().strip()
	        if not line:
	            break
	            
	        line_list = line.split(',')
	        
	        smile1 = line_list[0]
	        smile2 = line_list[1]
	        inter_relation = line_list[2]

	        smiles_set.add(smile1)
	        smiles_set.add(smile2)
	        if interaction_relation.get(smile1) is None:
	            interaction_relation[smile1] = {}
	        interaction_relation[smile1][smile2] = inter_relation
	        
	        if interaction_relation.get(smile2) is None:
	            interaction_relation[smile2] = {}
	        interaction_relation[smile2][smile1] = inter_relation
	        

	smiles_list = list(smiles_set)
	smiles_list = [MoleculeDatapoint(line=[s]) for s in smiles_list]

	dataset = MoleculeDataset(smiles_list)
	train, val, test = scaffold_split(dataset, sizes=(0.8, 0.1, 0.1))

	train_smiles = set(train.smiles())
	val_smiles = set(val.smiles())
	test_smiles = set(test.smiles())

	generated_result = set()

	train_output_path = data_train_path.split('.')[0] + '_scaffold.csv'
	val_output_path = data_valid_path.split('.')[0] + '_scaffold.csv'
	test_output_path = data_test_path.split('.')[0] + '_scaffold.csv'


	print(train_output_path)
	with open(train_output_path, 'w') as f:
		f.write('smiles_1,smiles_2,label\n')
		for train_smile in train_smiles:
			if interaction_relation.get(train_smile) is not None:
				dict_tmp = interaction_relation.get(train_smile)
				for key, value in dict_tmp.items():
					if ((train_smile, key) not in generated_result) and ((key, train_smile) not in generated_result) and (key in druglist) and (train_smile in druglist):
						generated_result.add((train_smile, key))
						relation = value
						line_list = [train_smile, key, relation]
						line = ','.join(line_list) + '\n'
						f.write(line)

	with open(val_output_path, 'w') as f:
		f.write('smiles_1,smiles_2,label\n')
		for val_smile in val_smiles:
			if interaction_relation.get(val_smile) is not None:
				dict_tmp = interaction_relation.get(val_smile)
				for key, value in dict_tmp.items():
					if key not in train_smiles:
						if ((val_smile, key) not in generated_result) and ((key, val_smile) not in generated_result):
							generated_result.add((val_smile, key))
							relation = value
							line_list = [train_smile, key, relation]
							line = ','.join(line_list) + '\n'
							f.write(line)


	with open(test_output_path, 'w') as f:
		f.write('smiles_1,smiles_2,label\n')
		for test_smile in test_smiles:
			if interaction_relation.get(test_smile) is not None:
				dict_tmp = interaction_relation.get(test_smile)
				for key, value in dict_tmp.items():
					if key in test_smiles:
						if ((test_smile, key) not in generated_result) and ((key, test_smile) not in generated_result):
							generated_result.add((test_smile, key))
							relation = value
							line_list = [train_smile, key, relation]
							line = ','.join(line_list) + '\n'
							f.write(line)


if data_train_path == 'datachem/ChChMiner_train.csv':

	drug2smile = {}
	smile2drug = {}
	interaction_relation = {}
	smiles_set = set()


	with open(data_train_path, 'r') as f:
	    tile_line = f.readline()
	    while True:
	        line = f.readline().strip()
	        if not line:
	            break
	            
	        line_list = line.split(',')
	        
	        drug1 = line_list[0]
	        drug2 = line_list[1]
	        smile1 = line_list[2]
	        smile2 = line_list[3]
	        inter_relation = line_list[4]
	        
	        drug2smile[drug1] = smile1
	        drug2smile[drug2] = smile2
	        smile2drug[smile1] = drug1
	        smile2drug[smile2] = drug2

	        smiles_set.add(smile1)
	        smiles_set.add(smile2)
	        if interaction_relation.get(smile1) is None:
	            interaction_relation[smile1] = {}
	        interaction_relation[smile1][smile2] = inter_relation
	        
	        if interaction_relation.get(smile2) is None:
	            interaction_relation[smile2] = {}
	        interaction_relation[smile2][smile1] = inter_relation
	        
	with open(data_valid_path, 'r') as f:
	    tile_line = f.readline()
	    while True:
	        line = f.readline().strip()
	        if not line:
	            break
	            
	        line_list = line.split(',')
	        
	        drug1 = line_list[0]
	        drug2 = line_list[1]
	        smile1 = line_list[2]
	        smile2 = line_list[3]
	        inter_relation = line_list[4]
	        
	        drug2smile[drug1] = smile1
	        drug2smile[drug2] = smile2
	        smile2drug[smile1] = drug1
	        smile2drug[smile2] = drug2

	        smiles_set.add(smile1)
	        smiles_set.add(smile2)
	        if interaction_relation.get(smile1) is None:
	            interaction_relation[smile1] = {}
	        interaction_relation[smile1][smile2] = inter_relation
	        
	        if interaction_relation.get(smile2) is None:
	            interaction_relation[smile2] = {}
	        interaction_relation[smile2][smile1] = inter_relation

	with open(data_test_path, 'r') as f:
	    tile_line = f.readline()
	    while True:
	        line = f.readline().strip()
	        if not line:
	            break
	            
	        line_list = line.split(',')
	        
	        drug1 = line_list[0]
	        drug2 = line_list[1]
	        smile1 = line_list[2]
	        smile2 = line_list[3]
	        inter_relation = line_list[4]
	        
	        drug2smile[drug1] = smile1
	        drug2smile[drug2] = smile2
	        smile2drug[smile1] = drug1
	        smile2drug[smile2] = drug2

	        smiles_set.add(smile1)
	        smiles_set.add(smile2)
	        if interaction_relation.get(smile1) is None:
	            interaction_relation[smile1] = {}
	        interaction_relation[smile1][smile2] = inter_relation
	        
	        if interaction_relation.get(smile2) is None:
	            interaction_relation[smile2] = {}
	        interaction_relation[smile2][smile1] = inter_relation
	        

	smiles_list = list(smiles_set)
	smiles_list = [MoleculeDatapoint(line=[s]) for s in smiles_list]

	dataset = MoleculeDataset(smiles_list)
	train, val, test = scaffold_split(dataset, sizes=(0.8, 0.1, 0.1))

	train_smiles = set(train.smiles())
	val_smiles = set(val.smiles())
	test_smiles = set(test.smiles())

	generated_result = set()

	train_output_path = data_train_path.split('.')[0] + '_scaffold.csv'
	val_output_path = data_valid_path.split('.')[0] + '_scaffold.csv'
	test_output_path = data_test_path.split('.')[0] + '_scaffold.csv'


	print(train_output_path)
	with open(train_output_path, 'w') as f:
		f.write('drugbank_id_1,drugbank_id_2,smiles_1,smiles_2,label\n')
		for train_smile in train_smiles:
			if interaction_relation.get(train_smile) is not None:
				dict_tmp = interaction_relation.get(train_smile)
				for key, value in dict_tmp.items():
					if ((train_smile, key) not in generated_result) and ((key, train_smile) not in generated_result):
						generated_result.add((train_smile, key))
						drug1 = smile2drug[train_smile]
						drug2 = smile2drug[key]

						relation = value
						line_list = [drug1, drug2, train_smile, key, relation]
						line = ','.join(line_list) + '\n'
						f.write(line)

	with open(val_output_path, 'w') as f:
		f.write('drugbank_id_1,drugbank_id_2,smiles_1,smiles_2,label\n')
		for val_smile in val_smiles:
			if interaction_relation.get(val_smile) is not None:
				dict_tmp = interaction_relation.get(val_smile)
				for key, value in dict_tmp.items():
					if key not in train_smiles:
						if ((val_smile, key) not in generated_result) and ((key, val_smile) not in generated_result):
							generated_result.add((val_smile, key))
							drug1 = smile2drug[val_smile]
							drug2 = smile2drug[key]
							relation = value
							line_list = [drug1, drug2, train_smile, key, relation]
							line = ','.join(line_list) + '\n'
							f.write(line)


	with open(test_output_path, 'w') as f:
		f.write('drugbank_id_1,drugbank_id_2,smiles_1,smiles_2,label\n')
		for test_smile in test_smiles:
			if interaction_relation.get(test_smile) is not None:
				dict_tmp = interaction_relation.get(test_smile)
				for key, value in dict_tmp.items():
					if key in test_smiles:
						if ((test_smile, key) not in generated_result) and ((key, test_smile) not in generated_result):
							generated_result.add((test_smile, key))
							drug1 = smile2drug[test_smile]
							drug2 = smile2drug[key]
							relation = value
							line_list = [drug1, drug2, train_smile, key, relation]
							line = ','.join(line_list) + '\n'
							f.write(line)


if data_train_path == 'datachem/ZhangDDI_train.csv':
	drug2smile = {}
	smile2drug = {}
	drug2cid = {}
	interaction_relation = {}
	smiles_set = set()



	with open(data_train_path, 'r') as f:
	    tile_line = f.readline()
	    while True:
	        line = f.readline().strip()
	        if not line:
	            break
	            
	        line_list = line.split(',')
	        
	        drug1 = line_list[0]
	        drug2 = line_list[1]
	        smile1 = line_list[3]
	        smile2 = line_list[2]
	        cid1 = line_list[4]
	        cid2 = line_list[5]
	        inter_relation = line_list[6]
	        
	        drug2smile[drug1] = smile1
	        drug2smile[drug2] = smile2
	        smile2drug[smile1] = drug1
	        smile2drug[smile2] = drug2
	        drug2cid[drug1] = cid1
	        drug2cid[drug2] = cid2
	        smiles_set.add(smile1)
	        smiles_set.add(smile2)
	        if interaction_relation.get(smile1) is None:
	            interaction_relation[smile1] = {}
	        interaction_relation[smile1][smile2] = inter_relation
	        
	        if interaction_relation.get(smile2) is None:
	            interaction_relation[smile2] = {}
	        interaction_relation[smile2][smile1] = inter_relation
	        
	with open(data_valid_path, 'r') as f:
	    tile_line = f.readline()
	    while True:
	        line = f.readline().strip()
	        if not line:
	            break
	            
	        line_list = line.split(',')
	        
	        drug1 = line_list[0]
	        drug2 = line_list[1]
	        smile1 = line_list[3]
	        smile2 = line_list[2]
	        cid1 = line_list[4]
	        cid2 = line_list[5]
	        inter_relation = line_list[6]
	        
	        drug2smile[drug1] = smile1
	        drug2smile[drug2] = smile2
	        smile2drug[smile1] = drug1
	        smile2drug[smile2] = drug2
	        drug2cid[drug1] = cid1
	        drug2cid[drug2] = cid2
	        smiles_set.add(smile1)
	        smiles_set.add(smile2)
	        if interaction_relation.get(smile1) is None:
	            interaction_relation[smile1] = {}
	        interaction_relation[smile1][smile2] = inter_relation
	        
	        if interaction_relation.get(smile2) is None:
	            interaction_relation[smile2] = {}
	        interaction_relation[smile2][smile1] = inter_relation

	with open(data_test_path, 'r') as f:
	    tile_line = f.readline()
	    while True:
	        line = f.readline().strip()
	        if not line:
	            break
	            
	        line_list = line.split(',')
	        
	        drug1 = line_list[0]
	        drug2 = line_list[1]
	        smile1 = line_list[3]
	        smile2 = line_list[2]
	        cid1 = line_list[4]
	        cid2 = line_list[5]
	        inter_relation = line_list[6]
	        
	        drug2smile[drug1] = smile1
	        drug2smile[drug2] = smile2
	        smile2drug[smile1] = drug1
	        smile2drug[smile2] = drug2
	        drug2cid[drug1] = cid1
	        drug2cid[drug2] = cid2
	        smiles_set.add(smile1)
	        smiles_set.add(smile2)
	        if interaction_relation.get(smile1) is None:
	            interaction_relation[smile1] = {}
	        interaction_relation[smile1][smile2] = inter_relation
	        
	        if interaction_relation.get(smile2) is None:
	            interaction_relation[smile2] = {}
	        interaction_relation[smile2][smile1] = inter_relation
	        

	smiles_list = list(smiles_set)
	smiles_list = [MoleculeDatapoint(line=[s]) for s in smiles_list]

	dataset = MoleculeDataset(smiles_list)
	train, val, test = scaffold_split(dataset, sizes=(0.8, 0.1, 0.1))

	train_smiles = set(train.smiles())
	val_smiles = set(val.smiles())
	test_smiles = set(test.smiles())

	generated_result = set()

	train_output_path = data_train_path.split('.')[0] + '_scaffold.csv'
	val_output_path = data_valid_path.split('.')[0] + '_scaffold.csv'
	test_output_path = data_test_path.split('.')[0] + '_scaffold.csv'


	print(train_output_path)
	with open(train_output_path, 'w') as f:
		f.write('drugbank_id_1,drugbank_id_2,smiles_2,smiles_1,cid_1,cid_2,label\n')
		for train_smile in train_smiles:
			if interaction_relation.get(train_smile) is not None:
				dict_tmp = interaction_relation.get(train_smile)
				for key, value in dict_tmp.items():
					if ((train_smile, key) not in generated_result) and ((key, train_smile) not in generated_result):
						generated_result.add((train_smile, key))
						drug1 = smile2drug[train_smile]
						drug2 = smile2drug[key]
						cid1 = drug2cid[drug1]
						cid2 = drug2cid[drug2]
						relation = value
						line_list = [drug1, drug2, key, train_smile, cid1, cid2, relation]
						line = ','.join(line_list) + '\n'
						f.write(line)

	with open(val_output_path, 'w') as f:
		f.write('drugbank_id_1,drugbank_id_2,smiles_2,smiles_1,cid_1,cid_2,label\n')
		for val_smile in val_smiles:
			if interaction_relation.get(val_smile) is not None:
				dict_tmp = interaction_relation.get(val_smile)
				for key, value in dict_tmp.items():
					if key not in train_smiles:
						if ((val_smile, key) not in generated_result) and ((key, val_smile) not in generated_result):
							generated_result.add((val_smile, key))
							drug1 = smile2drug[val_smile]
							drug2 = smile2drug[key]
							cid1 = drug2cid[drug1]
							cid2 = drug2cid[drug2]
							relation = value
							line_list = [drug1, drug2, key, val_smile, cid1, cid2, relation]
							line = ','.join(line_list) + '\n'
							f.write(line)


	with open(test_output_path, 'w') as f:
		f.write('drugbank_id_1,drugbank_id_2,smiles_2,smiles_1,cid_1,cid_2,label\n')
		for test_smile in test_smiles:
			if interaction_relation.get(test_smile) is not None:
				dict_tmp = interaction_relation.get(test_smile)
				for key, value in dict_tmp.items():
					if key in test_smiles:
						if ((test_smile, key) not in generated_result) and ((key, test_smile) not in generated_result):
							generated_result.add((test_smile, key))
							drug1 = smile2drug[test_smile]
							drug2 = smile2drug[key]
							cid1 = drug2cid[drug1]
							cid2 = drug2cid[drug2]
							relation = value
							line_list = [drug1, drug2, key, test_smile, cid1, cid2, relation]
							line = ','.join(line_list) + '\n'
							f.write(line)


print (len(generated_result))