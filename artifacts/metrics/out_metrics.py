import json
import pandas as pd
import numpy as np
import os

from tensorflow.python.lib.io import file_io

confmat = np.array([[53,23,15], [124,51,26], [2934,540,2634]])
vocab = ['Class A', 'Class B', 'Class C']
# vocab = np.arange(3)

cdata = []
for target_index, target_row in enumerate(confmat):
	print(target_index, target_row)
	print("\n")
	for pred_idx, count in enumerate(target_row):
		cdata.append((vocab[target_index], vocab[pred_idx], count))
	
df_cm = pd.DataFrame(cdata, columns=['target', 'predicted', 'count'])
cm_file = os.path.join('gs://data-folder/kubeflow_data_trial1', 'confusion_matrix.csv')

with file_io.FileIO(cm_file, 'w') as fl:
	df_cm.to_csv(fl, columns=['target', 'predicted', 'count'], header=False, index=False)

# metadata = {
# 	'outputs' : [
	
# 	]
# }
# with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as fl:
# 	json.dump(metadata, fl)

metrics = {
	'metrics': [
		{
			'name': 'accuracy-score', # The name of the metric. Visualized as the column name in the runs table.
			'numberValue':  str(0.859667), # The value of the metric. Must be a numeric value.
			'format': "PERCENTAGE",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
		},
		{
			'name': 'train-loss',
			'numberValue':  str(0.321564),
			'format': "RAW", 
		},
	]
}

with open('/mlpipeline-metrics.json', 'w') as fl:
	json.dump(metrics, fl)

with open('/mlpipeline-metrics.json', 'r') as fl:
	read_metrics = json.load(fl)
	print(read_metrics)


# output table 
def metrics(cm):
	cm = np.array(cm, dtype = float)
	tot = cm.sum(axis = -1)
	res = {}
	res['acc']= cm.diagonal()/ tot

	pred_tot = cm.sum(axis = 0)
	res['prec'] = cm.diagonal()/pred_tot

	act_tot = cm.sum(axis = 1)
	res['rec']= cm.diagonal()/act_tot


	nc = len(cm)
	tn = np.array([np.delete(np.delete(cm,i,0),i,1).sum() for i in range(nc)])
	cn = np.array([np.delete(cm,i,0).sum() for i in range(nc)])

	res['spec'] = tn/cn

	pe = (1. * (cm.sum(axis = 0) * cm.sum(axis = 1))/ (tot.sum()**2)).sum()
	po = cm.diagonal().sum()/tot.sum()
	res['ck'] =(po-pe)/(1-pe)
	res['overall_acc'] = cm.diagonal().sum() / cm.sum()

	return res

out_met = metrics(confmat)

mdata = []
for i in range(3):
	mdata.append([vocab[i], out_met['acc'][i], out_met['prec'][i], out_met['rec'][i], out_met['spec'][i]])

met_df = pd.DataFrame(mdata, columns=['Class', 'Accuracy', 'Precision', 'Recall', 'Specificity'])
mt_file = os.path.join('gs://data-folder/kubeflow_data_trial1', 'metrics_pr.csv')

with file_io.FileIO(mt_file, 'w') as fl:
	met_df.to_csv(fl, columns=['Class', 'Accuracy', 'Precision', 'Recall', 'Specificity'], header=False, index=False)

metadata = {
	'outputs' : [{
		'type': 'table',
		'storage': 'gcs',
		'format': 'csv',
		'header': ['Class', 'Accuracy', 'Precision', 'Recall', 'Specificity'],
		'source': mt_file
	},
	{
		'type': 'confusion_matrix',
		'storage': 'gcs',
		'format': 'csv',
		'schema': [
			{'name': 'target', 'type': 'CATEGORY'},
			{'name': 'predicted', 'type': 'CATEGORY'},
			{'name': 'count', 'type': 'NUMBER'},
		],
		'source': cm_file,
		# Convert vocab to string because for bealean values we want "True|False" to match csv data.
		'labels': list(map(str, vocab)),
	}]
}
with open('/mlpipeline-ui-metadata.json', 'w') as f:
		json.dump(metadata, f)


with file_io.FileIO('/mlpipeline-ui-metadata.json', 'r') as fl:
	read_metrics = json.load(fl)
	print("metadata")
	print(read_metrics)