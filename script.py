import numpy as np
import os
import json
import utils.feature_extractors as utils
import optuna
import joblib
import time
import sys

from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from utils.utils import evaluate_classification
from sklearn.svm import SVC

config_map = {}
if len(sys.argv) >= 5:
    train_positive_location = sys.argv[1]
    train_negative_location = sys.argv[2]
    test_positive_location = sys.argv[3]
    test_negative_location = sys.argv[4]

    config_map["train_positive_location"] = train_positive_location
    config_map["train_negative_location"] = train_negative_location
    config_map["test_positive_location"] = test_positive_location
    config_map["test_negative_location"] = test_negative_location

else:
    raise ValueError("Usage: your_script_name <positive training data> <negative training data> <positive testing data> <negative testing data>")

config_map["feat_combo_model_save_location"] = "./script_output/feature_combo_models"
config_map["final_model_save_location"] = "./script_output/final_model"
config_map ["random_seed"] = 9
config_map ["optuna_trials"] = 25


class ProteinFeatureGenerator:
    SELECTED_FEATURES = ["AAC", "DPC", "RScharge", "RSDHP", "RSpolar"]
    
    def __init__(self, positive_data_file: str, negative_data_file: str, feature_type: str = None) -> None:
        super().__init__()

        # Check feature param
        assert feature_type in ProteinFeatureGenerator.SELECTED_FEATURES or feature_type is None
        self.feature_type = feature_type

        # Data manipulation
        self.positive_data_file = positive_data_file
        self.negative_data_file = negative_data_file

        self.positive_data = utils.read_fasta(self.positive_data_file)
        self.negative_data = utils.read_fasta(self.negative_data_file)
        self.data = self.positive_data + self.negative_data

        self.targets = np.array([True]*len(self.positive_data) + [False]*len(self.negative_data))
        

        self.raw_sequences = [x[1] for x in self.data]
        
        
        self.AAC_feature = utils.AAC(self.data)[0]

        self.DPC_feature = utils.DPC(self.data, 0)[0]

        self.RScharge_feature = utils.reducedCHARGE(self.data)
        
        self.RSDHP_feature = utils.reducedDHP(self.data)
        
        self.RSpolar_feature = utils.reducedPOLAR(self.data)

    def get_feat_combo(self,selected:list = None):
        
        features =[self.AAC_feature,self.DPC_feature,self.RScharge_feature,self.RSDHP_feature,self.RSpolar_feature]
        
        if selected:
            select_index = sorted([ProteinFeatureGenerator.SELECTED_FEATURES.index(x) for x in selected])
            features = [features[x] for x in select_index]
            
        return np.concatenate(features,axis=-1)
        
    
    def __len__(self) -> int:
        return len(self.data)

print("Selecting Best Feature Combination..............")  
train_data = ProteinFeatureGenerator(positive_data_file=config_map["train_positive_location"],negative_data_file=config_map["train_negative_location"])

X_data = {
    "AAC-DPC-RScharge-RSDHP-RSpolar":train_data.get_feat_combo(["AAC","DPC","RScharge","RSDHP","RSpolar"]),
    "AAC-RScharge-RSDHP-RSpolar":train_data.get_feat_combo(["AAC","RScharge","RSDHP","RSpolar"]),
    "DPC-RScharge-RSDHP-RSpolar":train_data.get_feat_combo(["DPC","RScharge","RSDHP","RSpolar"]),
    "AAC-DPC-RScharge":train_data.get_feat_combo(["AAC","DPC","RScharge"]),
    "AAC-DPC-RSDHP":train_data.get_feat_combo(["AAC","DPC","RSDHP"]),
    "AAC-RSDHP-RSpolar":train_data.get_feat_combo(["AAC","RSDHP","RSpolar"]),
    "RScharge-RSDHP-RSpolar":train_data.get_feat_combo(["RScharge","RSDHP","RSpolar"]),
    "AAC-RSpolar":train_data.get_feat_combo(["AAC","RSpolar"]),
}


data_pipelines = {
    "AAC-DPC-RScharge-RSDHP-RSpolar":make_pipeline(StandardScaler()),
    "AAC-RScharge-RSDHP-RSpolar":make_pipeline(StandardScaler()),
    "DPC-RScharge-RSDHP-RSpolar":make_pipeline(StandardScaler()),
    "AAC-DPC-RScharge":make_pipeline(StandardScaler()),
    "AAC-DPC-RSDHP":make_pipeline(StandardScaler()),
    "AAC-RSDHP-RSpolar":make_pipeline(StandardScaler()),
    "RScharge-RSDHP-RSpolar":make_pipeline(StandardScaler()),
    "AAC-RSpolar":make_pipeline(StandardScaler()),
}

best_feature_combo = ""
best_acc = 0
for feature_type in X_data.keys():
    
    os.makedirs(os.path.join(config_map["feat_combo_model_save_location"],feature_type,"SVC"),exist_ok=True)
    model_dir = os.path.join(config_map["feat_combo_model_save_location"],feature_type,"SVC")
    
    X,y = X_data[feature_type],train_data.targets
    X,y = shuffle(X,y,random_state=config_map["random_seed"])
    X = data_pipelines[feature_type].fit_transform(X,y)
    
    clf = SVC()
    y_pred = cross_val_predict(clf, X, y, cv=5)
    
    result_values = evaluate_classification(y_pred,y,class_names=["Not Druggable","Druggable"],save_outputs=model_dir)
    
    clf.fit(X,y)
    
    # Select best combo to optimize
    if float(result_values["accuracy"])>best_acc:
        best_acc = result_values["accuracy"]
        best_feature_combo = feature_type
    joblib.dump(data_pipelines[feature_type], os.path.join(model_dir,"pipeline.sav"))
    joblib.dump(clf, os.path.join(model_dir,"model.sav"))

print("Selected Classifier: SVC")    
print("Best Feature Combo:",best_feature_combo)
print("Best Accuracy:",str(best_acc))

print("Optimizing Hyperparameters.......")
# Get the clf and best feature combo
model = "SVC"
best_feature_combo = best_feature_combo
    
    
# Shuffle the data and apply the data pipeline
X, y = shuffle(X_data[best_feature_combo], train_data.targets, random_state=config_map["random_seed"])
data_pipeline = data_pipelines[best_feature_combo]
X = data_pipeline.fit_transform(X, y)
    
# Define the objective function for the Optuna optimization
def obj_func_svc(trial: optuna.trial) -> SVC:
    c = trial.suggest_float('C', 1e-5, 1e2, log=True)
    kernel = 'rbf'
    gamma = 'auto'
        
    classifier = SVC(
        C=c, 
        kernel=kernel,
        class_weight={1: 0.482, 0: 0.518},
        gamma=gamma,
    )
        
    return classifier
def objective(trial):
    clf = obj_func_svc(trial)
    y_pred = cross_val_predict(clf, X, y, cv=5)
    result_values = evaluate_classification(
        y_pred, y, class_names=["Not Druggable", "Druggable"],
        save_outputs=None
    )
    return result_values["f1"]


# Create an Optuna study and optimize the objective function
print("Running",str(config_map["optuna_trials"]), "trials")
optuna.logging.set_verbosity(optuna.logging.ERROR)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=config_map["optuna_trials"])

# Print the best hyperparameters and f1 score obtained
print('Best F1 Score: {}'.format(study.best_trial.value))
print("Best Hyperparameters: {}".format(study.best_trial.params))


print("Training...........")
# Save the best hyperparameters, study object, data pipeline object, and trained model
params = study.best_trial.params
model_dir = os.path.join(config_map["final_model_save_location"], best_feature_combo, model)
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "hyperparams.json"), "w") as f:
    json.dump(params, f)

classifier = SVC(**params)
classifier.fit(X, y)

joblib.dump(classifier, os.path.join(model_dir, "model.sav"))
joblib.dump(study, os.path.join(model_dir, "optuna_study.sav"))
joblib.dump(data_pipeline, os.path.join(model_dir, "pipeline.sav"))

print("Testing...........")

test_data = ProteinFeatureGenerator(positive_data_file=config_map["test_positive_location"],negative_data_file=config_map["test_negative_location"])

X_test = {
    "AAC-DPC-RScharge-RSDHP-RSpolar":test_data.get_feat_combo(["AAC","DPC","RScharge","RSDHP","RSpolar"]),
    "AAC-RScharge-RSDHP-RSpolar":test_data.get_feat_combo(["AAC","RScharge","RSDHP","RSpolar"]),
    "DPC-RScharge-RSDHP-RSpolar":test_data.get_feat_combo(["DPC","RScharge","RSDHP","RSpolar"]),
    "AAC-DPC-RScharge":test_data.get_feat_combo(["AAC","DPC","RScharge"]),
    "AAC-DPC-RSDHP":test_data.get_feat_combo(["AAC","DPC","RSDHP"]),
    "AAC-RSDHP-RSpolar":test_data.get_feat_combo(["AAC","RSDHP","RSpolar"]),
    "RScharge-RSDHP-RSpolar":test_data.get_feat_combo(["RScharge","RSDHP","RSpolar"]),
    "AAC-RSpolar":test_data.get_feat_combo(["AAC","RSpolar"]),
}
feature_type = best_feature_combo
    
model_dir = os.path.join(config_map["final_model_save_location"],feature_type,"SVC")
os.makedirs(model_dir,exist_ok=True)

pipeline = joblib.load(os.path.join(config_map["final_model_save_location"],feature_type,"SVC","pipeline.sav"))
clf = joblib.load(os.path.join(config_map["final_model_save_location"],feature_type,"SVC","model.sav"))

X,y = X_test[feature_type],test_data.targets
X,y = shuffle(X,y,random_state=config_map["random_seed"])
X = pipeline.transform(X)

y_pred = clf.predict(X)

result_values = evaluate_classification(y_pred,y,class_names=["Not Druggable","Druggable"],save_outputs=model_dir)

print("Test Results...........")
for key, value in result_values.items():
    print(key.capitalize() + ':', value)

print("Saving Text Files...........")
# Save predicted labels for positive and negative data
with open(os.path.join("predictions_pos.txt"), "w") as f:
    for i in range(len(y_pred)):
        if y[i] == True:
            val = "1"
            if y_pred[i]==False: val="0"
            f.write(f"{val}\n")

with open(os.path.join("predictions_neg.txt"), "w") as f:
    for i in range(len(y_pred)):
        if y[i] == False:
            val = "0"
            if y_pred[i]==True: val="1"
            f.write(f"{val}\n")

time.sleep(2)
print("End.")