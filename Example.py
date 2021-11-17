import pandas as pd
import deepchem as dc
import os,sys,glob

class make_and_apply:
    def __init__(self,inf):
        self._inf = inf
        self.featurizer_func = dc.feat.CircularFingerprint(size=1024)
        self.fn = os.path.basename(inf).split(".")[0]
    def make_model(self):

        loader = dc.data.CSVLoader(tasks=["y"], feature_field="SMILES", featurizer=self.featurizer_func)
        dataset = loader.create_dataset(self._inf)
        splitter = dc.splits.RandomSplitter()

        train_data, valid_data, test_data = splitter.train_valid_test_split(dataset=dataset, frac_train=0.6,frac_valid=0.2, frac_test=0.2)

        model = dc.models.MultitaskClassifier(n_tasks=1, n_features=1024, layer_sizes=[1000], dropouts=0.2,learning_rate=0.0001, n_classes=2, model_dir="./dc_model")
        model.fit(train_data)
    def load_model(self,t):
        model = dc.models.MultitaskClassifier(n_tasks=1,n_features=1024,model_dir=t)
        model.restore()
        return model
    def apply_model(self,inf):

        model = self.load_model("./dc_model/")
        loader = dc.data.CSVLoader(tasks=["y"],feature_field="SMILES",featurizer=self.featurizer_func)
        datas = loader.create_dataset(inf)

        y_true = datas.y
        y_pred = model.predict(datas,transformers=[])
        metric = dc.metrics.roc_auc_score

        t_dic = {}
        for mol_id,y_p in zip(datas.ids,y_pred):
            pos_p = y_p[0][1]
            t_dic[mol_id] = pos_p
        df = pd.DataFrame.from_dict(t_dic,orient="index").reset_index().rename(columns={"index":"SMILES",0:""})
        print(df)

if __name__ == "__main__":
    print(os.getcwd())
    pp = make_and_apply("model_input/BRAF.1302.csv")
    pp.apply_model("./apply_input/yong_test3.csv")


