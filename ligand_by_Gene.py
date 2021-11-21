import pandas as pd
import deepchem as dc
import os,sys,glob
from functools import reduce


class make_and_apply:
    def __init__(self):
        self.featurizer_func = dc.feat.CircularFingerprint(size=1024)
    def make_model(self,md_dir,inf):

        loader = dc.data.CSVLoader(tasks=["y"], feature_field="SMILES", featurizer=self.featurizer_func)
        dataset = loader.create_dataset(inf)
        splitter = dc.splits.RandomSplitter()

        train_data, valid_data, test_data = splitter.train_valid_test_split(dataset=dataset, frac_train=0.6,frac_valid=0.2, frac_test=0.2)

        model = dc.models.MultitaskClassifier(n_tasks=1, n_features=1024, layer_sizes=[1000], dropouts=0.2,learning_rate=0.0001, n_classes=2, model_dir="./dc_model/%s"%md_dir)
        model.fit(train_data)
    def load_model(self,md_dir):
        model = dc.models.MultitaskClassifier(n_tasks=1,n_features=1024,model_dir="./dc_model/%s"%md_dir)
        model.restore()
        return model
    def apply_model(self,md_dir,inf):
        if os.path.exists("./dc_output/%s"%md_dir):
            pass
        else:
            os.makedirs("./dc_output/%s"%md_dir)
        model = self.load_model(md_dir)
        loader = dc.data.CSVLoader(tasks=["y"],feature_field="SMILES",featurizer=self.featurizer_func)
        datas = loader.create_dataset(inf)

        y_true = datas.y
        y_pred = model.predict(datas,transformers=[])
        metric = dc.metrics.roc_auc_score

        t_dic = {}
        for mol_id,y_p in zip(datas.ids,y_pred):
            pos_p = y_p[0][1]
            t_dic[mol_id] = pos_p
        df = pd.DataFrame.from_dict(t_dic,orient="index").reset_index().rename(columns={"index":"SMILES",0:"%s"%md_dir})
        print(df)
        df.to_csv("./dc_output/%s/output.csv"%md_dir,index=False)
        return df
    def apply_many_model(self,inf):
        md_list = os.listdir("./dc_model")
        df_list = []
        for md in md_list :
            df = self.apply_model(md,inf)
            df_list.append(df)
        mdf = reduce(lambda x,y:pd.merge(x,y,on="SMILES"),df_list)
        mdf.to_csv("./dc_output/All_model.out.csv",index=False)

if __name__ == "__main__":
    pp = make_and_apply()
    f_list = glob.glob("model_input/*.csv")
    for f in f_list:
        ns = int(os.path.basename(f).split(".")[1])
        fn = os.path.basename(f).split(".")[0]
        if ns > 300:
            pp.make_model(fn,f)
        else:
            pass
    pp.apply_many_model("./apply_input/yong_test3.csv")