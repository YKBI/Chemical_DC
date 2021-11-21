import os
import numpy as np
import pandas as pd

import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem as dc

from deepchem.utils import download_url, load_from_disk
from deepchem.utils.evaluate import Evaluator

from simtk.openmm.app import pdbfile
from pdbfixer import PDBFixer

from deepchem.utils.vina_utils import prepare_inputs
from sklearn.ensemble import RandomForestRegressor

from time import time
# Activate Tutorial

if __name__ == "__main__":
    data_dir = dc.utils.get_data_dir()
    dataset_file = os.path.join(data_dir, "pdbbind_core_df.csv.gz")
    #dataset_file = os.path.join(data_dir,"pdbbind_v2015.tar.gz")

    if not os.path.exists(dataset_file):
        print('File does not exist. Downloading file...')
        download_url("https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/pdbbind_core_df.csv.gz")
        #download_url("http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/pdbbind_v2015.tar.gz")
        print('File downloaded...')

    raw_dataset = load_from_disk(dataset_file)
    raw_dataset = raw_dataset[['pdb_id', 'smiles', 'label']]
    print(raw_dataset)
    pdbid = raw_dataset["pdb_id"].iloc[1]
    ligand = raw_dataset["smiles"].iloc[1]
    #########################
    # Single Sample Docking #
    #########################
    """
    fixer = PDBFixer(pdbid=pdbid)
    pdbfile.PDBFile.writeFile(fixer.topology,fixer.positions,open("%s.pdb"%pdbid,"w"))

    p,m = None,None

    try:
        p, m = prepare_inputs("%s.pdb"%pdbid,ligand)
    except:
        pritn("%s failed PDB Fixing"%pdbid)

    if p and m:
        print(pdbid,p.GetNumAtoms())
        Chem.rdmolfiles.MolToPDBFile(p,"%s.pdb"%pdbid)
        Chem.rdmolfiles.MolToPDBFile(m,"ligand_%s.pdb"%pdbid)
    finder = dc.dock.binding_pocket.ConvexHullPocketFinder()
    pockets = finder.find_pockets("3cyx.pdb")
    a_time = time()
    vpg = dc.dock.pose_generation.VinaPoseGenerator()

    complexes, scores = vpg.generate_poses(molecular_complex=("3cyx.pdb","ligand_3cyx.pdb"),out_dir="./",generate_scores=True)
    print(scores)
    print(time() - a_time)
    complex_mol = Chem.CombineMols(complexes[0][0],complexes[0][1])

    docker = dc.dock.docking.Docker(pose_generator=vpg)
    pose_complex,score = next(docker.dock(molecular_complex=("3cyx.pdb","ligand_3cyx.pdb"),use_pose_generator_scores=True))"""

    #######################
    # prepare input files #
    #######################
    # example TDO model
    temp_dic = {"4pw8":"[Co+2]","5ti9":"c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N","5tia":"c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N","6a4i":"c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N",\
                "6pyy":"c1cc2c(cc1F)c(c[nH]2)[C@@H]3CC(=O)NC3=O","6pyz":"c1cc2c(cc1F)c(c[nH]2)[C@@H]3CC(=O)NC3=O","6ud5":"c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N","6vbn":"c1ccc2c(c1)-c3cncn3[C@H]2[C@@H]4CCOC[C@H]4O"}
    #pdbids = raw_dataset["pdb_id"].values
    #ligand_smiles = raw_dataset["smiles"].values
    """for i in temp_dic:
        ligand = temp_dic[i]
        pdbid = i
        fixer = PDBFixer(url="https://files.rcsb.org/download/%s.pdb"%pdbid)
        pdbfile.PDBFile.writeFile(fixer.topology,fixer.positions,open("%s.pdb"%pdbid,"w"))
        p,m = None,None
        try:
            p,m = prepare_inputs("%s.pdb"%pdbid,ligand,replace_nonstandard_residues=False,remove_heterogens=False,remove_water=False,add_hydrogens=False)
        except:
            print("%s failed santitization"%pdbid)

        if p and m:
            Chem.rdmolfiles.MolToPDBFile(p,"%s.pdb"%pdbid)
            Chem.rdmolfiles.MolToPDBFile(m,"ligand_%s.pdb"%pdbid)"""
    proteins = [f for f in os.listdir(".") if len(f) == 8 and f.endswith(".pdb")]
    ligands = [f for f in os.listdir(".") if f.startswith("ligand") and f.endswith(".pdb")]

    failures = set([f[:-4] for f in proteins]) - set([f[7:-4] for f in ligands])

    for pdbid in failures:
        proteins.remove(pdbid + ".pdb")

    pdbids = [f[:-4] for f in proteins]
    print(pdbids)
    small_dataset = raw_dataset[raw_dataset["pdb_id"].isin(pdbids)]
    labels = small_dataset.label

    fp_featurizer = dc.feat.CircularFingerprint(size=2048)
    features = fp_featurizer.featurize([Chem.MolFromPDBFile(l) for l in ligands])
    print(labels)
    print(features)

    dataset = dc.data.NumpyDataset(X=features, y=labels, ids=pdbids)

    train_data, test_data = dc.splits.RandomSplitter().train_test_split(dataset, seed=42)
    """
    for (pdbid,ligand) in zip(pdbids,ligand_smiles):
        fixer = PDBFixer(url='https://files.rcsb.org/download/%s.pdb' % (pdbid))
        pdbfile.PDBFile.writeFile(fixer.topology,fixer.positions, open("%s.pdb"%pdbid,"w"))

        p,m = None,None

        try:
            p,m = prepare_inputs("%s.pdb"%pdbid,ligand,replace_nonstandard_residues=False,remove_heterogens=False,remove_water=False,add_hydrogens=False)
        except:
            print("%s failed sanitization"%pdbid)

        if p and m:
            Chem.rdmolfiles.MolToPDBFile(p,"%s.pdb"%pdbid)
            Chem.rdmolfiles.MolToPDBFile(m,"ligand_%s.pdb"%pdbid)"""
    """
    proteins = [f for f in os.listdir(".") if len(f) == 8 and f.endswith(".pdb")]
    ligands = [f for f in os.listdir(".") if f.startswith("ligand") and f.endswith(".pdb")]

    failures = set([f[:-4] for f in proteins]) - set([f[7:-4] for f in ligands])

    for pdbid in failures:
        proteins.remove(pdbid + ".pdb")

    pdbids = [f[:-4] for f in proteins]
    small_dataset = raw_dataset[raw_dataset["pdb_id"].isin(pdbids)]
    labels = small_dataset.label

    fp_featurizer = dc.feat.CircularFingerprint(size=2048)
    features = fp_featurizer.featurize([Chem.MolFromPDBFile(l) for l in ligands])

    dataset = dc.data.NumpyDataset(X=features,y=labels,ids = pdbids)

    train_data,test_data = dc.splits.RandomSplitter().train_test_split(dataset,seed=42)"""

    ##############
    # Make Model #
    ##############
    
    seed = 42
    sklearn_model = RandomForestRegressor(n_estimators=100,max_features="sqrt")
    sklearn_model.random_state =seed

    model = dc.models.SklearnModel(sklearn_model,model_dir="./dc_model/TDO/")
    model.fit(train_data)
    model.save()
    ##############
    # Load Model #
    ##############
    sklearn_model = RandomForestRegressor()
    model = dc.models.SklearnModel(sklearn_model,model_dir="./dc_model/TOD/")
    model.reload()

    ###########
    # Predict #
    ###########

    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

    evaluator = Evaluator(model,train_data,[])
    train_r2score = evaluator.compute_model_performance([metric])
    print("RF train set R^2 : %f"%(train_r2score["pearson_r2_score"]))

    evaluator = Evaluator(model, test_data,[])
    test_r2score = evaluator.compute_model_performance([metric])
    print("RF test set R^2 : %f"%(test_r2score["pearson_r2_score"]))
    """
    for i,j in zip(model.predict(train_data),train_data.y):
        print(i,j)
    
    for i,j in zip(model.predict(test_data),test_data.y):
        print(i,j)"""
"""    
class PLIP_deepchem:
    def __init__(self):

    def make_model(self,i_dir,seed):
        sklearn_model = RandomForestRegressor(n_estimators=100, max_features="sqrt")
        sklearn_model.random_state = seed

        model = dc.models.SklearnModel(sklearn_model, model_dir="./dc_model/")
        model.fit(train_data)
        model.save()
    def load_model(self,md_dir):
    def apply_model(self,a_dir):"""