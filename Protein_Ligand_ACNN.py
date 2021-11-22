import deepchem as dc
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem

import tensorflow as tf

from deepchem.molnet import load_pdbbind
from deepchem.models import AtomicConvModel
from deepchem.feat import AtomicConvFeaturizer

if __name__ == "__main__":
    f1_num_atoms = 100  # maximum number of atoms to consider in the ligand
    f2_num_atoms = 1000  # maximum number of atoms to consider in the protein
    max_num_neighbors = 12  # maximum number of spatial neighbors for an atom

    acf = AtomicConvFeaturizer(frag1_num_atoms=f1_num_atoms,
                          frag2_num_atoms=f2_num_atoms,
                          complex_num_atoms=f1_num_atoms+f2_num_atoms,
                          max_num_neighbors=max_num_neighbors,
                          neighbor_cutoff=4)

    tasks, datasets, transformers = load_pdbbind(featurizer=acf,
                                                 save_dir='.',
                                                 data_dir='.',
                                                 pocket=True,
                                                 reload=False,
                                                 set_name='core')
    train_data,valid_data,test_data = datasets
    acm = AtomicConvModel(n_tasks=1,
                          frag1_num_atoms=f1_num_atoms,
                          frag2_num_atoms=f2_num_atoms,
                          complex_num_atoms=f1_num_atoms + f2_num_atoms,
                          max_num_neighbors=max_num_neighbors,
                          batch_size=12,
                          layer_sizes=[32, 32, 16],
                          learning_rate=0.003,
                          )
    losses,val_losses = [],[]
    max_epochs = 50
    #print(enumerate(train_data.iterbatches(12, deterministic=True, pad_batches=pad_batches)))
    #acm.default_generator(train_data,mode="fit",epochs=50)
    #acm.fit(train_data,nb_epoch=1,max_checkpoints_to_keep=1,all_losses=losses)
    for epoch in range(max_epochs):
        for ind, (F_b,y_b,w_b,ids_b) in enumerate(datasets.iterbatches(batch_size=12,deterministic=True,pad_batches=True)):
            print(F_b)
        """
        loss = acm.fit(train, nb_epoch=1, max_checkpoints_to_keep=1, all_losses=losses)
        #loss = acm.default_generator(train_data,mode="fit",epochs=1)
        metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)
        val_losses.append(acm.default_generator(valid_data,mode="predict",epochs=1))
        #val_losses.append(acm.default_generator(valid_data,mode="predict", metrics=[metric])['rms_score'] ** 2)"""
    for i in val_losses:
        print(i)
    #f,ax = plt.subplots()
    #ax.scatter(range(len(losses)),losses,label="train loss")
    #ax.scatter(range(len(val_losses)),val_losses,label="val loss")
    #plt.legend(loc="upper right")
    #plt.show()