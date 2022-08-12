# coding: utf8
#!/bin/bash
'''
Author: Peng Bo
Date: 2022-07-25 13:30:40
LastEditTime: 2022-08-12 21:33:34
Description: 

'''

import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt


def tsne_visualize(X, y):
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=100)
    X_tsne = tsne.fit_transform(X)
    # visualize the distributions of landmarks
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    
    # # read all the landmarks in the folder
    # lms_list = []
    # diff_lms_list = []
    # for file in os.listdir(sys.argv[1]):
    #     lms = np.loadtxt(os.path.join(sys.argv[1], file))
    #     diff_lms = lms[1::4, :] - lms[:-1:4, :]
    #     diff_lms_list.append(diff_lms)
    #     lms_list.append(lms)

    
    # # normalize the landmarks
    # lms_whole = np.concatenate(lms_list, axis=0)[:, :14]
    # v_max, v_min = np.max(lms_whole), np.min(lms_whole)
    # normalize_lms_whole = (lms_whole-v_min) / (v_max-v_min)
    
    # diff_lms_whole = np.concatenate(diff_lms_list, axis=0)[:, :14]
    # v_max, v_min = np.max(diff_lms_whole), np.min(diff_lms_whole)
    # normalize_diff_lms_whole = (diff_lms_whole-v_min) / (v_max-v_min)

    
    # labels_list = []
    # diff_labels_list = []
    # for idx, (lms,diff_lms) in enumerate(zip(lms_list, diff_lms_list)):
    #     labels_list = labels_list + [ idx for _ in range(lms.shape[0])]
    #     diff_labels_list = diff_labels_list + [ idx for _ in range(diff_lms.shape[0])]
    
    # labels = np.array(labels_list)
    # diff_labels = np.array(diff_labels_list)

    normalize_lms_whole = np.loadtxt('datasets/dur30_step10_smo5_ratio0.7/lms.txt')
    labels = np.loadtxt('datasets/dur30_step10_smo5_ratio0.7/metas.txt')
    
    tsne_visualize(normalize_lms_whole, labels)
    # tsne_visualize(normalize_diff_lms_whole, diff_labels)