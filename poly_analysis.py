import pickle
import numpy as np
import sklearn
import itertools

import pandas as pd
from sklearn.manifold import TSNE

import seaborn as sns

from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

def plot_pca(data, labels, data_name):
    colors = plt.cm.tab10.colors

    import os

    # Set the environment variable
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    pca = TSNE(n_components=2)
    X_pca = pca.fit_transform(data)

    plt.figure(figsize=(12,12))
    plt.style.use('ggplot')
    for k in np.unique(labels):
        X = X_pca[labels == k]
        plt.scatter(X[:, 0], X[:, 1], color=colors[k], label=f'Dataset {data_name[k]}')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA')
    plt.savefig(f'{data_name[0]}_{data_name[1]}_feat_2jigsaw_tsne.png')




def decision_tree(X_train, y_train, X_test, y_test, X_test_prompt, data_name):
    
    # collapse all labels but toxicity

        
    clf = tree.DecisionTreeClassifier(max_depth=8, class_weight='balanced')
    clf_ = clf.fit(X_train, y_train)
    test_accuracy = round(clf_.score(X_test,y_test),2)*100
    
    predict = clf_.predict(X_test)
    error_idx = np.where(predict != y_test)[0]
    
    df = pd.DataFrame({'prompt': X_test_prompt[error_idx], 'pred': predict[error_idx], 'label': y_test[error_idx] })
    df.to_csv('miss_classified.csv')
    
    print('Accuracy Train= ', clf_.score(X_train,y_train))
    print('Accuracy Test= ', test_accuracy)
    print('Tree Depth= ', clf_.get_depth())
    
    # feat_importance = clf_.tree_.compute_feature_importances(normalize=False)
    # explanatory_features = feat_importance.argmax()
    

    # data_plot = X_test[:, [explanatory_features, explanatory_features +1]]
    # # Separate data by label
    # label_0 = [data_plot[i] for i in range(len(X_test)) if y_test[i] == 0]
    # label_1 = [data_plot[i] for i in range(len(X_test)) if y_test[i] == 1]

    # # Convert to lists of x and y coordinates for plotting
    # x_label_0, y_label_0 = zip(*label_0)
    # x_label_1, y_label_1 = zip(*label_1)


    # Accessing the tree structure
    tree_structure = clf_.tree_

    # Printing split values for each decision
    # for node in range(tree_structure.node_count):
    #     if tree_structure.children_left[node] != tree._tree.TREE_LEAF:
    #         feature_index = tree_structure.feature[node]
    #         threshold = tree_structure.threshold[node]
            
    #         num_statistic = feature_index % 7  
    #         num_layer = feature_index  // 7
            
            #print(f"Decision node {node}: Feature: (layer) {num_layer} (statistic) {num_statistic} <= {threshold}")




    # plt.figure(figsize=(12,12))
    # plt.style.use('ggplot')
    # # Plotting
    # plt.scatter(x_label_0, y_label_0, color='red', label='Toxicity Samples')
    # plt.scatter(x_label_1, y_label_1, color='blue', label='Healthy Samples')
    # #plt.ylabel(f'Feature Importance: {feat_importance[explanatory_features +1]}')
    # #plt.xlabel(f'Feature Importance: {feat_importance[explanatory_features]}')
    # plt.title(f"Decision Tree - Depth {clf_.get_depth()}  - Test Accuracy {test_accuracy}")
    # plt.legend()
    # plt.savefig('scatterplot.png')
    
    
    
    conf_matrix = confusion_matrix(y_test, clf_.predict(X_test))

    # Plotting the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f"Decision Tree - Depth {clf_.get_depth()}  - Test Accuracy {test_accuracy}")
    plt.savefig(f'{data_name[0]}_{data_name[1]}_feat_2jigsaw_confusion_depth_{clf_.get_depth()}.png')
    plt.show()
    return clf_



def get_inference_time(data_path):
    data = pd.read_csv(data_path)
    concatenated_dict = {}
    keys = eval(data['inference_time'][0])
    concatenated_dict = {key: [] for key in keys}
    for idx, row in enumerate(data['inference_time']):
        if idx == 0:
            continue
        for key in keys:
            concatenated_dict[key].extend(eval(row)[key])    
    return concatenated_dict

if __name__ == "__main__":

    data_names = ["jigsaw_insult", "toxic_pile",'hotel', "FreeLaw","PubMed Abstracts", "DM Mathematics", "USPTO Backgrounds", "Github"] #"toxicity"  "knowledge", 
    #data_names = ['jigsaw_toxic', 'jigsaw_non_toxic'] #
    data_competition = [ ["jigsaw_insult",  "FreeLaw"], ["jigsaw_insult","PubMed Abstracts" ], ["jigsaw_insult","DM Mathematics" ] , ["jigsaw_insult","USPTO Backgrounds" ]  , ["jigsaw_insult","Github" ], 
                        ["jigsaw_insult", "toxic_pile"], ["toxic_pile",  "FreeLaw"], ["toxic_pile","PubMed Abstracts" ], ["toxic_pile","DM Mathematics" ] , ["toxic_pile","USPTO Backgrounds" ]  , ["toxic_pile","Github" ]]
    for data_names in data_competition:
        data = {}
        labels = {}
        selected_layer = ...
        selected_stat = ...

        for k, data_name in enumerate(data_names):
            data_path = f"/home/ubuntu/polytope/{data_name}/statistics.csv"
            data[data_name] = pd.read_csv(data_path)
            #shape data (num_data, num_layers, num_statistics)
            labels[data_name] = [k]*len(data[data_name])
            
            
        X_ = []
        import ast 
        X_prompt = []
        for data_name in data_names:
            for index, row in data[data_name].iterrows():
                result_list = ast.literal_eval(row['stats'])

                X_.append(np.array(result_list)[:, 2].reshape(-1))
                X_prompt.append(row['prompt'])
            

        Y_ = np.concatenate([labels[key] for key in data_names ])
        idx_permut = np.random.permutation(len(X_))
        X_ = np.array(X_)[idx_permut]
        Y_ = np.array(Y_)[idx_permut]
        print('SHAPE:', X_.shape)
        X_prompt = np.array(X_prompt)[idx_permut]


        X_pca = X_[:10000]
        Y_pca = Y_[:10000]
        # X_prompt = X_prompt[:20000]
        plot_pca(X_pca, Y_pca, data_names)
        # Y_[Y_== 1] = 0
        # for k in np.unique(Y_):
        #     if k != 0:
        #         Y_[Y_ == k] = 1
        
        X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.3, random_state=3, stratify=Y_)
        X_train_prompt, X_test_prompt, _, _ = train_test_split(X_prompt, Y_, test_size=0.3, random_state=3, stratify=Y_)
        print('Number toxic samples in trainset:', len(y_train[y_train ==0]))
        print('Number clean samples in trainset:', len(y_train[y_train ==1]))
        print('\n')
        print('Number toxic samples in testset:', len(y_test[y_test ==0]))
        print('Number clean samples in testset:', len(y_test[y_test ==1]))


        clf_ = decision_tree(X_train, y_train, X_test, y_test, X_test_prompt, data_names)





        # #['jigsaw_toxic', 'jigsaw_non_toxic'] #

        # data = {}
        # labels = {}

        # data_path = f"/home/ubuntu/polytope/jigsaw_insult/statistics.csv"
        # data_jig = pd.read_csv(data_path)
        # labels_jig = [0]*len(data_jig)
            

        # X_jig = []
        # import ast 
        # X_jig_prompt = []
        # for index, row in data_jig.iterrows():
        #     result_list = ast.literal_eval(row['stats'])
        #     X_jig.append(np.array(result_list)[:, 2].reshape(-1))
        #     X_jig_prompt.append(row['prompt'])
            


        # print('Jigsaw Test= ', clf_.score(X_jig, labels_jig))
