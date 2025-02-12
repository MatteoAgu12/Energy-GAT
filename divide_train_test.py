import pickle
from Dict_to_Graph import create_graph_data_interpol, create_graph_data_K

def create_graph_set(path_pickle_DB, mode: str):
    with open(path_pickle_DB, "rb") as file:
        DB = pickle.load(file)
    
    print(f'length of DB {len(DB)}')
       
    graphs = []
    if mode == 'interpol': 
        for mat in DB:
            graph = create_graph_data_interpol(mat)
            if graph is not None: graphs.append(graph)
    elif mode == 'K': 
        for mat in DB:
            graph = create_graph_data_K(mat)
            if graph is not None: graphs.append(graph)
    else: raise RuntimeError('The mode you selected was not good.')
    
    print(f'length of graphs {len(graphs)}')
    return graphs

def divide_test_and_train(graphs, path_train, path_test, fraction: int=15, save_on_pkl: bool=True):
    N = len(graphs)
    test_set = []
    train_set = graphs
    N_test = int(fraction * N / 100)
    step = int(N / N_test)
    
    for i in range(N_test):
        index = i * step
        if index < N:
            test_set.append(graphs[index])
        else: break
        
    for item in test_set:
        train_set.remove(item)
    
    if save_on_pkl:
        with open(path_train, "wb") as file:
            pickle.dump(train_set, file)
        print('Train set saved on pickle file.')
    
        with open(path_test, "wb") as file:
            pickle.dump(test_set, file)
        print('Test set saved on pickle file.')
        
    return train_set, test_set

if __name__ == '__main__':
    # Create the train and test sets for interpoling versione of GAT
    path_interpol = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\DB_interpol.pkl'
    path_interpol_train = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TrainSet_interpol.pkl'
    path_interpol_test = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TestSet_interpol.pkl'
    
    graphs_interpol = create_graph_set(path_interpol, 'interpol')
    print(f'DB interpole created with success! Length of DB {len(graphs_interpol)}')
    
    train_interpol, test_interpol = divide_test_and_train(graphs_interpol, path_interpol_train, path_interpol_test)
    print(f'Length of train set: {len(train_interpol)} \nLength of test set: {len(test_interpol)}')
    
    # Create the train and test sets for K-points versione of GAT
    path_K = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\DB_K.pkl'
    path_K_train = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TrainSet_K.pkl'
    path_K_test = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\TestSet_K.pkl'
    
    graphs_K = create_graph_set(path_K, 'K')
    print(f'DB with K-points created with success! Length of DB {len(graphs_K)}')
    
    train_K, test_K = divide_test_and_train(graphs_K, path_K_train, path_K_test)
    print(f'Length of train set: {len(train_K)} \nLength of test set: {len(test_K)}')