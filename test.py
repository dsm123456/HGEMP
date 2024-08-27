# -*- coding: UTF-8 -*-

import torch
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.loader import NeighborLoader
from MPLHGE import MPLHGE
import time


def train_a_epoch(s_vector, learning_rate, nodetype):
    total_loss = 0
    # s_vector = torch.zeros(128, dtype=torch.float).to(device)
    for batch in loader:
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        out, s_vector = model(batch, s_vector, nodetype, Batch_size)

        lbl_1 = torch.ones(out.size()[0] // 2)
        lbl_2 = torch.zeros(out.size()[0] // 2)
        lbl = torch.cat((lbl_1, lbl_2)).to(device)

        loss_a_batch = criterion(out, lbl)  # 修改成针对batch的主搜寻节点
        loss_a_batch.backward()
        optimizer.step()

        s_vector = s_vector.clone().detach()
        total_loss += loss_a_batch

    return total_loss.item(), s_vector


def get_embedding(model, data, nodeType, model_saved_path, embedding_save_path):
    try:
        model.load_state_dict(
            torch.load(model_saved_path)
        )
    except FileNotFoundError:
        print('model_saved_path %s no found' % model_saved_path)

    model.eval()
    embedding = model.get_embedding(data, nodeType, None)
    print('embedding.size()=', embedding.size())
    import numpy as np
    np.save(embedding_save_path, embedding.numpy())
    print('embedding is saved at', embedding_save_path)


if __name__ == '__main__':
    
    
    metapaths = [
                 [('business', 'user'), ('user', 'business')],
                 [('business', 'reservation'), ('reservation', 'business')],
                 [('business', 'service'), ('service', 'business')],
                 [('business', 'stars_level'), ('stars_level', 'business')],        
                 [('business', 'user'), ('user', 'business'),('business', 'reservation'), ('reservation', 'business')],
                 [('business', 'user'), ('user', 'business'),('business', 'service'), ('service', 'business')],
                 [('business', 'user'), ('user', 'business'),('business', 'stars_level'), ('stars_level', 'business')],
                 [('business', 'user'), ('user', 'business'),('business', 'reservation'), ('reservation', 'business'),('business', 'service'), ('service', 'business')],
                 [('business', 'user'), ('user', 'business'),('business', 'reservation'), ('reservation', 'business'),('business', 'stars_level'), ('stars_level', 'business')],
                 [('business', 'user'), ('user', 'business'),('business', 'service'), ('service', 'business'),('business', 'stars_level'), ('stars_level', 'business')],
                ]
    
    dataset = torch.load('D:\MPLHGE\Yelp\pro-process\data_0.pt')
    data = dataset
    data = T.ToUndirected()(data)
    data=T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
                   drop_unconnected_nodes=True, max_sample=20)(data)
    print(data)

    print('=================================================================================================')
    neighbor = 10  # neighbor sampling for target node
    # neighbor_step = 3
    neighbor_step = 2  # neighbor_step sampling for target node,(邻居的阶数)
    Batch_size = 2 * 7
    # Batch_size = 3 * 7
    input_dim = 128
    att_dim = 64
    dropout_rate = 0
    epoch = 100
    # epoch = 200
    # epoch = 500
    # epoch = 50
    best_loss = 1000
    learning_rate = 0.0001
    # learning_rate = 0.0005
    # learning_rate = 0.0010
    # learning_rate = 0.0020
    weight_decay = 0
    model_saved_path = 'model_saved/MPLHGE-Yelp-business.pkl'
    embedding_save_path = 'embedding_saved/MPLHGE-Yelp-embedding-business'
    nodeType = 'business'

    loader = NeighborLoader(data,
                            num_neighbors={key: [neighbor] * neighbor_step for key in data.edge_types},
                            batch_size=Batch_size,
                            directed=True,
                            input_nodes=(nodeType, None),
                            )

    print('=================================================================================================')
    model = MPLHGE(
        input_dim=input_dim,
        att_dim=att_dim,
        dropout_rate=dropout_rate
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)
    s_vector = torch.zeros(input_dim, dtype=torch.float).to(device)

    print('=================================================================================================')

    for i in range(epoch):
        start_time = time.perf_counter()
        loss, s_vector = train_a_epoch(s_vector, learning_rate, nodetype=nodeType)
        end_time = time.perf_counter()  # 程序结束时间
        run_time = end_time - start_time  # 程序的运行时间，单位为秒
        print('a epoch time : {:.3f} s'.format(run_time))
        if best_loss > loss:
            best_loss = loss
            # save model
            torch.save(model.state_dict(), model_saved_path)
            print('model saved at %s' % model_saved_path)
        print('epoch %d loss=%.3f best_loss=%.3f' % (i, loss, best_loss))

#get_embedding(model, data, nodeType, model_saved_path, embedding_save_path)
