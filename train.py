import time
import numpy as  np
from statistics import mean
from dblpm import DblpmDataset,DblpmDatasetTest,RateDataset
from torch.utils.data import DataLoader
from model import Able
# from rate_model import  MovieNet
from config import model_params
import torch

from torch.optim.lr_scheduler import StepLR
import os
from  graph import helper
from sklearn.metrics import roc_auc_score,accuracy_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.metrics import average_precision_score
# from sklearn.metrics import mean_squared_error #MSE
# from sklearn.metrics import mean_absolute_error #MAE

from tqdm import tqdm




if __name__ == '__main__':
    # pre_fix = "/home/public/zwb/yelp/data/new/"
    # pre_fix= "F:\\zwb\\torch_able\data\imdb\\new\\"
    # pre_fix = "/mnt/driver0/zwb/douban/data/new/"/
    pre_fix = "/home/zwb/zwb/douban/data/new/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('use device:', device)
    model_params["LOOKUP_TABLE"] = torch.from_numpy(helper.load_lookup_from_file(pre_fix + 'node_id_to_class.txt')).to(device)

    for node_att in ['able']:
        # ,'self_att','able','att'
        print('start train mode :',node_att)
        model_params['node_att'] = node_att
        for train_file in ['train/2/3']:
            train_rate = '_2_3'
            rats = [2]
            temp_model = 'temp_douban' + train_rate + '.pth'
            # ,'train_80_1'
            model = Able(model_params)
            # print(model.look_node_type[1])
            model.to(device)
            learning_rate = 1e-3
            batch_size = 1000
            epochs = 50
            # {0.005, 0.01, 0.02,0.05}
            # optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps = 1e-06, weight_decay=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
            criterion = nn.CrossEntropyLoss()
            base_dir = pre_fix + train_file +'/pos/'
            files = sorted(os.listdir(base_dir))
            save_base_path = './models/'
            save_model_path = './models/' + train_file +'/'
            save_all_path = save_model_path + 'all/'
            if not os.path.exists(save_base_path):
                os.mkdir(save_base_path)
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            if not os.path.exists(save_all_path):
                os.makedirs(save_all_path)
            min_loss = float('inf')
            # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            best_rmse = 999
            train_early_stop = 1
            for t in range(epochs):
                if train_early_stop > 10:
                    print('train early stop,10 times')
                    break
                model.train()
                epoch_loss1 = 0.0
                for file in files:
                    # print(file, 'start train....')
                    pos_train = pre_fix + train_file +'/pos/' + file
                    neg_train = pre_fix +train_file +'/neg/' + 'neg_' + file
                    training_data = DblpmDataset(pos_train, neg_train)
                    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=8)
                    for _, (X, y) in enumerate(train_dataloader):

                        z = torch.cat((X, y), dim=0)
                        z = z.to(device)
                        optimizer.zero_grad()

                        # label_1 = torch.ones((X.shape[0], 1), dtype=torch.long)
                        # label_0 = torch.zeros((X.shape[0], 1), dtype=torch.long)
                        # pred = model.calculate(z)
                        # label = torch.cat((label_1, label_0), dim=0)
                        # label = torch.squeeze(label, dim=-1).to(device)
                        # loss = criterion(pred, label)
                        loss = model(z)
                        # Backpropagation
                        loss.backward()
                        optimizer.step()
                        epoch_loss1 += loss.item()
                    # 每隔30epoch更新学习率
                    # scheduler.step()
                print('Epoch [{}/{}], Loss: {:.4f}'.format(t + 1, epochs, epoch_loss1))
                # print('saving temp model:',temp_model)
                torch.save(model.state_dict(), temp_model)

                for rat in rats:
                    # print('user model able...')
                    model_params['mod_switch'] = 2
                    model2 = Able(model_params)

                    # print('loading paramas ... ', temp_model)
                    static_dic = torch.load(temp_model)
                    model2.load_state_dict(static_dic)

                    model2.to(device)
                    learning_rate = 0.001
                    batch_size = 128
                    epochs1 = 200
                    # 损失函数和优化器
                    criterion = nn.MSELoss()
                    # {0.005, 0.01, 0.02,0.05}

                    optimizer1 = torch.optim.Adam(model2.parameters(), lr=0.0005, weight_decay=0.001)
                    # scheduler1 = StepLR(optimizer1, step_size=10, gamma=0.1)
                    # 加载数据
                    train_rate_file = pre_fix + '/rate/3/' + str(rat) + '/new_u_m.txt.' + str(rat)
                    test_rate_file = pre_fix + '/rate/3/' + str(rat) + '/test_new_u_m.txt.' + str(rat)
                    # print('test data:',test_rate_file)
                    train_data = RateDataset(train_rate_file)
                    test_data = RateDataset(test_rate_file)
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                    # print('test_leng', len(test_data))
                    # test_bacth = len(test_data) + 1
                    test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)
                    rmse_list = []
                    mae_list = []

                    evalution_early_stop = 1
                    for epoch1 in range(epochs1):
                    #     model2.train()
                    #     epoch_loss = 0.0
                    #     for inputs in train_loader:
                    #         # print(inputs)
                    #         user_index, movie_index, label = torch.chunk(inputs, 3, dim=-1)
                    #         user_index = torch.as_tensor(user_index, dtype=torch.long)
                    #         movie_index = torch.as_tensor(movie_index, dtype=torch.long)
                    #         edge_index = torch.zeros(user_index.shape, dtype=torch.long)
                    #         # train data
                    #         input_tensor = torch.cat((user_index, edge_index, movie_index), dim=-1).to(device)
                    #         # print(input_tensor)
                    #         # labels
                    #         label = torch.as_tensor(label, dtype=torch.float)
                    #         label = torch.squeeze(label, dim=-1).to(device)
                    #         # print('label',label)
                    #         optimizer1.zero_grad()
                    #         outputs = model2(input_tensor)
                    #         loss = criterion(outputs, label)
                    #         loss.backward()
                    #         optimizer1.step()
                    #         # print(loss)
                    #         epoch_loss += loss.item()
                            # print('out',outputs)
                        # scheduler1.step()
                        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch1 + 1, epochs1, epoch_loss))

                        # 在测试集上进行测试并计算RMSE
                        with torch.no_grad():
                            model2.eval()
                            targets_all = []
                            predictions_all = []
                            for inputs in test_loader:
                                user_index, movie_index, label = torch.chunk(inputs, 3, dim=-1)
                                user_index = torch.as_tensor(user_index, dtype=torch.long)
                                movie_index = torch.as_tensor(movie_index, dtype=torch.long)
                                edge_index = torch.zeros(user_index.shape, dtype=torch.long)
                                # test data
                                input_tensor = torch.cat((user_index, edge_index, movie_index), dim=-1).to(device)

                                # labels
                                label = torch.as_tensor(label, dtype=torch.float)
                                label = torch.squeeze(label, dim=-1).to(device)

                                outputs = model2(input_tensor)
                                predicted = outputs
                                # _, predicted = torch.max(outputs.data, 1)
                                predicted = predicted.cpu()
                                # label = label -1
                                label = label.cpu()
                                # print(predicted)
                                targets_all.append(label.numpy())
                                predictions_all.append(predicted.numpy())
                                # print('RMSE of the model on the test dataset: ', rmse)
                                # print('MAE of the model on the test dataset: ', MAE)
                            targets_all = np.concatenate(targets_all)
                            predictions_all = np.concatenate(predictions_all)
                            rmse = mean_squared_error(targets_all, predictions_all, squared=False)
                            MAE = mean_absolute_error(targets_all, predictions_all)
                            MAPE = mean_absolute_percentage_error(targets_all,predictions_all)
                            rmse_list.append(rmse)
                            mae_list.append(MAE)
                            if rmse < best_rmse:
                                evalution_early_stop = 0
                                best_rmse = rmse
                                save_path = save_base_path + train_file + '/' + model_params[
                                    'node_att'] + '_nodeDim_' + str(
                                    model_params['NODE_EMBEDDING_DIM']) + '_book_best' + '.pth'
                                torch.save(model.state_dict(), save_path)
                                print('====================best result  start=============',rat)
                                print(epoch1,'at epoch.save best evalution result model done ! best_rmse:',best_rmse,'best mea:',MAE,'mape',MAPE)
                                print('====================best result  end =============')
                                # np.savetxt(save_model_path+'rmse_'+str(best_rmse)+'_'+str(epoch1)+'.txt', np.array(rmse_list), fmt='%s')
                                # np.savetxt(save_model_path+'mae_'+ str(MAE) + '_' + str(epoch1) + '.txt', np.array(mae_list), fmt='%s')
                            else:
                                evalution_early_stop += 1
                            if evalution_early_stop > 30:
                                print('over 30 times ,evalution early stop')
                                break
                            # print(epoch1, 'epoch: Test set: Average loss: {:.4f}, RMSE: {:.4f}'.format(epoch_loss, rmse))
                            # print(epoch,'epoch Test set: Average loss: {:.4f}, RMSE: {:.4f}'.format(loss, MAE))
                    mean_rmse = np.array(rmse_list)
                    mean_mae = np.array(mae_list)
                    me_rmse = mean_rmse.mean()
                    ma_mae = mean_mae.mean()
                    # print('train rate:',rat, '----,rmse:', me_rmse, 'mae:', ma_mae, 'min rmse:', mean_rmse.min(), 'min mae:',
                    #       mean_mae.min())
                    if mean_rmse.min() > best_rmse:
                        train_early_stop += 1
                    # else:
                    #     rmse_file_name = save_model_path + 'all/' + node_att + '_' + str(rat) + '_mean_rmse_' + str(
                    #         me_rmse) + 'min_' + str(mean_rmse.min()) + '_train_epoch_' + str(t) + '_.txt'
                    #     mae_file_name = save_model_path + 'all/' + node_att + '_' + str(rat) + '_mean_mae_' + str(
                    #         ma_mae) + 'min_' + str(mean_mae.min()) + '_train_epoch_' + str(t) + '_.txt'
                    #     np.savetxt(rmse_file_name, mean_rmse, fmt='%s')
                    #     np.savetxt(mae_file_name, mean_mae, fmt='%s')
                # 保存模型
                # 1、保存Loss最小的模型s
                # 2、制作测试集，保存测试机上auc最小的模型s
            #     if epoch_loss1 < min_loss:
            #         min_loss = epoch_loss1
            #         print('new min loss,save model....new min loss is:', min_loss)
            #         save_path = save_base_path + train_file + '/' + model_params['node_att'] + '_nodeDim_' + str(model_params['NODE_EMBEDDING_DIM']) + '_yelp' + '.pth'
            #         torch.save(model.state_dict(), save_path)
            #
            # save_path_last = save_base_path + train_file + '/' + model_params['node_att'] + '_nodeDim_' + str(model_params['NODE_EMBEDDING_DIM']) + '_yelp_last' + '.pth'
            # torch.save(model.state_dict(), save_path_last)




