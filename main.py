import pandas as pd
import random
import os
from model import LGCNNDTI
from dataset import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import os
import torch
import xgboost as xgb
import torch.nn as nn
import torch.optim as optim
import argparse

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from torch.utils.data import random_split


def show_result(save_path,lable,Accuracy_List,Precision_List,Recall_List,AUC_List,AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.std(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.std(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.std(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.std(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.std(AUPR_List)
    print("The {} model's results:".format(lable))
    with open("./{}/results.txt".format(save_path), 'a+') as f:
        f.write("\n")
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + ' ')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + ' ')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + ' ')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + ' ')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))


def test_precess(model,pbar,LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, labels = data
            compounds = compounds.cuda()
            proteins = proteins.cuda()
            labels = torch.clamp(torch.round(labels), max=1, min=0).long().cuda()

            predicted_scores = model(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_labels = torch.clamp(torch.round(predicted_scores), min=0, max=1).long().to('cpu').data.numpy()
            predicted_scores = predicted_scores.to('cpu').data.numpy()

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)  
    return Y, P, test_loss, Accuracy, Precision, Reacll, AUC, PRC

def test_model(dataset_load,save_path,DATASET, LOSS, dataset = "Train",lable = "best",save = True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    model = LGCNNDTI(args).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    model.load_state_dict(torch.load(save_path + "valid_best_checkpoint.pth"))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_precess(model,test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET,dataset,lable), 'a+') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results,Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


def K_Fold(datasets, proteins_list, i, k=5, mothod="norm"):
    
    if mothod == "norm":
        fold_size = len(datasets) // k  
        val_start = i * fold_size
        if i != k - 1 and i != 0:
            val_end = (i + 1) * fold_size
            validset = datasets[val_start:val_end]
            trainset = datasets[0:val_start] + datasets[val_end:]
        elif i == 0:
            val_end = fold_size
            validset = datasets[val_start:val_end]
            trainset = datasets[val_end:]
        else:
            validset = datasets[val_start:] 
            trainset = datasets[0:val_start]
    elif mothod == "unseen":
        trainset = []
        validset = []
        fold_size = len(proteins_list) // k
        val_start = i * fold_size
        if i != k - 1 and i != 0:
            val_end = (i + 1) * fold_size
            valid_prot = proteins_list[val_start:val_end]
            train_prot = proteins_list[0:val_start] + proteins_list[val_end:]
        elif i == 0:
            val_end = fold_size
            valid_prot = proteins_list[val_start:val_end]
            train_prot = proteins_list[val_end:]
        else:
            valid_prot = proteins_list[val_start:]
            train_prot = proteins_list[0:val_start]
        
        temp = []
        for prot in valid_prot:
            validset.extend([z for z in datasets if z[0]==prot])
            temp.append(prot)
            if len(validset) > int(len(datasets)*0.2):
                break
        p_train = [z for z in valid_prot if z not in temp]

        if len(p_train) != 0:
            train_prot.extend(p_train)
        for prot in train_prot:
            trainset.extend([z for z in datasets if z[0]==prot])
    
    trainset = pd.DataFrame(trainset, columns=["protein_id", "drug_id", "AAS", "SMILE", "Label"])
    validset = pd.DataFrame(validset, columns=["protein_id", "drug_id", "AAS", "SMILE", "Label"])

    return trainset, validset



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='LGCNN_Bind for Target-Drug Bind')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='use multi_gpu or not')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0, 1, 2', help='device ids of multile gpus')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--task_type', type=str, default='Classification', help='Classification or regression')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjust')
    parser.add_argument('--prot_len', type=int, default=1000, help='The length of Target')
    parser.add_argument('--drug_len', type=int, default=100, help='The length of Ligand')
    parser.add_argument('--prot_embedding', type=int, default=128, help='The dim of Targeting embedding')
    parser.add_argument('--drug_embedding', type=int, default=128, help='The dim of Ligand embedding')
    parser.add_argument('--kernel_model', type=str, default = 'self', help="Choosing the modol of kernel_size")

    parser.add_argument('--drug_kernel', type=str, default = '[13, 5, 3, 11, 5, 7]', help="The size of kernel")
    parser.add_argument('--prot_kernel', type=str, default = '[17, 5, 3, 15, 5, 9]', help="The size of kernel")
    parser.add_argument('--drug_channel', type=str, default = '[128, 128, 128, 128, 128, 128, 128]', help="The count of channel")
    parser.add_argument('--prot_channel', type=str, default = '[128, 128, 128, 128, 128, 128, 128]', help="The count of channel")

    args = parser.parse_args(args=[])

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    xgb_params = {
        "booster" : "gbtree", 
        "nthread" : 4,
        "num_feature" : 1024, 
        "seed" : 1234, 
        "objective" : "binary:logistic", 
        "num_class" : 1, 
        "gamma" : 0.1, 
        "max_depth" : 6, 
        "lambda" : 2, 
        "subsample" : 0.8, 
        "colsample_bytree" : 0.8, 
        "min_child_weight" : 2, 
        "eta" : 0.1 
        }


    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    
    plst = list(xgb_params.items())
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
    for DATA in ["DUDE"]:
        train_input = f"./{DATA}_dataset/{DATA}.csv"
        train_dataset = pd.read_csv(train_input)
        protein_list = []
        for p in train_dataset.protein_id.values.tolist():
            if p not in protein_list:
                protein_list.append(p)
        train_dataset = train_dataset.values.tolist()
        np.random.shuffle(train_dataset)
        np.random.shuffle(protein_list)
    
        for j in range(5):
            train_data_list, test_data_list = K_Fold(train_dataset, protein_list, i=j)

            train_set = train_data_list.values.tolist()
            test_set = test_data_list.values.tolist()

            test_set = CustomDataSet(test_set)
            VT_set = CustomDataSet(train_set)

            train_size = int(len(VT_set)*0.8)
            valid_size = len(VT_set) - train_size
            train_set, valid_set = random_split(VT_set, [train_size, valid_size])

            train_dataset_load = DataLoader(train_set, batch_size=args.batch_size, collate_fn = collate_fn)        
            valid_dataset_load = DataLoader(valid_set, batch_size=args.batch_size, collate_fn = collate_fn)
            test_dataset_load = DataLoader(test_set, batch_size=args.batch_size, collate_fn = collate_fn)

            """ create model"""
            
            model = LGCNNDTI(args).cuda()
            if args.use_multi_gpu:
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)

            weight_p, bias_p = [], []
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for name, p in model.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            optimizer = optim.AdamW([{"params": weight_p, "weight_decay": 1e-4}, {"params": bias_p, "weight_decay": 0}], lr=args.learning_rate)
            Loss = nn.MSELoss()
            save_path = f"./{DATA}_dataset_{j}/"
            note = ""
            writer = SummaryWriter(log_dir=save_path, comment=note)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_results = save_path + "The_results_of_whole_dataset.txt"
            
            print("Training")
            strat = timeit.default_timer()
            
            max_ = 0
            
            for epoch in range(1, args.train_epochs+1):
                train_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(train_dataset_load)),
                total=len(train_dataset_load))
                
                train_losses_in_epoch = []
                model.train()
                for train_i, train_data in train_pbar:
                    train_compounds, train_proteins, train_labels = train_data
                    train_compounds = train_compounds.cuda()
                    train_proteins = train_proteins.cuda()
                    train_labels = train_labels.cuda()

                    optimizer.zero_grad()
                    
                    predicted_interaction = model(train_compounds, train_proteins)
                    train_loss = Loss(predicted_interaction, train_labels)
                    train_losses_in_epoch.append(train_loss.item())
                    train_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    optimizer.step()

                train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss
                writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
                
                valid_pbar = tqdm(
                        enumerate(
                            BackgroundGenerator(valid_dataset_load)),
                        total=len(valid_dataset_load))
                    # loss_dev, AUC_dev, PRC_dev = valider.valid(valid_pbar,weight_CE)
                valid_losses_in_epoch = []
                model.eval()
                Y, P, S = [], [], []
                with torch.no_grad():
                    for valid_i, valid_data in valid_pbar:
                        compounds, proteins, labels = valid_data
                        compounds = compounds.cuda()
                        proteins = proteins.cuda()
                        labels = torch.clamp(torch.round(labels), min=0, max=1).long().cuda()

                        predicted_scores = model(compounds, proteins)
                        loss = Loss(predicted_scores, labels)
                        correct_labels = labels.to('cpu').data.numpy()
                        predicted_labels = torch.clamp(torch.round(predicted_scores), min=0, max=1).long().to('cpu').data.numpy()
                        predicted_scores = predicted_scores.to('cpu').data.numpy()

                        Y.extend(correct_labels)
                        P.extend(predicted_labels)
                        S.extend(predicted_scores)
                        valid_losses_in_epoch.append(loss.item())

                Precision_dev = precision_score(Y, P)
                Reacll_dev = recall_score(Y, P)
                Accuracy_dev = accuracy_score(Y, P)
                AUC_dev = roc_auc_score(Y, S)
                tpr, fpr, _ = precision_recall_curve(Y, S)
                PRC_dev = auc(fpr, tpr)
                valid_loss_a_epoch = np.average(valid_losses_in_epoch)  

                epoch_len = len(str(args.train_epochs))

                print_msg = (f'[{epoch:>{epoch_len}}/{args.train_epochs:>{epoch_len}}] ' +
                                f'train_loss: {train_loss_a_epoch:.5f} ' +
                                f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                                f'valid_AUC: {AUC_dev:.5f} ' +
                                f'valid_PRC: {PRC_dev:.5f} ' +
                                f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                                f'valid_Precision: {Precision_dev:.5f} ' +
                                f'valid_Reacll: {Reacll_dev:.5f} ')

                writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
                # writer.add_scalar('Valid AUC', AUC_dev, epoch)
                writer.add_scalar('Valid AUPR', PRC_dev, epoch)
                writer.add_scalar('Valid Accuracy', Accuracy_dev, epoch)
                writer.add_scalar('Valid Precision', Precision_dev, epoch)
                writer.add_scalar('Valid Reacll', Reacll_dev, epoch)
                writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)

                print(print_msg)
                if Accuracy_dev > max_:
                    max_ = Accuracy_dev
                    torch.save(model.state_dict(), save_path + "valid_best_checkpoint.pth")
                    print("save model")

            testset_test_stable_results,Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
                test_model(test_dataset_load, save_path, DATA, Loss, dataset="Test", lable="stable")
            AUC_List_stable.append(AUC_test)
            Accuracy_List_stable.append(Accuracy_test)
            AUPR_List_stable.append(PRC_test)
            Recall_List_stable.append(Recall_test)
            Precision_List_stable.append(Precision_test)
            with open(save_path + "The_results_of_whole_dataset.txt", 'a+') as f:
                f.write("Test the stable model" + '\n')
                f.write(testset_test_stable_results + '\n')

            """XGBOOST_train"""
            model.eval()
            X_train = None
            y_train = None
            X_valid = None
            y_valid = None
            X_test = None
            y_test = None
            train_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(train_dataset_load)),
                total=len(train_dataset_load))
            for trian_i, train_data in train_pbar:
                '''data preparation '''
                trian_compounds, trian_proteins, trian_labels = train_data
                trian_compounds = trian_compounds.cuda()
                trian_proteins = trian_proteins.cuda()
                trian_labels = trian_labels.cuda()

                predicted_interaction = model.module.fussion(trian_compounds, trian_proteins)
                X = predicted_interaction.cpu().detach().numpy()
                y = trian_labels.cpu().detach().numpy()
                if X_train is None:
                    X_train = X
                else:
                    X_train = np.vstack((X_train, X))
                if y_train is None:
                    y_train = y
                else:
                    y_train = np.hstack((y_train, y))
            dtrain = xgb.DMatrix(X_train, y_train)
            xgb_model = xgb.train(plst, dtrain, num_boost_round = 200)
            xgb_model.save_model(save_path + "xgboost_model")
            """valid"""
            test_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(test_dataset_load)),
                total=len(test_dataset_load))
            with torch.no_grad():
                for valid_i, valid_data in test_pbar:
                    '''data preparation '''
                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.cuda()
                    valid_proteins = valid_proteins.cuda()
                    valid_labels = valid_labels.cuda()

                    valid_scores = model.module.fussion(valid_compounds, valid_proteins)
                    X = valid_scores.cpu().detach().numpy()
                    y = valid_labels.cpu().detach().numpy()
                    if X_valid is None:
                        X_valid = X
                    else:
                        X_valid = np.vstack((X_valid, X))
                    if y_valid is None:
                        y_valid = y
                    else:
                        y_valid = np.hstack((y_valid, y))
                x_valid = xgb.DMatrix(X_valid)
                S = xgb_model.predict(x_valid)
                Y = np.round(y_valid)
                P = np.round(S)
                Precision_dev = precision_score(Y, P)
                Reacll_dev = recall_score(Y, P)
                Accuracy_dev = accuracy_score(Y, P)
                AUC_dev = roc_auc_score(Y, S)
                tpr, fpr, _ = precision_recall_curve(Y, S)
                PRC_dev = auc(fpr, tpr)

                with open("./{}/results.txt".format(save_path), 'a+') as f:
                    f.write("\n")
                    f.write('Accuracy(std):{:.4f} '.format(Accuracy_dev))
                    f.write('Precision(std):{:.4f} '.format(Precision_dev))
                    f.write('Recall(std):{:.4f} '.format(Reacll_dev))
                    f.write('AUC(std):{:.4f} '.format(AUC_dev))
                    f.write('PRC(std):{:.4f}\n'.format(PRC_dev))

                print('Accuracy(std):{:.4f} '.format(Accuracy_dev))
                print('Precision(std):{:.4f} '.format(Precision_dev))
                print('Recall(std):{:.4f} '.format(Reacll_dev))
                print('AUC(std):{:.4f} '.format(AUC_dev))
                print('PRC(std):{:.4f} '.format(PRC_dev))

                AUC_List_stable.append(AUC_dev)
                Accuracy_List_stable.append(Accuracy_dev)
                AUPR_List_stable.append(PRC_dev)
                Recall_List_stable.append(Reacll_dev)
                Precision_List_stable.append(Precision_dev)

    show_result(save_path, "stable",
                            Accuracy_List_stable, Precision_List_stable, Recall_List_stable,
                            AUC_List_stable, AUPR_List_stable)
