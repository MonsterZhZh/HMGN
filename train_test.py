import torch
from utils import *
import copy
import time



def train(epoches, net, trainloader, testloader, optimizer, scheduler, lr_adjt, dataset, CELoss, tree, device, devices, save_name):
    lr = [0.002, 0.002, 0.002, 0.0002]
    max_val_acc = 0
    best_epoch = 0
    if len(devices) > 1:
        ids = list(map(int, devices))
        netp = torch.nn.DataParallel(net, device_ids=ids)
    for epoch in range(epoches):
        epoch_start = time.time()
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        family_total= 0
        species_total= 0

        idx = 0
        if lr_adjt == 'Cos':
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, epoches, lr[nlr])
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, dataset, device)
            optimizer.zero_grad()
            if len(devices) > 1:
                xc1_sig, xc2_sig, xc3, xc3_sig = netp(inputs)
            else:
                xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
            tree_loss = tree(torch.cat([xc1_sig, xc2_sig, xc3_sig], 1), target_list_sig, device)
            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
            elif dataset == 'Air':
                leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
            if leaf_labels.shape[0] > 0:
                if dataset == 'CUB':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
                elif dataset == 'Air':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 100
                select_fc_soft = torch.index_select(xc3, 0, leaf_labels.squeeze())
                ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
                loss = ce_loss_species + tree_loss
            else:
                loss = tree_loss
                
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    
            with torch.no_grad():
                _, order_predicted = torch.max(xc1_sig.data, 1)
                order_total += order_targets.size(0)
                order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

                _, family_predicted = torch.max(xc2_sig.data, 1)
                family_total += family_targets.size(0)
                family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

                if leaf_labels.shape[0] > 0:
                    select_xc3 = torch.index_select(xc3, 0, leaf_labels.squeeze())
                    select_xc3_sig = torch.index_select(xc3_sig, 0, leaf_labels.squeeze())
                    _, species_predicted_soft = torch.max(select_xc3.data, 1)
                    _, species_predicted_sig = torch.max(select_xc3_sig.data, 1)
                    species_total += select_leaf_labels.size(0)
                    species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                    species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()

        if lr_adjt == 'Step':
            scheduler.step()

        train_order_acc = 100.*order_correct/order_total
        train_family_acc = 100.*family_correct/family_total
        train_species_acc_soft = 100.*species_correct_soft/species_total
        train_species_acc_sig = 100.*species_correct_sig/species_total
        train_loss = train_loss/(idx+1)
        epoch_end = time.time()
        print('Iteration %d, train_order_acc = %.5f,train_family_acc = %.5f,train_species_acc_soft = %.5f,train_species_acc_sig = %.5f, train_loss = %.6f, Time = %.1fs' % \
            (epoch, train_order_acc, train_family_acc, train_species_acc_soft, train_species_acc_sig, train_loss, (epoch_end - epoch_start)))

        test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss = test(net, testloader, CELoss, tree, device, dataset)
        
        if test_species_acc_soft > max_val_acc:
            max_val_acc = test_species_acc_soft
            best_epoch = epoch
            net.cpu()
            # torch.save(net, './models_'+dataset+'/model_'+save_name+'.pt')

            torch.save({'state_dict': net.state_dict()}, './models_'+dataset+'/model_'+save_name+'.pth.tar')
            net.to(device)
        # torch.save({'state_dict': net.state_dict()}, './models_'+dataset+'/model_'+save_name+'.pth.tar')
        
    print('\n\nBest Epoch: %d, Best Results: %.5f' % (best_epoch, max_val_acc))


def test(net, testloader, CELoss, tree, device, dataset):
    epoch_start = time.time()
    with torch.no_grad():
        net.eval()
        test_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        family_total= 0
        species_total= 0

        idx = 0
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, dataset, device)

            xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
            hex_loss = tree(torch.cat([xc1_sig, xc2_sig, xc3_sig], 1), target_list_sig, device)
            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
            elif dataset == 'Air':
                leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 100
            select_fc_soft = torch.index_select(xc3, 0, leaf_labels.squeeze())
            ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
            loss = ce_loss_species + hex_loss

            test_loss += loss.item()
    
            _, order_predicted = torch.max(xc1_sig.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

            _, family_predicted = torch.max(xc2_sig.data, 1)
            family_total += family_targets.size(0)
            family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

            _, species_predicted_soft = torch.max(xc3.data, 1)
            _, species_predicted_sig = torch.max(xc3_sig.data, 1)
            species_total += select_leaf_labels.size(0)
            species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
            species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()


        test_order_acc = 100.* order_correct/order_total
        test_family_acc = 100.* family_correct/family_total
        test_species_acc_soft = 100.* species_correct_soft/species_total
        test_species_acc_sig = 100.* species_correct_sig/species_total
        test_loss = test_loss/(idx+1)
        epoch_end = time.time()
        print('test_order_acc = %.5f,test_family_acc = %.5f,test_species_acc_soft = %.5f,test_species_acc_sig = %.5f, test_loss = %.6f, Time = %.4s' % \
             (test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss, epoch_end - epoch_start))

    return test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss
    