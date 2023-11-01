from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *

### Methods
def train_FedAvg(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, rand_seed=0):
    method_name = 'FedAvg'
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        #clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            #loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            #loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1))) 
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                #np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])

                np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        #os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all


def train_SCAFFOLD(data_obj, act_prob, learning_rate, batch_size, n_minibatch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, rand_seed=0, global_learning_rate=1):
    method_name = 'Scaffold'

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it
        
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    state_param_list = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par    
    clnt_models = list(range(n_clnt))
    
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        state_param_list = np.load('Output/%s/%s/%d_state_param_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):
            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                # Scale down c
                state_params_diff_curr = torch.tensor(-state_param_list[clnt] + state_param_list[-1]/weight_list[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_scaffold_mdl(clnt_models[clnt], model_func, state_params_diff_curr, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, n_minibatch, print_per, weight_decay, data_obj.dataset)

                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_param_list[clnt] - state_param_list[-1] + 1/n_minibatch/learning_rate * (prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_param_list[clnt])*weight_list[clnt]
                state_param_list[clnt] = new_c
                clnt_params_list[clnt] = curr_model_param

            avg_model_params = global_learning_rate*np.mean(clnt_params_list[selected_clnts], axis = 0) + (1-global_learning_rate)*prev_params
            state_param_list[-1] += 1 / n_clnt * delta_c_sum

            avg_model = set_client_from_params(model_func().to(device), avg_model_params)
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1)))     
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                #np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)
                #np.save('Output/%s/%s/%d_state_param_list.npy' %(data_obj.name, method_name, (i+1)), state_param_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])

                np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_state_param_list.npy' %(data_obj.name, method_name, i+1-save_period)) 
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

def train_FedDyn(data_obj, act_prob, learning_rate, batch_size, epoch, com_amount, print_per, weight_decay,  model_func, init_model, alpha_coef, save_period, lr_decay_per_round,
        args=None,rand_seed=0):
    
    method_name  = 'FedDyn' 
    
    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)) # Avg active clients
    fed_mdls_all = list(range(n_save_instances)) # Avg all clients
    fed_mdls_cld = list(range(n_save_instances)) # Cloud models 
    
    trn_perf_sel = np.zeros((com_amount, 2)); trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); tst_perf_all = np.zeros((com_amount, 2))
        
    n_par = len(get_mdl_params([model_func()])[0])
    
    local_param_list = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list  = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
        
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    cld_model = model_func().to(device)
    cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    cld_mdl_param = get_mdl_params([cld_model], n_par)[0]
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_cld.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_cld[j] = fed_model
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        #local_param_list = np.load('Output/%s/%s/%d_local_param_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):
            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                # Train locally 
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True

                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt] # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32, device=device)
                
                if args.add_reg == 0:
                    clnt_models[clnt] = train_feddyn_mdl(model, model_func, alpha_coef_adpt, cld_mdl_param_tensor, local_param_list_curr, trn_x, trn_y, 
                            learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                else:
                    clnt_models[clnt] = train_feddyn_mdl_reg(model, model_func, alpha_coef_adpt, cld_mdl_param_tensor, local_param_list_curr, trn_x, trn_y, 
                            learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset,args)

                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                local_param_list[clnt] += curr_model_par-cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis = 0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

            avg_model = set_client_from_params(model_func(), avg_mdl_param)
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))
            cld_model = set_client_from_params(model_func().to(device), cld_mdl_param) 

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            #loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            #loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            #if i == com_amount -1:
            #    pred_scores_test = get_true_pred_scores(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            #    np.save('Output/%s/%s/%d_com_tst_pred_all.npy' %(data_obj.name, method_name, (i+1)), pred_scores_test)
            #    pred_scores_trn = get_true_pred_scores(cent_x, cent_y, all_model, data_obj.dataset)
            #    np.save('Output/%s/%s/%d_com_trn_pred_all.npy' %(data_obj.name, method_name, (i+1)), pred_scores_trn)

            #    pred_scores_test_sel = get_true_pred_scores(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
            #    np.save('Output/%s/%s/%d_com_tst_pred_sel.npy' %(data_obj.name, method_name, (i+1)), pred_scores_test_sel)
            #    pred_scores_trn_sel = get_true_pred_scores(cent_x, cent_y, avg_model, data_obj.dataset)
            #    np.save('Output/%s/%s/%d_com_trn_pred_sel.npy' %(data_obj.name, method_name, (i+1)), pred_scores_trn_sel)




                            

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1)))
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                torch.save(cld_model.state_dict(), 'Output/%s/%s/%d_com_cld.pt' %(data_obj.name, method_name, (i+1)))

                #np.save('Output/%s/%s/%d_local_param_list.npy' %(data_obj.name, method_name, (i+1)), local_param_list)
                #np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])

                np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                    os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                    os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                    os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                    #os.remove('Output/%s/%s/%d_local_param_list.npy' %(data_obj.name, method_name, i+1-save_period))
                    #os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                fed_mdls_cld[i//save_period] = cld_model
            
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all, fed_mdls_cld

def train_FedProx(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, mu, lr_decay_per_round, rand_seed=0):
    method_name = 'FedProx'

    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
        
    # Average them based on number of datapoints (The one implemented)
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        #clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
    
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            avg_model_param = get_mdl_params([avg_model], n_par)[0]
            avg_model_param_tensor = torch.tensor(avg_model_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_fedprox_mdl(clnt_models[clnt], avg_model_param_tensor, mu, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1)))     
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                #np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])

                np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        #os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all



def train_FedAvgReg(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, mu, lr_decay_per_round, 
        args=None,rand_seed=0):
    method_name = 'FedAvgReg'

    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
        
    # Average them based on number of datapoints (The one implemented)
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    clnt_act_stats = list(range(n_clnt)) 
    
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        #clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
            
        for i in range(com_amount):
            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                #print("act_list:",act_list)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            avg_model_param = get_mdl_params([avg_model], n_par)[0]
            avg_model_param_tensor = torch.tensor(avg_model_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                
                clnt_models[clnt] = train_fedavgreg_mdl(clnt_models[clnt],avg_model_param_tensor, mu, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, 
                        epoch, print_per, weight_decay, data_obj.dataset,args)
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))
            #server_stat_dict = aggregate_client_activations(clnt_act_stats,selected_clnts,weight_list)
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            #loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ###
            #loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1)))     
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])

                np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all




def train_centralized(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, args,rand_seed=0):
    method_name = 'centralized'
    print("centralized")
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
    epoch = args.epoch

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
    
    c_model = model_func().to(device)
    c_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    train_central_data, test_central_data = train_central_model(data_obj,c_model, cent_x, cent_y, learning_rate, batch_size, epoch, print_per,lr_decay_per_round,weight_decay, data_obj.dataset,args)
    np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, epoch), train_central_data)
    np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, epoch), test_central_data)




def train_FedSpeed(data_obj, act_prob,
                  learning_rate, batch_size, epoch, com_amount, test_per,
                  weight_decay, model_func, init_model, alpha_coef,
                  sch_step, sch_gamma, rho, rand_seed=0, lr_decay_per_round=1,args=None):
    
    method_name = 'FedSpeed'
    
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))

    n_client = data_obj.n_client
    client_x = data_obj.clnt_x; client_y=data_obj.clnt_y
    
    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)
    
    weight_list = np.asarray([len(client_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client
    
    train_perf = np.zeros((com_amount, 2))
    test_perf = np.zeros((com_amount, 2))
    
    n_par = len(get_mdl_params([model_func()])[0])
    
    hist_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    client_params_list  = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_client X n_par
    client_models = list(range(n_client))

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    server_model = model_func().to(device)
    server_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    all_model_param = get_mdl_params([all_model], n_par)[0]

    for i in range(com_amount):
        inc_seed = 0
        while(True):
            np.random.seed(i + rand_seed + inc_seed)
            act_list    = np.random.uniform(size=n_client)
            act_clients = act_list <= act_prob
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clients) != 0:
                break

        print('Communication Round', i + 1, flush = True)
        print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clients])))
        all_model_param_tensor = torch.tensor(all_model_param, dtype=torch.float32, device=device)

        del client_models
        client_models = list(range(n_client))

        for client in selected_clients:
            train_x = client_x[client]
            train_y = client_y[client]

            client_models[client] = model_func().to(device)

            model = client_models[client]
            # Warm start from current avg model
            model.load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))
            for params in model.parameters():
                params.requires_grad = True
            # Scale down
            alpha_coef_adpt = alpha_coef / weight_list[client] # adaptive alpha coef
            hist_params_diffs_curr = torch.tensor(hist_params_diffs[client], dtype=torch.float32, device=device)
            client_models[client] = train_model_speed(args,model, model_func, alpha_coef_adpt,
                                                 all_model_param_tensor, hist_params_diffs_curr,
                                                 train_x, train_y, learning_rate * (lr_decay_per_round ** i),
                                                 batch_size, epoch, 5, weight_decay,
                                                 data_obj.dataset, sch_step, sch_gamma, rho, print_verbose=False)
            curr_model_par = get_mdl_params([client_models[client]], n_par)[0]
            hist_params_diffs[client] += curr_model_par-all_model_param
            client_params_list[client] = curr_model_par

        avg_mdl_param_sel = np.mean(client_params_list[selected_clients], axis = 0)
        all_model_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)
        all_model     = set_client_from_params(model_func().to(device), all_model_param)
        server_model  = set_client_from_params(model_func(), np.mean(client_params_list, axis = 0))

        if (i + 1) % test_per == 0:
            loss_test, acc_test = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             server_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_test, loss_test))
            test_perf[i] = [loss_test, acc_test]

            loss_test, acc_test = get_acc_loss(cent_x, cent_y,
                                             server_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Train Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_test, loss_test), flush = True)
            
            train_perf[i] = [loss_test, acc_test]


        if ((i+1) % args.save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1)))     
                torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                #np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                #np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), train_perf[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), test_perf[:i+1])
                if (i+1) > args.save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-args.save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-args.save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-args.save_period))

        # Freeze model
        for params in server_model.parameters():
            params.requires_grad = False
    return




