from utils_libs import *
from utils_dataset import *
from utils_models import *

# Global parameters
# os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
max_norm = 10
from minimizers import SAM, ASAM


# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0;
    loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    # batch_size = min(6000, data_x.shape[0])
    batch_size = 64
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval();
    model = model.to(device)
    raw_scores = []
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred, _, _ = model(batch_x)

            # print("y_pred:",y_pred,"\t","batch_y:",batch_y)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


def get_true_pred_scores(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0;
    loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval();
    model = model.to(device)
    raw_scores = []
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred, _, _ = model(batch_x)

            # print("y_pred:",y_pred,"\t","batch_y:",batch_y)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred_raw = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred_raw, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
            ## get confident scores ##
            y_pred_scores = np.max(y_pred_raw, axis=1).reshape(-1)
            raw_scores.append(y_pred_scores[y_pred == batch_y])

    loss_overall /= n_tst
    raw_scores = np.concatenate(raw_scores, axis=0)

    model.train()
    return raw_scores


# --- Helper functions

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


# --- Train functions

def train_model(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, _, _ = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" % (e + 1, acc_trn, loss_trn))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def run_epoch(self, optimizer, criterion):
    minimizer = SAM(optimizer, self.model, self.rho, self.eta)
    running_loss = 0.0
    i = 0
    for inputs, targets in self.trainloader:
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Ascent Step
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        minimizer.ascent_step()

        # Descent Step
        criterion(self.model(inputs), targets).backward()
        minimizer.descent_step()

        with torch.no_grad():
            running_loss += loss.item()

        i += 1
        if i == 0:
            print("Not running epoch", self.id)
            return 0
    return running_loss / i


def train_model_sam(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name,args):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #minimizer = SAM(optimizer, model, 0.05, 0.0)
    if args.rule=='iid' and args.dataset_name == 'TinyImageNet':
        if args.mu_mean > 0:
            minimizer = SAM(optimizer, model, 0.01, 0.0) ##### iid
        else:
            minimizer = SAM(optimizer, model, 0.03, 0.0) ##### iid
    else:
        minimizer = SAM(optimizer, model,args.search_param, 0.0)
    
    #print("search_param:",args.search_param)
    model.train();
    model = model.to(device)

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, var_val, _ = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            # loss = loss / list(batch_y.size())[0]
            loss_var_reg = 0.0
            for val in var_val:
                loss_var_reg += torch.mean(val)
            
            loss += (args.mu_mean/2)*loss_var_reg
            #optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.ascent_step()

            #loss_fn(model(batch_x)[0], batch_y.reshape(-1).long()).backward()
            y_pred,var_val,_ = model(batch_x) 
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            
            loss_var_reg = 0.0
            for val in var_val:
                loss_var_reg += torch.mean(val)
            
            loss += (args.mu_mean/2)*loss_var_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.descent_step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" % (e + 1, acc_trn, loss_trn))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_asam(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name,args):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if args.rule == 'iid' and (args.dataset_name== 'TinyImageNet' or args.mu_mean > 0.0):
        minimizer = ASAM(optimizer, model, 0.1, 0.2)
    else: 
        minimizer = ASAM(optimizer, model, 0.5, 0.2)
        #minimizer = ASAM(optimizer, model, 0.1, 0.2)

    model.train();
    model = model.to(device)

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred,var_val, _ = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            # loss = loss / list(batch_y.size())[0]
            loss_var_reg = 0.0
            for val in var_val:
                loss_var_reg += torch.mean(val)
            
            loss += (args.mu_mean/2)*loss_var_reg
            #optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.ascent_step()
            
            y_pred,var_val,_ = model(batch_x) 
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_var_reg = 0.0
            for val in var_val:
                loss_var_reg += torch.mean(val)
            
            loss += (args.mu_mean/2)*loss_var_reg
            loss.backward()
            #loss_fn(model(batch_x)[0], batch_y.reshape(-1).long()).backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.descent_step()
            # optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" % (e + 1, acc_trn, loss_trn))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_scaffold_mdl(model, model_func, state_params_diff, trn_x, trn_y, learning_rate, batch_size, n_minibatch,
                       print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)

    n_par = get_mdl_params([model_func()]).shape[1]
    n_iter_per_epoch = int(np.ceil(n_trn / batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)
    count_step = 0
    is_done = False

    step_loss = 0;
    n_data_step = 0
    for e in range(epoch):
        # Training
        if is_done:
            break
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, _, _ = model(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = torch.sum(local_par_list * state_params_diff)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0];
            n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay) / 2 * np.sum(params * params)
                print("Step %3d, Training Loss: %.4f" % (count_step, step_loss))
                step_loss = 0;
                n_data_step = 0
                model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_feddyn_mdl(model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate,
                     batch_size, epoch, print_per, weight_decay, dataset_name):
    # print("FedDyn no reg")
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)
    model.train();
    model = model.to(device)

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, _, _ = model(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            # loss_algo = alpha_coef * torch.sum(local_par_list * (local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" % (e + 1, epoch_loss))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_feddyn_mdl_reg(model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate,
                         batch_size, epoch, print_per,
                         weight_decay, dataset_name, batch_dim_lst, args):
    # print("Fed Dyn with reg")
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    model.eval()

    srvr_model_copy = copy.deepcopy(model)
    for param in srvr_model_copy.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)
    model.train();
    model = model.to(device)
    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred, tmp_var_, tmp_mean_ = model(batch_x)

            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            loss_var_algo = 0.0

            for var_val in tmp_var_:
                loss_var_algo += torch.mean(var_val)

            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo + (args.mu_mean / 2) * loss_var_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" % (e + 1, epoch_loss))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model




def train_feddyn_mdl_asam(model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate,
                     batch_size, epoch, print_per, weight_decay, dataset_name):
    # print("FedDyn no reg")
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    minimizer = ASAM(optimizer, model, 0.5, 0.2)

    model.train();
    model = model.to(device)

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, _, _ = model(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            # loss_algo = alpha_coef * torch.sum(local_par_list * (local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            ##### Ascent Steps ########
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.ascent_step()

            
            descent_loss = loss_fn(model(batch_x)[0], batch_y.reshape(-1).long())
            loss_algo_descent = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss_descent = descent_loss + loss_algo_descent
            loss_descent.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.descent_step()

            #optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" % (e + 1, epoch_loss))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model




###
def train_fedprox_mdl(model, avg_model_param_, mu, trn_x, trn_y, learning_rate, batch_size, epoch, print_per,
                      weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)

    n_par = len(avg_model_param_)

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, _, _ = model(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = mu / 2 * torch.sum(local_par_list * local_par_list)
            loss_algo += -mu * torch.sum(local_par_list * avg_model_param_)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += weight_decay / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f" % (e + 1, epoch_loss))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_fedavgreg_mdl(model, avg_model_param_, mu, trn_x, trn_y, learning_rate, batch_size, epoch, print_per,
                        weight_decay, dataset_name, batch_dim_lst, args):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)

    # client_labels = torch.from_numpy(np.squeeze(trn_y))
    # label_count = torch.bincount(client_labels).to(device)
    # label_probs = (1.0*label_count)/torch.sum(label_count)
    batch_index = torch.arange(batch_size).to(device)
    n_par = len(avg_model_param_)
    # torch_inf = torch.tensor(float('inf')).to(device)

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, tmp_var_, tmp_mean_ = model(batch_x)

            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            loss_var_algo = 0.0

            for var_val in tmp_var_:
                loss_var_algo += torch.mean(var_val)

            loss = loss_f_i + (args.mu_mean / 2) * loss_var_algo
            # loss = loss_f_i

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f" % (e + 1, epoch_loss))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False

    model.eval()
    return model


def train_central_model(data_obj, model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, lr_decay_per_round,
                        weight_decay, dataset_name, args):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_per_round)
    model.train();
    model = model.to(device)

    trn_perf_all = np.zeros((epoch, 2))
    tst_perf_all = np.zeros((epoch, 2))
    print("lr_decay_per_round:", lr_decay_per_round)
    print("epoch:", epoch)
    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, mean_list, var_list = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]

            loss_var_algo = 0.0
            # loss_mean_algo =  torch.sqrt(torch.sum((mean_list[-1] )**2))
            if args.add_reg == 1:
                for var_param in var_list:
                    loss_var_algo += torch.mean(var_param)
                # loss = loss + (args.mu_mean/2) *loss_mean_algo + (args.mu_var/2)* loss_var_algo
                loss = loss + (args.mu_var / 2) * loss_var_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()

        scheduler.step()
        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, model, data_obj.dataset)
        tst_perf_all[e] = [loss_tst, acc_tst]
        loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, data_obj.dataset)
        trn_perf_all[e] = [loss_trn, acc_trn]

        print("e:", e, "\t", "trn_loss:", loss_trn, "\t", "tst_loss:", loss_tst, "\t", "acc_trn:", acc_trn, "\t",
              "acc_tst:", acc_tst)

    torch.save(model.state_dict(), 'centralized_reg.pth')
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return trn_perf_all, tst_perf_all


def train_model_FedDC(model, model_func, alpha, local_update_last, global_update_last, global_model_param, hist_i,
                      trn_x, trn_y,
                      learning_rate, batch_size, epoch, print_per,
                      weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device=device)
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, tmp_mean_, tmp_var_ = model(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - hist_i)) * (local_parameter - (global_model_param - hist_i)))
            loss_cg = torch.sum(local_parameter * state_update_diff)

            loss = loss_f_i + loss_cp + loss_cg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_FedDC_reg(model, model_func, alpha, local_update_last, global_update_last, global_model_param, hist_i,
                          trn_x, trn_y,
                          learning_rate, batch_size, epoch, print_per,
                          weight_decay, dataset_name, sch_step, sch_gamma, args):
    n_trn = trn_x.shape[0]
    state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device=device)
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # srvr_model_copy = copy.deepcopy(model)

    # for param in srvr_model_copy.parameters():
    #    param.requires_grad = False

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    client_labels = torch.from_numpy(np.squeeze(trn_y).astype(int))
    # print("client_lables:",client_labels)
    label_count = torch.bincount(client_labels).to(device)
    label_probs = (1.0 * label_count) / torch.sum(label_count)

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            label_weights = -torch.log(label_probs[batch_y.reshape(-1).long()])
            y_pred, tmp_var_, tmp_mean_ = model(batch_x)

            # with torch.no_grad():
            #    spred,smean_list,svar_list = srvr_model_copy(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            loss_var_algo = 0.0

            for var_val in tmp_var_:
                loss_var_algo += torch.mean(var_val)

            # print("loss_var_algo:",loss_var_algo)
            # exit()
            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - hist_i)) * (local_parameter - (global_model_param - hist_i)))
            loss_cg = torch.sum(local_parameter * state_update_diff)
            loss = loss_f_i + loss_cp + loss_cg + ((args.mu_mean / 2) * loss_var_algo)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


# SWA methods
def update_swa_model(swa_model, model, alpha):
    for swa_param, model_param in zip(swa_model.parameters(), model.parameters()):
        swa_param.data = swa_param.data * (1 - alpha)
        swa_param.data += model_param.data * alpha


def schedule_cyclic_lr(rnd, swa_c, lr1, lr2):
    t = 1 / swa_c * (rnd % swa_c + 1)
    lr = (1 - t) * lr1 + t * lr2
    return lr


def schedule_cyclic_tri_lr(rnd, swa_c, lower_bound, upper_bound):
    cycle = np.floor(1 + rnd / (2 * swa_c))
    x = np.abs(rnd / swa_c - 2 * cycle + 1)
    swa_lr = lower_bound + (upper_bound - lower_bound) * np.maximum(0, (1 - x))
    return swa_lr
