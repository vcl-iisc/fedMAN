from utils_general import *
from utils_methods import *
from utils_methods_FedDC import train_FedDC
import random
import sys
import argparse


#### select some random seed for fair comparison ######
seed_val = 83217
#seed_val = 43217
#seed_val = 23217
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_val)
random.seed(seed_val)
torch.cuda.empty_cache()

# Dataset initialization

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
# For Shakespeare experiments
# First generate dataset using LEAF Framework and set storage_path to the data folder
# storage_path = 'LEAF/shakespeare/data/'
#     - In IID use

# name = 'shakepeare'
# data_obj = ShakespeareObjectCrop(storage_path, dataset_prefix)

#     - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, dataset_prefix)
#########


# Generate IID or Dirichlet distribution
# IID
n_client = 100



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,required=True,default='cifar10', help='dataset used for training')
    parser.add_argument('--dataset_name', type=str,required=True,default='CIFAR10', help='dataset use CIFAR10 or CIFAR100')
    parser.add_argument('--batch-size', type=int, default=50, help='input batch size for training (default: 64)')
    parser.add_argument('--epoch', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_client', type=int, default=100,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg_name', type=str, default='FedDyn',
                            help='algorithm: FedDyn/FedAvgReg/FedProx/FedAvg/SCAFFOLD')
    parser.add_argument('--com_amount', type=int, default=500, help='number of maximum communication rounds')
    parser.add_argument('--save_period', type=int, default=100, help='save model and output at this comm rounds')
    parser.add_argument('--add_reg', type=int, default=0, help='add proposed reg to feddyn')
    parser.add_argument('--unbalanced_sgm', type=int, default=0, help='balanced configuration')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--rule', type=str, required=False, default='Dirichlet', help='iid or Dirichlet')
    parser.add_argument('--rule_arg', type=float, default=0.6, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--mu', type=float, default=0.0001, help='the mu parameter for fedprox')
    parser.add_argument('--mu_mean', type=float, default=0.01, help='propsed reg for mean loss')
    parser.add_argument('--mu_var', type=float, default=0.1, help='proposed reg var loss')
    parser.add_argument('--alpha_coef', type=float, default=0.01, help='feddyn reg')
    parser.add_argument('--act_prob', type=float, default=0.1, help='Sample ratio for each communication round')
    parser.add_argument('--lr_decay_per_round', type=float, default=0.998, help='learning deacy per comm round')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning_rate')
    # SWA parameters
    parser.add_argument('-swa', action='store_true', help='server-side swa aggregation flag (default: off)')
    parser.add_argument('--swa_start', type=float, default=None, help='SWA start round')
    parser.add_argument('--swa_c', type=int, default=1, help='SWA model collection frequency')
    parser.add_argument('--swa_lr', type=float, default=1e-4, help='SWA learning rate')
    parser.add_argument('--add_sam', type=int, default=0, help='adding SAM')
    parser.add_argument('--opt_method', type=str, required=False, default='SAM', help='SAM or ASAM')
    parser.add_argument('--search_param', type=float, default=0.05, help='propsed reg for mean loss')
    args = parser.parse_args()
    return args


def main():
    
    args = get_args() 
    data_obj = DatasetObject(dataset=args.dataset_name, n_client=args.n_client, unbalanced_sgm=args.unbalanced_sgm, rule=args.rule, rule_arg=args.rule_arg)

    model_name         = args.model_name      #'cifar100' # Model type
    com_amount         = args.com_amount     # 1000
    save_period        = 100     # 
    weight_decay       = 1e-3
    batch_size         = args.batch_size
    act_prob           = args.act_prob
    lr_decay_per_round = args.lr_decay_per_round
    epoch              = args.epoch
    learning_rate      = args.learning_rate
    print_per          = 20
    alg_name = args.alg_name  ## FedAvg, FedDyn, FedProx, Scaffold, FedBsReg
    
    print("args.alg_name:",args.alg_name )
    if args.model_name == 'cifar100' or  args.model_name == 'cifar100c':
        batch_dim_lst = [64,64,384,192,100]
        #batch_dim_lst = [50176,6400,384,192,100]
    elif args.model_name == 'cifar10' or args.model_name == 'cifar10c':
        batch_dim_lst = [64,64,384,192,10]
        #batch_dim_lst = [784,100,384,192,10]
        #batch_dim_lst = [50176,6400,384,192,10]
    elif args.model_name == 'ConvNet':
        batch_dim_lst = [64,64,64,512,200]    

    # Model function
    model_func = lambda : client_model(model_name)

    init_model = model_func()
    # Initalise the model for all methods or load it from a saved initial model
    init_model = model_func()
    if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
        print("New directory!")
        print(data_obj.name)
        os.mkdir('Output/%s/' %(data_obj.name))
        torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))    
    
    # Methods
    args.Whash = torch.randn(192,192*4)*1.0/torch.sqrt(torch.Tensor([192.0]))
    args.bgash = (torch.zeros(192*4))

    if args.alg_name == 'FedDyn':  
        print('FedDyn')
        alpha_coef = args.alpha_coef
        [fed_mdls_sel_FedFyn, trn_perf_sel_FedFyn, tst_perf_sel_FedFyn,
        fed_mdls_all_FedFyn, trn_perf_all_FedFyn, tst_perf_all_FedFyn,
        fed_mdls_cld_FedFyn] = train_FedDyn(data_obj=data_obj, act_prob=act_prob, 
                                         learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                     save_period=save_period, lr_decay_per_round=lr_decay_per_round,args = args,batch_dim_lst = batch_dim_lst)
    
        tst_perf_alg = tst_perf_all_FedFyn.copy()

    # ###
    elif alg_name == 'SCAFFOLD': 
        print('SCAFFOLD')
        n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
        n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
        n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
        print_per_ = print_per*n_iter_per_epoch

        [fed_mdls_sel_SCAFFOLD, trn_perf_sel_SCAFFOLD, tst_perf_sel_SCAFFOLD,
        fed_mdls_all_SCAFFOLD, trn_perf_all_SCAFFOLD,
        tst_perf_all_SCAFFOLD] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                         batch_size=batch_size, n_minibatch=n_minibatch, com_amount=com_amount,
                                         print_per=print_per_, weight_decay=weight_decay, model_func=model_func,
                                         init_model=init_model, save_period=save_period, lr_decay_per_round=lr_decay_per_round)
        tst_perf_alg = tst_perf_all_SCAFFOLD.copy()
    
    # ####
    elif alg_name == 'FedAvg':
        print('FedAvg')
        [fed_mdls_sel_FedAvg, trn_perf_sel_FedAvg, tst_perf_sel_FedAvg,
        fed_mdls_all_FedAvg, trn_perf_all_FedAvg,
        tst_perf_all_FedAvg] = train_FedAvg(data_obj=data_obj, act_prob=act_prob, 
                                         learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round, args=args)
        tst_perf_alg = tst_perf_all_FedAvg.copy()
        
    # #### 
    elif alg_name == 'FedProx':
        print('FedProx')
        mu = args.mu
        [fed_mdls_sel_FedProx, trn_perf_sel_FedProx, tst_perf_sel_FedProx,
        fed_mdls_all_FedProx, trn_perf_all_FedProx,
        tst_perf_all_FedProx] = train_FedProx(data_obj=data_obj, act_prob=act_prob, 
                                           learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     mu=mu, lr_decay_per_round=lr_decay_per_round)
    
        tst_perf_alg = tst_perf_all_FedProx.copy()

    #### 
    elif alg_name == 'FedAvgReg':
        print('FedAvgReg')
        mu = args.mu
        [fed_mdls_sel_Fedbsreg, trn_perf_sel_Fedbsreg, tst_perf_sel_Fedbsreg,
        fed_mdls_all_Fedbsreg, trn_perf_all_Fedbsreg,
        tst_perf_all_Fedbsreg] = train_FedAvgReg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,                                                           
                batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per,                                                   
                weight_decay=weight_decay,model_func=model_func, init_model=init_model,               
                save_period=save_period,mu=mu, lr_decay_per_round=lr_decay_per_round,args = args,batch_dim_lst = batch_dim_lst)
    
        tst_perf_alg = tst_perf_all_Fedbsreg.copy()

    
    elif alg_name == 'centralized':
        train_centralized(data_obj=data_obj, act_prob=act_prob, 
                                         learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round,args = args)
        #tst_perf_alg = tst_perf_all_FedAvg.copy()

    elif alg_name == 'FedDC':
        print('FedDC')

        epoch = 5
        alpha_coef = 1e-2
        learning_rate = 0.1

        n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
        n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
        n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
        suffix = model_name

        [avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf] = train_FedDC(data_obj=data_obj, act_prob=act_prob, n_minibatch=n_minibatch,learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, 
            init_model=init_model, alpha_coef=alpha_coef,sch_step=1, sch_gamma=1,save_period=save_period, suffix=suffix, trial=False,lr_decay_per_round=lr_decay_per_round,
            args = args)


if __name__ == '__main__':
    main()
