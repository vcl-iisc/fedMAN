datasetname="TinyImageNet"
num_clients="100"
partition="Dirichlet"
dir_val="0.300"
reg_val="0.2"
m_name="ConvNet"
us="_"
hs="/"

f_name1="Output"
f_name2=$datasetname$us$num_clients$us$partition$us$dir_val

feddyn_noreg_str="FedDyn_noreg"
feddyn_reg_str="FedDyn_reg"
fedavg_noreg_str="FedAvg_noreg"
fedavgreg_reg_str="FedAvgReg_reg"
feddc_noreg="FedDC_noreg"
feddc_reg="FedDC_reg"
fedavgstr="FedAvg"
feddynstr="FedDyn"
fedspeedstr="FedSpeed"
fedspeedregstr="FedSpeed_reg"
fedspeednoregstr="FedSpeed_noreg"
feddcstr="FedDC"
fedavgregstr="FedAvgReg"




python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvg --mu_mean=0.0 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1

old_f=$f_name1$hs$f_name2$hs$fedavgstr

echo $old_f

#rm -rf ${old_f}/*.pt

rm -rf ${old_f}/*param*

new_f=$f_name1$hs$f_name2$hs$fedavg_noreg_str

echo $new_f

mv $old_f $new_f

python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvgReg --mu_mean=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1

old_f=$f_name1$hs$f_name2$hs$fedavgregstr

#rm -rf ${old_f}/*.pt

rm -rf ${old_f}/*param*

new_f=$f_name1$hs$f_name2$hs$fedavgreg_reg_str

mv $old_f $new_f



#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedDyn --mu_mean=0.0 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$feddynstr
#
##rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$feddyn_noreg_str
#
#mv $old_f $new_f
#
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedDyn --mu_mean=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$feddynstr
#
#rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$feddyn_reg_str
#
#mv $old_f $new_f
#
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedDC --mu_mean=0.0 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$feddcstr
##rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$feddc_noreg
#
#mv $old_f $new_f
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedDC --mu_mean=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$feddcstr
#
#rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$feddc_reg
#
#mv $old_f $new_f

## for fedspeed set --rho=0.01 for iis tinyImageNet setting  
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedSpeed --mu_mean=0.0 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$fedspeedstr
#
##rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$fedspeednoregstr
#
#mv $old_f $new_f

#f_name1="Output"
#f_name2=$datasetname$us$num_clients$us$partition$us$dir_val


#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedSpeed --mu_mean=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$fedspeedstr
#
##rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$fedspeedregstr
#
#mv $old_f $new_f
# 
#data_str="Data"
#
#data_dir=$data_str$hs$f_name2
#
#rm -rf $data_dir 


#dir_val="0.300"
#f_name1="Output"
#f_name2=$datasetname$us$num_clients$us$partition$us$dir_val
#
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedSpeed --mu_mean=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$fedspeedstr
#
##rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$fedspeedregstr
#
#mv $old_f $new_f
# 
#data_str="Data"
#
#data_dir=$data_str$hs$f_name2
#
#rm -rf $data_dir 
#

#####################################################################################################################
#dir_val="0.600"
#f_name1="Output"
#f_name2=$datasetname$us$num_clients$us$partition$us$dir_val
#
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvgReg --mu_mean=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$fedavgregstr
#
##rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$fedavgreg_reg_str
#
#mv $old_f $new_f
#
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedDyn --mu_mean=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$feddynstr
#
#rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$feddyn_reg_str
#
#mv $old_f $new_f
#
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedDC --mu_mean=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$feddcstr
#
#rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$feddc_reg
#
#mv $old_f $new_f
#
#
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedSpeed --mu_mean=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1
#
#old_f=$f_name1$hs$f_name2$hs$fedspeedstr
#
#rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$fedspeedregstr
#
#mv $old_f $new_f
# 
#data_str="Data"
#
#data_dir=$data_str$hs$f_name2
#
#rm -rf $data_dir 
#
