datasetname="TinyImageNet"
num_clients="100"
com_amount="500"
partition="iid"
dir_val="0.600"
m_name="ConvNet"
us="_"
hs="/"
f_name1="Output"
f_name2=$datasetname$us$num_clients$us$partition$us$dir_val

fedsam_str="FedAvg_sam"
fedasam_str="FedAvg_asam"
fedavgstr="FedAvg"
fedavg_reg_sam_str="FedAvgReg_sam"
fedavgreg_asam_str="FedAvgReg_asam"
fedavgstr3="FedAvgReg3"
fedavgstr4="FedAvgReg4"
fedavgstr5="FedAvgReg5"
data_str="Data"


python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvg --mu_mean=0.2 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --opt_method=SAM --com_amount=$com_amount

old_f=$f_name1$hs$f_name2$hs$fedavgstr

echo $old_f

rm -rf ${old_f}/*.pt

rm -rf ${old_f}/*param*

new_f=$f_name1$hs$f_name2$hs$fedavg_reg_sam_str

echo $new_f

mv $old_f $new_f




python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvg --mu_mean=0.2 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --opt_method=ASAM --com_amount=$com_amount

old_f=$f_name1$hs$f_name2$hs$fedavgstr

echo $old_f

rm -rf ${old_f}/*.pt

rm -rf ${old_f}/*param*

new_f=$f_name1$hs$f_name2$hs$fedavgreg_asam_str

echo $new_f

mv $old_f $new_f

data_dir=$data_str$hs$f_name2

rm -rf $data_dir 

num_clients="100"
partition="Dirichlet"
dir_val="0.300"
us="_"
hs="/"
f_name1="Output"
f_name2=$datasetname$us$num_clients$us$partition$us$dir_val


#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvg --mu_mean=0.2 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --opt_method=SAM --com_amount=$com_amount
#
#old_f=$f_name1$hs$f_name2$hs$fedavgstr
#
#echo $old_f
#
#rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$fedavg_reg_sam_str
#
#echo $new_f
#
#mv $old_f $new_f
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvg --mu_mean=0.2 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --opt_method=ASAM --com_amount=$com_amount
#
#old_f=$f_name1$hs$f_name2$hs$fedavgstr
#
#echo $old_f
#
#rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$fedavgreg_asam_str
#
#echo $new_f
#
#mv $old_f $new_f
#
#data_dir=$data_str$hs$f_name2
#
#rm -rf $data_dir
#
##datasetname="CIFAR100"
#num_clients="100"
#partition="iid"
#dir_val="0.600"
##m_name="ConvNet"
#us="_"
#hs="/"
#f_name1="Output"
#f_name2=$datasetname$us$num_clients$us$partition$us$dir_val
#
#
##mu_mean=0.05 for tiny
##python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvg --mu_mean=0.6 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --opt_method=SAM --com_amount=$com_amount
##
##old_f=$f_name1$hs$f_name2$hs$fedavgstr
##
##echo $old_f
##
##rm -rf ${old_f}/*.pt
##
##rm -rf ${old_f}/*param*
##
##new_f=$f_name1$hs$f_name2$hs$fedavg_reg_sam_str
##
##echo $new_f
##
##mv $old_f $new_f
#
#
#
##mu_mean=0.2 for tiny
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvg --mu_mean=0.6 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --opt_method=ASAM --com_amount=$com_amount
#
#old_f=$f_name1$hs$f_name2$hs$fedavgstr
#
#echo $old_f
#
#rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$fedavgreg_asam_str
#
#echo $new_f
#
#mv $old_f $new_f
#
#data_dir=$data_str$hs$f_name2
#
#rm -rf $data_dir
#
