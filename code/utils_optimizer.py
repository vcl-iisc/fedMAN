import torch
import torch.nn.functional as F
import random

class speedOpt(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,beta=1.0,gamma=1.0,adaptive=False,**kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.beta = beta
        self.gamma = gamma
        self.max_norm = 10

        defaults = dict(rho=rho,adaptive=adaptive, **kwargs)
        super(speedOpt, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def ascent_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def descent_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

                if random.random() > self.beta:
                    p.requires_grad = False

    def step(self):
        inputs,targets,loss_fct,model,defined_backward,mu_mean = self.paras
        assert defined_backward is not None, "ERROR: speedOpt requires defined_backward() !"

        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True


        logits,tmp_var_,_ = model(inputs)
        loss = loss_fct(logits,targets)
        
        predictions = logits
        return_loss = loss.clone().detach()
        loss = loss.mean()

        loss_var_algo = 0.0
        for var_val in tmp_var_:
            loss_var_algo += torch.mean(var_val)
        
        loss += (mu_mean/2) *loss_var_algo
        defined_backward(loss)
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_norm)

        self.ascent_step(True)
        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False
        
        logits,tmp_var_,_ = model(inputs)
        loss = loss_fct(logits, targets)
        
        #loss = loss_fct(model(inputs)[0], targets)
        loss = loss.mean()
        
        loss_var_algo = 0.0
        for var_val in tmp_var_:
            loss_var_algo += torch.mean(var_val)
        
        loss += (mu_mean/2)*loss_var_algo

 
        defined_backward(loss)
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_norm)
        self.descent_step(True)

        self.returnthings = (predictions,return_loss)
 

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
