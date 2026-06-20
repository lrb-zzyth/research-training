import argparse
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from util.task_util import cal_topo_emb, accuracy, construct_graph, DiversityLoss
from util.base_util import seed_everything, load_dataset
from model import GCN, FedTAD_ConGenerator, ConditionalDiffusionGenerator

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()


# experimental environment setup
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--root', type=str, default='/home/ai2/work/fedtad/dataset')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default="Cora")
parser.add_argument('--partition', type=str, default="Louvain", choices=["Louvain", "Metis"])
parser.add_argument('--part_delta', type=int, default=20)
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--num_rounds', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--num_dims', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hid_dim', type=int, default=64)



# for fedtad
parser.add_argument('--glb_epochs', type=int, default=5)
parser.add_argument('--it_g', type=int, default=1)
parser.add_argument('--it_d', type=int, default=5)
parser.add_argument('--lr_g', type=float, default=1e-3)
parser.add_argument('--lr_d', type=float, default=1e-3)
parser.add_argument('--fedtad_mode', type=str, default='raw_distill', choices=['raw_distill', 'rep_distill'])
parser.add_argument('--num_gen', type=int, default=100)
parser.add_argument('--lam1', type=float, default=1)
parser.add_argument('--lam2', type=float, default=1)
parser.add_argument('--topk', type=float, default=5)


# loss improvements
parser.add_argument('--use_weighted_ce', action='store_true', default=False)
parser.add_argument('--class_weight_method', type=str, default='effective_num', choices=['inverse', 'effective_num'])
parser.add_argument('--beta', type=float, default=0.999)
parser.add_argument('--use_contrastive', action='store_true', default=False)
parser.add_argument('--lambda_cl', type=float, default=0.1)
parser.add_argument('--cl_tau', type=float, default=0.5)


# diffusion generator
parser.add_argument('--generator_type', type=str, default='mlp', choices=['mlp', 'diffusion'])
parser.add_argument('--diffusion_steps', type=int, default=50)
parser.add_argument('--diffusion_hidden', type=int, default=256)
parser.add_argument('--diffusion_beta_start', type=float, default=1e-4)
parser.add_argument('--diffusion_beta_end', type=float, default=0.02)


args = parser.parse_args()

def compute_class_weights(labels, num_classes, method='effective_num', beta=0.999):
    """
    Compute per-class weights for Weighted CrossEntropy Loss.
    labels: 1D tensor of class labels from local train_mask.
    """
    n_c = torch.bincount(labels, minlength=num_classes).float()

    if method == 'effective_num':
        # effective number of samples: weight_c = (1 - beta) / (1 - beta ** n_c)
        weight_c = (1 - beta) / (1 - beta ** n_c)
        weight_c[n_c == 0] = 0.0
    elif method == 'inverse':
        N = labels.shape[0]
        weight_c = N / (num_classes * n_c)
        weight_c[n_c == 0] = 0.0
    else:
        raise ValueError(f"Unknown class_weight_method: {method}")

    # Normalize non-zero weights to have mean ~1
    non_zero = weight_c > 0
    if non_zero.sum() > 0:
        weight_c[non_zero] = weight_c[non_zero] / weight_c[non_zero].mean()

    return weight_c


def supervised_contrastive_loss(embeddings, labels, tau=0.5):
    """
    Supervised Contrastive Loss (SupCon) on train nodes.
    embeddings: [N, D] tensor from model penultimate layer.
    labels: [N] class labels.
    tau: temperature scaling.
    Returns scalar loss (0 if no valid positive pair exists).
    """
    device = embeddings.device
    labels = labels.view(-1)

    # Normalize embeddings to unit sphere
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Similarity matrix: [N, N]
    sim = torch.mm(embeddings, embeddings.t()) / tau

    # Positive mask: same-class pairs (excluding self)
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    identity = torch.eye(labels.shape[0], device=device, dtype=torch.bool)
    pos_mask = label_eq & ~identity

    # If no valid positive pair, return 0 to avoid NaN
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    # Numerical stability: subtract row-wise max
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()
    exp_sim = torch.exp(sim_stable)

    # Denominator: sum over all negatives (all j != i)
    neg_mask = ~identity
    denom = (exp_sim * neg_mask.float()).sum(dim=1)  # [N]

    # Numerator: sum of positive-pair exps per anchor
    pos_exp = exp_sim * pos_mask.float()
    numer = pos_exp.sum(dim=1)  # [N]

    # Only consider anchors with at least one positive pair
    valid = numer > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    log_prob = torch.log(numer[valid] / denom[valid])
    loss = -log_prob.mean()
    return loss


if __name__ == "__main__":
    
    seed_everything(seed=args.seed)
    dataset = load_dataset(args)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device(f"cuda:{args.gpu_id}")
    subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]
    local_models = [ GCN(feat_dim=subgraphs[client_id].x.shape[1], 
                         hid_dim=args.hid_dim, 
                         out_dim=dataset.num_classes,
                         dropout=args.dropout).to(device)
                    for client_id in range(args.num_clients)]
    local_optimizers = [Adam(local_models[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay) for client_id in range(args.num_clients)]
    global_model = GCN(feat_dim=subgraphs[0].x.shape[1], 
                         hid_dim=args.hid_dim, 
                         out_dim=dataset.num_classes,
                         dropout=args.dropout).to(device)
    
    feat_dim = args.hid_dim if args.fedtad_mode == 'rep_distill' else subgraphs[0].x.shape[1]
    if args.generator_type == 'diffusion':
        generator = ConditionalDiffusionGenerator(
            feat_dim=feat_dim,
            num_classes=dataset.num_classes,
            hidden_dim=args.diffusion_hidden,
            num_steps=args.diffusion_steps,
            beta_start=args.diffusion_beta_start,
            beta_end=args.diffusion_beta_end,
        ).to(device)
        print(f"[generator] ConditionalDiffusionGenerator (steps={args.diffusion_steps}, "
              f"hidden={args.diffusion_hidden}, feat_dim={feat_dim})")
    else:
        generator = FedTAD_ConGenerator(noise_dim=32, feat_dim=feat_dim,
                                        out_dim=dataset.num_classes, dropout=0).to(device)
    global_optimizer = Adam(global_model.parameters(), lr=args.lr_d, weight_decay=args.weight_decay)
    generator_optimizer = Adam(generator.parameters(), lr=args.lr_g, weight_decay=args.weight_decay)

    best_server_val = 0
    best_server_test = 0


    if os.path.exists(f"./ckr/{args.dataset}_{args.partition}_{args.num_clients}.pt"):
        ckr = torch.load(f"./ckr/{args.dataset}_{args.partition}_{args.num_clients}.pt").to(device)
    else:
        os.makedirs("./ckr", exist_ok=True)
        ckr = torch.zeros((args.num_clients, dataset.num_classes)).to(device)
        for client_id in range(args.num_clients):
            data = subgraphs[client_id]  
            graph_emb = cal_topo_emb(edge_index=data.edge_index, num_nodes=data.x.shape[0], max_walk_length=5).to(device)    
            ft_emb = torch.cat((data.x, graph_emb), dim=1).to(device)
            for train_i in data.train_idx.nonzero().squeeze():
                neighbor = data.edge_index[1,:][data.edge_index[0, :] == train_i] 
                node_all = 0
                for neighbor_j in neighbor:
                    node_kr = torch.cosine_similarity(ft_emb[train_i], ft_emb[neighbor_j], dim=0)
                    node_all += node_kr
                node_all += 1
                node_all /= (neighbor.shape[0] + 1)
                
                label = data.y[train_i]
                ckr[client_id, label] += node_all
        torch.save(ckr, f"./ckr/{args.dataset}_{args.partition}_{args.num_clients}.pt")
    
    
    normalized_ckr = ckr / ckr.sum(0)


    # Pre-compute per-client class weights for Weighted CE (if enabled)
    if args.use_weighted_ce:
        class_weights = []
        for client_id in range(args.num_clients):
            train_labels = subgraphs[client_id].y[subgraphs[client_id].train_idx]
            w = compute_class_weights(train_labels, dataset.num_classes,
                                      args.class_weight_method, args.beta)
            class_weights.append(w.to(device))
        print(f"[class weights] computed for {args.num_clients} clients "
              f"using '{args.class_weight_method}' method (beta={args.beta})")


    l_glb_acc_test = []
    
    for round_id in range(args.num_rounds):
        global_model.eval()
        generator.eval()
        
        # global model broadcast
        for client_id in range(args.num_clients):
            local_models[client_id].load_state_dict(global_model.state_dict())
        
        
        # global eval
        global_acc_val = 0
        global_acc_test = 0
        for client_id in range(args.num_clients):

            local_models[client_id].eval()
            logits = local_models[client_id].forward(subgraphs[client_id])
            loss_train = loss_fn(logits[subgraphs[client_id].train_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].train_idx])
            loss_val = loss_fn(logits[subgraphs[client_id].val_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].val_idx])
            loss_test = loss_fn(logits[subgraphs[client_id].test_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].test_idx])
            acc_train = accuracy(logits[subgraphs[client_id].train_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].train_idx])
            acc_val = accuracy(logits[subgraphs[client_id].val_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].val_idx])
            acc_test = accuracy(logits[subgraphs[client_id].test_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].test_idx])
            
            print(f"[client {client_id}]: acc_train: {acc_train:.2f}\tacc_val: {acc_val:.2f}\tacc_test: {acc_test:.2f}\tloss_train: {loss_train:.4f}\tloss_val: {loss_val:.4f}\tloss_test: {loss_test:.4f}")
            global_acc_val += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_val
            global_acc_test += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_test
            
        print(f"[server]: current_round: {round_id}\tglobal_val: {global_acc_val:.2f}\tglobal_test: {global_acc_test:.2f}")
        
        if global_acc_val > best_server_val:
            best_server_val = global_acc_val
            best_server_test = global_acc_test
            best_round = round_id
        print(f"[server]: best_round: {best_round}\tbest_val: {best_server_val:.2f}\tbest_test: {best_server_test:.2f}")
        print("-"*50)
        
        
        l_glb_acc_test.append(global_acc_test)
        
        
        # local train
        for client_id in range(args.num_clients):
            for epoch_id in range(args.num_epochs):
                local_models[client_id].train()
                local_optimizers[client_id].zero_grad()

                if args.use_contrastive:
                    logits, embeddings = local_models[client_id].forward(
                        subgraphs[client_id], return_embedding=True)
                else:
                    logits = local_models[client_id].forward(subgraphs[client_id])

                train_logits = logits[subgraphs[client_id].train_idx]
                train_labels = subgraphs[client_id].y[subgraphs[client_id].train_idx]

                # Weighted / standard CrossEntropy loss
                if args.use_weighted_ce:
                    ce_loss = F.cross_entropy(train_logits, train_labels,
                                              weight=class_weights[client_id])
                else:
                    ce_loss = loss_fn(train_logits, train_labels)

                loss = ce_loss

                # Supervised Contrastive loss on train-node embeddings
                if args.use_contrastive:
                    train_embeddings = embeddings[subgraphs[client_id].train_idx]
                    cl_loss = supervised_contrastive_loss(
                        train_embeddings, train_labels, tau=args.cl_tau)
                    loss = loss + args.lambda_cl * cl_loss

                loss_train = loss
                loss_train.backward()
                local_optimizers[client_id].step()
                
        # global aggregation
        with torch.no_grad():
            for client_id in range(args.num_clients):
                weight = subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] 
                for (local_state, global_state) in zip(local_models[client_id].parameters(), global_model.parameters()):
                    if client_id == 0:
                        global_state.data = weight * local_state
                    else:
                        global_state.data += weight * local_state
        
        
        num_gen = args.num_gen
        c_cnt = [0] * dataset.num_classes
        for class_i in range(dataset.num_classes):
            c_cnt[class_i] = int(num_gen * 1 / dataset.num_classes)
        c_cnt[-1] += num_gen - sum(c_cnt)

        print(f"pseudo label distribution: {c_cnt}")
        c = torch.zeros(num_gen).to(device).long()
        ptr = 0
        for class_i in range(dataset.num_classes):
            for _ in range(c_cnt[class_i]):
                c[ptr] = class_i
                ptr += 1
                
                
        each_class_idx = {}
        for class_i in range(dataset.num_classes):
            each_class_idx[class_i] = c == class_i
            each_class_idx[class_i] = each_class_idx[class_i].to(device)


        
        for client_id in range(args.num_clients):
            local_models[client_id].eval()
        
        
        for _ in range(args.glb_epochs):
            
            ############ sampling noise ##############
            z = torch.randn((num_gen, 32)).to(device)
            
            
            ############ train generator ##############
            generator.train()
            global_model.eval()
            for it_g in range(args.it_g):
                loss_sem = 0
                loss_diverg = 0
                loss_div = 0
                
                
                generator_optimizer.zero_grad()
                for client_id in range(args.num_clients):
                    ######  generator forward  ########
                    if args.generator_type == 'diffusion':
                        node_logits = generator.sample(
                            labels=c, num_steps=args.diffusion_steps, device=device)
                    else:
                        node_logits = generator.forward(z=z, c=c)
                    node_norm = F.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(
                        node_logits, adj_logits, k=args.topk)
                    
                    ##### local & global model -> forward #########
                    if args.fedtad_mode == 'rep_distill':
                        local_pred = local_models[client_id].rep_forward(
                            data=pseudo_graph)
                        global_pred = global_model.rep_forward(
                            data=pseudo_graph)
                    else:
                        local_pred = local_models[client_id].forward(
                            data=pseudo_graph)
                        global_pred = global_model.forward(
                            data=pseudo_graph)  
                        
                        
                    ##########  semantic loss  #############
                    acc_list = [0] * dataset.num_classes
                    for class_i in range(dataset.num_classes):
                        loss_sem += normalized_ckr[client_id][class_i] * nn.CrossEntropyLoss()(local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
                        acc = accuracy(local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
                        acc_list[class_i] = f"{acc:.2f}"
                    acc_tot = float(accuracy(local_pred, c))
                    # print(f"[client {client_id}] accuracy on each class for pseudo_graph: {acc_list}, on all classes: {acc_tot:.2f}")


                    ############  diversity loss  ##############
                    if args.generator_type == 'diffusion':
                        z_for_div = torch.randn_like(node_logits)
                        loss_div += DiversityLoss(metric='l1').to(device)(
                            z_for_div, node_logits)
                    else:
                        loss_div += DiversityLoss(metric='l1').to(device)(
                            z.view(z.shape[0],-1), node_logits)
                
                
                    ############  divergence loss  ############   
                    for class_i in range(dataset.num_classes):
                        loss_diverg += - normalized_ckr[client_id][class_i] * torch.mean(torch.mean(
                            torch.abs(global_pred[each_class_idx[class_i]] - local_pred[each_class_idx[class_i]].detach()), dim=1))


                ############ generator loss #############
                loss_G = args.lam1 * loss_sem + loss_diverg + args.lam2 * loss_div              
                # print(f'[generator] loss_sem: {loss_sem:.4f}\tloss_div: {loss_div:.4f}\tloss_diverg: {loss_diverg:.4f}\tloss_G: {loss_G:.4f}')
                
                loss_G.backward()
                generator_optimizer.step()
                    
                    
                    
                
            ########### train global model ###########
            generator.eval()
            global_model.train()
            for it_d in range(args.it_d):
                global_optimizer.zero_grad()
                loss_D = 0
                
                for client_id in range(args.num_clients):
                    ######  generator forward  ########
                    if args.generator_type == 'diffusion':
                        with torch.no_grad():
                            node_logits = generator.sample(
                                labels=c, num_steps=args.diffusion_steps, device=device)
                    else:
                        node_logits = generator.forward(z=z, c=c)
                    node_norm = F.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(node_logits.detach(), adj_logits.detach(), k=args.topk)
                        
                    #######  local & global model -> forward  #######
                    if args.fedtad_mode == 'rep_distill':
                        local_pred = local_models[client_id].rep_forward(
                            data=pseudo_graph)
                        global_pred = global_model.rep_forward(
                            data=pseudo_graph)
                    else:
                        local_pred = local_models[client_id].forward(
                            data=pseudo_graph)
                        global_pred = global_model.forward(
                            data=pseudo_graph)  
                    
                    ############  divergence loss  ############   
                    
                    for class_i in range(dataset.num_classes):
                        loss_D += normalized_ckr[client_id][class_i] * torch.mean(torch.mean(
                            torch.abs(global_pred[each_class_idx[class_i]] - local_pred[each_class_idx[class_i]].detach()), dim=1))

                
                
                # print(f"[global] loss_diverg: {loss_D:.4f}")
                loss_D.backward()
                global_optimizer.step()

