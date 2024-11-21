import torch
import numpy as np
import torch.nn.functional as F

def cluster (train_loader_cluster,model,cluster_number,args):
    model.eval()
    features_sum = []
    features_by_class = {0: [], 1: []}
    for i, (input,target,M,hard_y,cluster_target,indexs) in enumerate(train_loader_cluster):
        input = input.cuda()
        target =target.cuda()

        with torch.no_grad():
            pred, features,cam  = model(input)
            features = features.detach()
            features_sum.append(features)
        for feature, label in zip(features, target):
            features_by_class[int(label.item())].append(feature)
    for label in range(2):
        if features_by_class[label]:
            features_by_class[label] = torch.stack(features_by_class[label], dim=0)


    features_list = [features_by_class[label] for label in range(2) if features_by_class[label].nelement() != 0]
    features_combined = torch.cat(features_list, dim=0) 

    features = torch.split(features_combined, args.cls_num_list, dim=0)

    if args.train_rule == 'Rank':
         feature_center = [torch.mean(t, dim=0) for t in features]
         feature_center = torch.cat(feature_center,axis = 0)
         feature_center=feature_center.reshape(args.num_classes,args.feat_dim)
         density = np.zeros(len(cluster_number))
         for i in range(len(cluster_number)):
            center_distance = F.pairwise_distance(features[i], feature_center[i], p=2).mean()/np.log(len(features[i])+10)
            density[i] = center_distance.cpu().numpy()
         density = density.clip(np.percentile(density,20),np.percentile(density,80))
         density = args.temperature*(density/density.mean())
         for index, value in enumerate(cluster_number):
            if value==1:
                density[index] = args.temperature
    target = [[] for i in range(len(cluster_number))]
    for i in range(len(cluster_number)):
        if cluster_number[i] >1:
            cluster_ids_x, _ = kmeans(X=features[i], num_clusters=cluster_number[i], distance='cosine', tol=1e-3, iter_limit=35, device=torch.device("cuda"))
            #run faster for cluster
            target[i]=cluster_ids_x
        else:
            target[i] = torch.zeros(1,features[i].size()[0], dtype=torch.int).squeeze(0)
    cluster_number_sum=[sum(cluster_number[:i]) for i in range(len(cluster_number))]
    for i ,k in enumerate(cluster_number_sum):
         target[i] =  torch.add(target[i], k)
    targets=torch.cat(target,dim=0)
    targets = targets.numpy().tolist()
    if args.train_rule == 'Rank':
        return targets,density
    else:
        return targets