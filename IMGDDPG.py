''' Spatial-level DDPG'''

from MADDPG import MADDPG
from utils import *
import spp

np.random.seed(1234)
th.manual_seed(1234)
n_agents_image = 1
n_states_image = 8896
n_actions_image = 8
capacity_image = 10000000
batch_size_image = 100

episodes_before_train_image = 1
ddpg_image=MADDPG(n_agents_image, n_states_image, n_actions_image, batch_size_image, capacity_image,
                    episodes_before_train_image)

def img_update(x,y,w,h,action,unit):
    flag = np.argmax(action[0])
    oldx,oldy,oldw,oldh = x,y,w,h

    if flag == 0:
        x = x + unit #left
    elif flag == 1:
        x = x - unit #right
    elif flag == 2:
        y = y + unit #up   
    elif flag == 3:
        y = y - unit #down  
    elif flag == 4:
        w = w + unit
        h = h + unit #bigger 
    elif flag == 5:
        w = w - unit
        h = h - unit #smaller  
    elif flag == 6:
        h = h + unit #taller  
    elif flag == 7:
        w = w + unit #fatter
    else:
        stop = True  #stop
        return x,y,w,h

    #abnormal
    if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w+x > 1 or h < 0 or h+y > 1:
        stop = True
        return oldx,oldy,oldw,oldh

    return x,y,w,h

def image_state(path,video_info,w_vec,location):
    path = path + video_info+'/'
    start=location[0]
    end=location[1]
    l_v_vec=th.tensor(resnet(path,start,end)).float().cuda()

    x,y,w,h=0.5，0.5，0.8，0.8
    quick = 0.01  # 1
    spp_state_all=list()
    old=0
    spp_state_old=None
    for i in range(start,end):
        img_path=path+str(i)+'.jpg'
        spp = spp.SppNet(w_vec, l_v_vec)
        spp_state = th.tensor(extractor_spp(img_path, spp, True, x, y, w, h)).unsqueeze(0).cuda()
        spp_state_all.append(th.squeeze(spp_state).cpu().numpy())

        #update tracking box
        spp_state_train=th.unsqueeze(th.cat((th.squeeze(spp_state),w_vec,l_v_vec),0),0)
        action = ddpg_image.select_action(spp_state_train).data.cpu()
        x, y, w, h = img_update(x, y, w, h, action, quick)

        if old==0:
            spp_state_old=spp_state_train
        else:
            ddpg_image.memory.push(spp_state_old, action, spp_state_train, 0)
            spp_state_old=spp_state_train

    #feature_all=attention(spp_state_all)
    feature_all=np.squeeze(np.array(spp_state_all).mean(axis=0))

    return feature_all,action

#test_state_one
def image_state_one(path,video_info,w_vec,location):
    path = path + video_info+'/'
    start=location[0]
    end=location[1]
    l_v_vec=th.tensor(resnet(path,start,end)).float().cuda()

    x,y,w,h=0.5,0.5,0.8,0.8
    quick = 0.01  # 1
    spp_state_all=list()
    old=0
    spp_state_old=None
    for i in range(start,end):
        img_path=path+str(i)+'.jpg'
        spp = spp.SppNet(w_vec, l_v_vec)
        spp_state = th.tensor(extractor_spp(img_path, spp, True, x, y, w, h)).unsqueeze(0).cuda()
        spp_state_all.append(th.squeeze(spp_state).cpu().numpy())

    feature_all = th.tensor(np.squeeze(np.array(spp_state_all).mean(axis=0))).cuda()
    spp_state_train=th.unsqueeze(th.cat((th.squeeze(feature_all),w_vec,l_v_vec),0),0)
    action = ddpg_image.select_action(spp_state_train).data.cpu()

    return feature_all.cpu(),action

