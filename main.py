from MADDPG import MADDPG
from utils import *
import IMGDDPG

np.random.seed(1234)
n_agents = 1
n_states = 8898
n_actions = 6
capacity = 10000000
batch_size = 100

n_episode = 1
max_steps = 20
episodes_before_train = 1

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                    episodes_before_train)
CHANGE=5
CHANGE_epoch=2
SPACE=False

log_path='./log_RL_all'

def init_state(path,video_info,w_vec,globle_vec):
    #Initialization L is [0.25, 0.75]
    location=th.tensor([0.25,0.75])
    patha=path+video_info
    start=int(int(video_info.split('-')[2])*0.25)
    end=int(int(video_info.split('-')[2])*0.75)

    local_vec, _= IMGDDPG.image_state_one(path, video_info, w_vec.cuda(), [start,end])
    local_vec=th.tensor(local_vec)
    state = th.cat((w_vec, local_vec,globle_vec.float(),location), 0).unsqueeze(0).cuda()
    return state

def state_update(w_start,w_end,state,action,unit,step,fi):
    for n in range(n_agents):
        location=state[n][-2:]
        old=location.cpu().numpy()
        flag=np.argmax(action[n])

        old_iou = calculate_IoU(location, [w_start, w_end])
        if flag==0:
            old[0]=location[0]-unit
            old[1] = location[1] - unit
        elif flag==1:
            old[0] = location[0] + unit
            old[1] = location[1] + unit
        elif flag==2:
            old[0] = location[0] - unit
        elif flag==3:
            old[0] = location[0] + unit
        elif flag==4:
            old[1] = location[1] - unit
        elif flag==5:
            old[1] = location[1] + unit
        else:
            stop=True
            return state,[fi],old_iou

        if old[0]<0 or old[1]>1 or old[0]>=old[1]:
            stop=True
            return state,[fi],old_iou

        state[n][-2] = th.tensor(old[0]).cuda()
        state[n][-1] = th.tensor(old[1]).cuda()
        state_new = state

        now_iou=calculate_IoU(old,[w_start,w_end])
        reward=[calculate_reward(old_iou,now_iou,step)]

    return state_new,reward,now_iou

def state_update_img(path,video_info,state,w_start,w_end):
    f_all = video_info.split('-')[2]
    image_feature,action = IMGDDPG.image_state_one(path, video_info, state[0][0:4800],
                                        [int(w_start * int(f_all)), int(w_end * int(f_all))])

    state[0][4800:6848] = th.squeeze(th.tensor(image_feature)).cuda()
    state_new = state

    return state_new,action




def train():
    print('###############TRAIN##################')
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

    rr = np.zeros((n_agents,))
    loader=dataloader('./feature_all_train.npy')
    #Workspace
    cut_path='./data/video_cut2image/'
    dataload = th.utils.data.DataLoader(dataset=loader,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    n=len(dataload)
    for batch_idx, all_f in enumerate(dataload):
        w_start = float(all_f[:][3][0])
        w_vec = all_f[:][1][0]
        w_end = float(all_f[:][4][0])
        v_vec = all_f[:][2][0]
        video_info = all_f[:][0][0]
        fps = all_f[:][5][0]

        print(n,batch_idx, video_info)
        start_f=(int(w_start*fps.data)-int(video_info.split('-')[1]))/int(video_info.split('-')[2])
        end_f=(int(w_end*fps.data)-int(video_info.split('-')[1]))/int(video_info.split('-')[2])
        IOU_recode = list()
        for i_episode in range(n_episode):
            reward_record = []
            total_steps = 0
            total_reward = 0.0
            quick=0.05 #moving step
            fi=-1 #abnormal penalty

            state=init_state(cut_path,video_info,th.squeeze(th.squeeze(w_vec,0)),v_vec)
            t=0
            while True:
                action = maddpg.select_action(state).data.cpu()
                state_, reward, iou_now = state_update(start_f,end_f,state,action.numpy(),quick,total_steps,fi)
                #Joint training
                if SPACE==True and t%CHANGE==CHANGE-1:
                    for i in range(CHANGE_epoch):
                        img_state,img_action=state_update_img(cut_path,video_info,state,start_f,end_f)
                        action_img = maddpg.select_action(img_state).data.cpu()
                        Q_P=maddpg.get_q(state,th.tensor(action).cuda())
                        Q_N=maddpg.get_q(img_state,th.tensor(action_img).cuda())
                        r=calculate_reward_img(Q_P[0],Q_N[0])

                        IMGDDPG.ddpg_image.memory.push(img_state, img_action, None, r)
                        IMGDDPG.ddpg_image.update_policy()
                    state_=img_state
                reward = th.FloatTensor(reward).type(FloatTensor)

                if t != max_steps - 1:
                    state_ = state_
                else:
                    state_ = None
                    IOU_recode.append(iou_now)
                    break

                total_reward += reward.sum()
                rr += reward.cpu().numpy()
                maddpg.memory.push(state.data, action, state_, reward)
                state = state_
                c_loss, a_loss = maddpg.update_policy()

                t=t+1
                total_steps=total_steps+1

            maddpg.episode_done += 1
            reward_record.append(total_reward)

        iou_recode=th.mean(th.tensor(IOU_recode))
        print(batch_idx,iou_recode,'over--------------------')
        info='\n' + str(batch_idx) + ' '+video_info+' '+str(iou_recode)
        log(log_path,info)

def test():
    print('###############TEST##################')
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

    rr = np.zeros((n_agents,))
    loader = dataloader('./feature_all_test.npy')
    #Workspace
    cut_path = './data/test/'
    dataload = th.utils.data.DataLoader(dataset=loader,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=4)
    iou_all=list()
    for batch_idx, all_f in enumerate(dataload):
        w_start = float(all_f[:][3][0])
        w_vec = all_f[:][1][0]
        w_end = float(all_f[:][4][0])
        v_vec = all_f[:][2][0]
        video_info = all_f[:][0][0]
        fps = all_f[:][5][0]

        print(batch_idx,video_info)
        start_f=(int(w_start*fps.data)-int(video_info.split('-')[1]))/int(video_info.split('-')[2])
        end_f=(int(w_end*fps.data)-int(video_info.split('-')[1]))/int(video_info.split('-')[2])
        IOU_recode = list()
        for i_episode in range(n_episode):
            reward_record = []
            total_steps = 0
            total_reward = 0.0
            quick=0.05
            fi=-1

            state=init_state(cut_path,video_info,w_vec,v_vec)
            t=0
            while True:
                action = maddpg.select_action(state).data.cpu()
                state_, reward, iou_now = state_update(start_f,end_f,state,action.numpy(),quick,total_steps,fi)

                if SPACE==True and t%CHANGE==CHANGE-1:
                    for i in range(CHANGE_epoch):
                        img_state,_=state_update_img(cut_path,video_info,state,start_f,end_f)
                    state_=img_state
                reward = th.FloatTensor(reward).type(FloatTensor)

                if t != max_steps - 1:
                    state_ = state_
                else:
                    state_ = None
                    IOU_recode.append(iou_now)
                    break

                total_reward += reward.sum()
                rr += reward.cpu().numpy()
                state = state_
                
                t=t+1
                total_steps=total_steps+1

            maddpg.episode_done += 1
            reward_record.append(total_reward)

        iou_recode=th.mean(th.tensor(IOU_recode))
        print(batch_idx,iou_recode,'over--------------------')
        info='\n' + str(batch_idx) + ' ' + video_info + ' ' + str(iou_recode)
        log(log_path, info)
        iou_all.append(iou_recode)

    IoU_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for k in range(len(IoU_thresh)):
        IoU = IoU_thresh[k]
        acc = compute_IoU_recall_top_n(1, IoU, iou_all)
        print('ACC@%.1f:  %f'%(IoU,acc))
        info_c='\n----------------------' + str(IoU) + ' ' +  str(acc)
        log(log_path,info_c)



if __name__ == '__main__':
    pre_epoch=2
    epoch=100
    
    for i in range(pre_epoch): 
        train()
        test()
        info='\n#############Pre##############'
        log(log_path,info)
    
    SPACE=True
    for i in range(epoch):
        train()
        test()
        info='\n#############Ture##############'
        log(log_path,info)

