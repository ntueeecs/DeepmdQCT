import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model.deepmdqct import DeepmdQCT
import transforms
from torchvision.utils import save_image
import pandas as pd
plt.switch_backend('agg')


def predict_single_person():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    flip_test = True
    resize_hw = (512, 512)
    weights_path = "modelnewnew-500.pth"
    # model path:https://1drv.ms/u/s!Auy8I-BCHDaJgRu5EgZjYlRf4PLY?e=46cLuw
    keypoint_json_path = "person_keypoints.json"

    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    #todo Data transformation
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read model
    model = DeepmdQCT(base_channel=32,num_joints=1)
    model=torch.nn.DataParallel(model)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    # if isinstance(model, torch.nn.DataParallel):  #Determine whether to use single card prediction for multi card training
    #     model = model.module
    #     model.load_state_dict(weights)
    model.to(device)
    model.eval()
    data= pd.read_excel("class.xlsx")
    data_true=data["label"].tolist()
    qct_data=data["qct"].tolist()
    s=0
    tr=[]
    pr=[]
    tr_qct=[]
    pr_qct=[]

    #todo example
    name = 1 #image name
    img_path1 = f"DATA/images1/{name}.png"
    img_path2 = f"DATA/images2/{name}.png"
    assert os.path.exists(img_path1), f"file: {img_path1} does not exist."
    assert os.path.exists(img_path2), f"file: {img_path1} does not exist."
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)   #BGR-->RGB
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)   #BGR-->RGB

    img_tensor1, target1 = data_transform(img1, {"box": [0, 0, img1.shape[1] - 1, img1.shape[0] - 1]})
    img_tensor2, target2 = data_transform(img2, {"box": [0, 0, img2.shape[1] - 1, img2.shape[0] - 1]})

    img_tensor=torch.cat([img_tensor1,img_tensor2],dim=0)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    clinic=data.iloc[:,2:-1].values.tolist()   #todo Clinical data
    ct_clinic_data=torch.tensor(clinic[int(name)-1]).view(1,12).to(device)

    name1 = ["osteopenia", "normal", "osteoporosis"]
    with torch.no_grad():
        alpha=0
        outputs,result_segement1,result_segement2,result_class,domain_output,qct =model(img_tensor.to(device),target1,ct_clinic_data,alpha)
        #show result
        print("list",name,name1[torch.max(result_class,dim=1)[1].cpu().item()],"true",name1[int(data_true[name-1])])
        print("predict qct",qct.cpu().numpy()[0][0],"tre qct",qct_data[name-1])
        tr.append(name1[int(data_true[name-1])])
        pr.append(name1[torch.max(result_class,dim=1)[1].cpu().item()])
        tr_qct.append(qct_data[name-1])
        pr_qct.append(qct.cpu().numpy()[0][0])
        if data_true[name-1]==name1[torch.max(result_class,dim=1)[1].cpu().item()]:
            s+=1
        save_image(result_segement1.cpu(),f"segement_L1.png")
        save_image(result_segement2.cpu(),f"segement_L2.png")
        # img_ju = cv2.imread("segement_L1.png")
        # img_result_ori=np.where(img_ju>0,img1,0)
        img_result_L1=torch.where(result_segement1.cpu()*255>1,img_tensor1,result_segement1.cpu())
        save_image(img_result_L1,f"roi_result_L1.png")
        img_result_L2=torch.where(result_segement2.cpu()*255>1,img_tensor1,result_segement1.cpu())
        save_image(img_result_L2,f"roi_result_L2.png")

        cv2.imshow("L1",cv2.imread(f"segement_L1.png"))
        cv2.imshow("L2",cv2.imread(f"segement_L2.png"))
        cv2.waitKey(100)

        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device),target1,ct_clinic_data,alpha)[0], person_info["flip_pairs"]),
            )
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5
        #Calculate key point information
        keypoints, scores = transforms.get_final_preds(outputs, [target1["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        # scores = np.squeeze(scores)
        print("keypoints",keypoints)
        # plot_img = draw_keypoints(img1, keypoints, scores, thresh=0.01, r=5)
        # plot_img.save("keypoint.jpg")
    tr=pd.DataFrame(tr)
    pr=pd.DataFrame(pr)
    tr_qct=pd.DataFrame(tr_qct)
    pr_qct=pd.DataFrame(pr_qct)
    result=pd.concat([tr,pr,tr_qct,pr_qct],1)
    result.columns=["tre_cla","pre_cla","qct_tr","qct_pr"]
    print(result)
    result.to_excel("predict.xlsx")


if __name__ == '__main__':
    predict_single_person()
