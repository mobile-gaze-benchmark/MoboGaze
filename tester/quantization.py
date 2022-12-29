import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fx.graph_module import ObservedGraphModule
from torch.quantization import get_default_qconfig

def main(train, test):

    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    # torch.cuda.set_device(test.device)

    data = test.data
    load = test.load
 

    # ===============================> Read Data <=========================
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 

    print(f"==> Test: {data.label} <==")
    dataset = reader.loader(data, 32, num_workers=4, shuffle=False)

    modelpath = os.path.join(train.save.metapath,
                                train.save.folder, f"checkpoint/")
    
    logpath = os.path.join(train.save.metapath,
                                train.save.folder, f"{test.savename}")

  
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Test <=============================

    begin = load.begin_step; end = load.end_step; step = load.steps

    for saveiter in range(begin, end+step, step):

        print(f"Test {saveiter}")

        net = model.Model()

        # statedict = torch.load(
        #                 os.path.join(modelpath, 
        #                     f"Iter_{saveiter}_{train.save.model_name}.pt"), 
        #                 map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
        #             )

        device = torch.device("cpu")
        net.cpu()
        # net.load_state_dict(statedict); 
        net.eval()

        qconfig = get_default_qconfig("qnnack")  # 默认是静态量化
        qconfig_dict = {
            "": qconfig,
            # 'object_type': []
        }
        model_to_quantize = copy.deepcopy(net)

        # feature = {"head_pose": torch.zeros(1, 2).to(device),
        #            "left": torch.zeros(1, 3, 448, 448).to(device),
        #            "right": torch.zeros(1, 3, 448, 448).to(device)
                  #  }
        feature = edict()
        feature.face = torch.zeros(1, 3, 448, 448).to(device)

        prepared_model = prepare_fx(model_to_quantize, qconfig_dict, feature)
        prepared_model.cpu()
        # # print("prepared model: ", prepared_model)
        # print("Calibrating")
        # with torch.inference_mode():
        #   for j, (data, label) in enumerate(dataset):
        #       # print(inputs)
        #       for key in data:
        #             if key != 'name': data[key] = data[key].cpu()
        #       # inputs[].to(device)
        #       prepared_model(data)
        #       # print("already prepared")
        # print("calib done.")

        quantized_model = convert_fx(prepared_model)
        quantized_model.cpu()
        print("check")

        length = len(dataset); accs = 0; count = 0

        logname = f"{saveiter}.log"

        outfile = open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")
        

        with torch.no_grad():
            for j, (data, label) in enumerate(dataset):

                for key in data:
                    if key != 'name': data[key] = data[key].cpu()

                names =  data["name"]
                gts = label.cpu()
           
                gazes = quantized_model(data)

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]

                    count += 1                
                    accs += gtools.angular(
                                gtools.gazeto3d(gaze),
                                gtools.gazeto3d(gt)
                            )
            
                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    log = name + [",".join(gaze)] + [",".join(gt)]
                    outfile.write(" ".join(log) + "\n")
                    loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
                    outfile.write(loger)
                    print(loger)

            loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
            outfile.write(loger)
            print(loger)
        outfile.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    # print("=======================>(Begin) Config of training<======================")
    # print(ctools.DictDumps(train_conf))
    # print("=======================>(End) Config of training<======================")
    # print("")
    # print("=======================>(Begin) Config for test<======================")
    # print(ctools.DictDumps(test_conf))
    # print("=======================>(End) Config for test<======================")

    main(train_conf.train, test_conf.test)

 
