import os
import torch

def save_res(res,file_path):
        line = str(res) + "\n"

        with open(os.path.join(file_path),"a") as outfile:
            outfile.write(line)

def save_model(model, model_dir, model_name):
    torch.save(model, model_dir+'/'+ (model_name + '.pt'))