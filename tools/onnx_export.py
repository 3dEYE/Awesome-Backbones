from argparse import ArgumentParser
import os
import sys
sys.path.insert(0,os.getcwd())
import torch
import onnx

from utils.inference import inference_model, init_model, show_result_pyplot
from utils.train_utils import get_info, file2dict
from models.build import BuildNet

def main():
    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--classes-map', default='datas/annotations.txt', help='classes map of datasets')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--save-path',
        help='The path to save prediction image, default not to save.')
    args = parser.parse_args()

    classes_names, label_names = get_info(args.classes_map)
    # build the model from a config file and a checkpoint file
    model_cfg, train_pipeline, val_pipeline,data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')
    
    # test a single image
    #result = inference_model(model, args.img, val_pipeline, classes_names,label_names)
    dummy_input = torch.zeros(1,3,240,240).to(device=device)
    
    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "model.onnx",       # where to save the model  
         export_params=True,
         opset_version=16,    # the ONNX version to export the model to 
         do_constant_folding=True,
         input_names = ['input'],   # the model's input names 
         output_names = ['output']) 
    print(" ") 
    print('Model has been converted to ONNX') 
    
    model = onnx.load("model.onnx")
    onnx.checker.check_model(model)
    
    import onnxsim

    print('\nStarting to simplify ONNX...')
    model, check = onnxsim.simplify(model)
    onnx.save(model,"model.onnx")


if __name__ == '__main__':
    main()
