import argparse
import os
import sys
import shutil

from datetime import datetime, timedelta

def _log(text):
    now = datetime.now()
    now = now + timedelta(hours=9)
    print(f"({now.strftime('%Y-%m-%d %H:%M:%S')})\tLOG:\t",f"start {text}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--sample_num', type=int, default=10)

    args = parser.parse_args()
    
    input_path = "/data4/kchanwo/clipall/datasets/"
    output_path = args.output_dir+'export/'

    for dataset in os.listdir(input_path):
        domains = os.listdir(input_path+dataset)
        for domain in domains:
            if '.' in domain: continue;
            classes = os.listdir(input_path+dataset+'/'+domain)
            for _class in classes:
                os.makedirs(output_path+dataset+'/'+domain+'/'+_class, exist_ok=True)
                files = os.listdir(input_path+dataset+'/'+domain+'/'+_class)
                images = [file for file in files if file.endswith('.jpg') or file.endswith('.png')]
                for i in range(args.sample_num):
                    if i >= len(images):
                        print("# of images in '"+input_path+dataset+'/'+domain+'/'+_class+"': "+str(len(images))) 
                        break
                    from_path = input_path+dataset+'/'+domain+'/'+_class+'/'+images[i]
                    to_path = output_path+dataset+'/'+domain+'/'+_class+'/'+images[i]
                    shutil.copyfile(from_path, to_path)
