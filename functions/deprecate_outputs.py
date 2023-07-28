import os 
import argparse
import shutil

except_keywords = ['shots']

def main(args):
    path = args.dir
    os.mkdir(path+'!deprecated')
    for folder in os.listdir(path):
        if folder == '!deprecated': continue
        
        is_alive = False
        for keyword in except_keywords:
            if keyword in folder: 
                is_alive = True; break
        if is_alive: continue
        filename = path+folder
        deprecated_folder = path+'!deprecated/'+folder
        shutil.move(filename, deprecated_folder)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="", help="output folder's path")
    args = parser.parse_args()
    main(args)
