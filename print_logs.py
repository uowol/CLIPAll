import argparse
import os 

def main(args):
    if args.type == 'fewshot':
        path = args.dir
        path += args.dataset + '/'
        path += args.trainer + '/'
        if 'ZeroshotCLIP' in args.trainer:
            path += args.cfg + '/'
            for seed in os.listdir(path):
                print("="*20+" "+path[len(args.dir):]+str(seed)+" "+"="*20)
                for file_name in os.listdir(path+seed):
                    if 'log' in file_name:             
                        print(f">>> print '{file_name}'\n")   
                        with open(path+seed+'/'+file_name, 'r') as f:
                            last_line = f.readlines()[-7:]
                        print("".join(last_line), end='')
        else:
            path += args.cfg+f"_{args.num_shots}shots" + '/'
            for seed in os.listdir(path):
                print("="*20+" "+path[len(args.dir):]+str(seed)+" "+"="*20)
                for file_name in os.listdir(path+seed):
                    if 'log' in file_name:             
                        print(f">>> print '{file_name}'\n")   
                        with open(path+seed+'/'+file_name, 'r') as f:
                            last_line = f.readlines()[-8:]
                        print("".join(last_line), end='')
    elif args.type == 'base2novel':
        path = args.dir
        print(">> base <<")
        path += 'base2new' + '/'
        path1 = path + 'train_base' + '/'
        path1 += args.dataset + f'/shots_{args.num_shots}/'
        path1 += args.trainer + '/'
        path1 += args.cfg + '/'
        for seed in os.listdir(path1):
            print("="*20+" "+path1[len(args.dir):]+str(seed)+" "+"="*20)
            for file_name in os.listdir(path1+seed):
                if 'log' in file_name:             
                    print(f">>> print '{file_name}'\n")   
                    with open(path1+seed+'/'+file_name, 'r') as f:
                        last_line = f.readlines()[-8:]
                    print("".join(last_line), end='')
        print(">> new <<")
        path2 = path + 'test_new' + '/'
        path2 += args.dataset + f'/shots_{args.num_shots}/'
        path2 += args.trainer + '/'
        path2 += args.cfg + '/'
        for seed in os.listdir(path2):
            print("="*20+" "+path2[len(args.dir):]+str(seed)+" "+"="*20)
            for file_name in os.listdir(path2+seed):
                if 'log' in file_name:             
                    print(f">>> print '{file_name}'\n")   
                    with open(path2+seed+'/'+file_name, 'r') as f:
                        last_line = f.readlines()[-8:]
                    print("".join(last_line), end='')
    elif args.type == 'crossdataset':
        path = args.dir
        path += 'evaluation' + '/'
        path += args.trainer + '/'
        if 'ZeroshotCLIP' in args.trainer:
            path += args.cfg + '/'
        else:
            path += args.cfg+f"_{args.num_shots}shots" + '/'
        path += args.dataset + '/'
        for seed in os.listdir(path):
            print("="*20+" "+path[len(args.dir):]+str(seed)+" "+"="*20)
            for file_name in os.listdir(path+seed):
                if 'log' in file_name:             
                    print(f">>> print '{file_name}'\n")   
                    with open(path+seed+'/'+file_name, 'r') as f:
                        last_line = f.readlines()[-8:]
                    print("".join(last_line), end='')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="", help="path to output")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--trainer", type=str, default="ZeroshotCLIP")
    parser.add_argument("--cfg", type=str, default="vit_b16")
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--type", type=str, default="fewshot") # fewshot, base2novel, crossdataset
    args = parser.parse_args()
    main(args)
