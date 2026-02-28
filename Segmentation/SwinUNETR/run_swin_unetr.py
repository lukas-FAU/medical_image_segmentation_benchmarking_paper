import swin_unetr
import sys


NUM_EPOCHS = 1

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("ERROR! Usage: run.py DATA_ROOT SAVE_PATH")
        exit(1)
    path_root = sys.argv[1]
    path_save = sys.argv[2]

    swin_unetr.train(path_root, path_save, NUM_EPOCHS)

    exit()