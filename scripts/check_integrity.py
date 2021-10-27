import os
from PIL import Image

def verifyImg(path, verbose=False):
    if verbose: print(f"checking {path}", end="...")
    try:
        img = Image.open(path)
        img.verify() #I perform also verify, don't know if he sees other types o defects
        img.close() #reload is necessary in my case
        img = Image.open(path) 
        img.transpose(Image.FLIP_LEFT_RIGHT)
        img.close()
        if verbose: print("ok")
        return True
    except:
        if verbose: print("failed")
        return False

def verifyDir(path, verbose=False):
    source = os.path.join(os.getcwd, path)
    broken = []
    
    print("cheking directory recursively...",)
    for dirpath, _, filenames in os.walk(source):
        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)
            if not verifyImg(fullpath, verbose): broken.append(filename)
    print("Done")
                    
    if broken != []: print(f"The following images can't be read by PIL:\n{broken}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image")
    parser.add_argument("-d", "--directory")
    parser.add_argument("-v", "--verbose")
    args = parser.parse_args()

    if args.image is not None: verifyImg(args.image, args.verbose)
    if args.directory is not None: verifyDir(args.directory, args.verbose)
    