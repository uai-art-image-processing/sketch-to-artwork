import os
from PIL import Image

def checkImg(path):
    img = Image.open(path)
    img.verify() #I perform also verify, don't know if he sees other types o defects
    img.close() #reload is necessary in my case
    img = Image.open(path) 
    img.transpose(Image.FLIP_LEFT_RIGHT)
    img.close()

def checkDir(path, verbose=False):
    source = os.path.join(os.getcwd, path)
    for dirpath, _, filenames in os.walk(source):
            for filename in filenames:
                fullpath = os.path.join(dirpath, filename)
                if verbose: print(f"checking {filename}", end="...")
                try:
                    checkImg(fullpath)
                except IOError:
                    print("failed")
                    raise Exception(f"{filename} is truncated")
                except:
                    print("failed")
                    raise Exception("Something went wrong")
                print("ok")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image")
    parser.add_argument("-d", "--directory")
    parser.add_argument("-v", "--verbose")
    args = parser.parse_args()

    if args.image is not None: checkImg(args.image, args.verbose)
    if args.directory is not None:checkDir(args.directory, args.verbose)
    