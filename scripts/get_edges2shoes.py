import requests
import tarfile

def getFile(url, target_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            try:
                f.write(response.raw.read())
            except:
                f.close()
                raise Exception("Download failed")
            f.close()

def extarctFile(target_path):
    tar = tarfile.open(target_path, "r:gz")
    tar.extractall()
    tar.close()

def main():
    url = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz'
    target_path = '../data/edges2shoes.tar.gz'

    getFile(url, target_path)
    extarctFile(target_path)

if __name__ == "__main__":
    main()

    