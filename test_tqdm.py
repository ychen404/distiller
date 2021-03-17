import tqdm
import time
from tqdm import trange
from time import sleep



# for i in trange(100):
#     sleep(0.01)


def transfer():
    for i in trange(1000):#, position=0, desc="Transfer progress", ncols=100, bar_format='{l_bar}{bar}|', leave=True):
        time.sleep(.3)
        # print("Transfering")
        # tqdm._instances.clear()

        # interrupt()

def interrupt():
    type("File transfer interrupted, to restart the transfer, type 'restart'")


# transfer()

nums = [1,2,3,4]
for num in tqdm.tqdm(nums):
    time.sleep(0.3)

