from multiprocessing import Process
import time
import get_i2v

if __name__ == "__main__":
    p = Process(target = get_i2v.main)
    p.start()

    while True:
        time.sleep(10)
        if p.is_alive():
            continue
        else:
            with open('cache','r') as cache:
                cache_set = cache.readlines()
            if "finish" in cache_set[-1]:
                break
            p = Process(target = get_i2v.main)
            p.start()
    