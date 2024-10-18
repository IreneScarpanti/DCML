#comment 
from time import sleep
import psutil
import csv

def read_cpu_usage():
    cpu_t =psutil.cpu_times()
    urs_sp_cputime = cpu_t.user
    idle_time = cpu_t.idle
    cpu_dict= {"idle_time": cpu_t.idle, "user_time": cpu_t.user}
    return cpu_dict

def write_dict_to_csv(filename, dict_item, is_firt_time):
    if is_firt_time:
       f = open(filename, 'w', newline="") 
    else:
        f = open(filename, 'a', newline="")
    w = csv.DictWriter(f, dict_item.keys())
    if is_firt_time:
        w.writeheader()
    w.writerow(dict_item)
    f.close()


if __name__ == "__main__":
    is_first_time=True
    while True:
        cpu_dict = read_cpu_usage()
        write_dict_to_csv("my_first_dataset", cpu_dict, is_first_time)
        is_first_time=False
        print(cpu_dict)
        sleep(1)

