#comment 
from time import sleep
import psutil
def read_cpu_usage():
    cpu_t =psutil.cpu_times()
    urs_sp_cputime = cpu_t.user
    idle_time = cpu_t.idle
    return urs_sp_cputime, idle_time

if __name__ == "__main__":
    while True:
        u_t, i_t = read_cpu_usage()
        #print(str(u_t) +" " + str(i_t))
        print ("user time: %.2f" % (u_t))
        sleep(1)

