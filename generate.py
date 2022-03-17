import os
import sys
import math
import argparse

from numpy import random

MSG_SIZE = 1


############################################################
#                        utility
############################################################
# def write_traffic_matrix_to_csv(file, matrix):
#     matrix.to_csv(file)

# def write_traffic_flows_to_txt(file, src, dst, flows):
#     src_n = len(src)
#     dst_n = len(dst)
#     flow_n = len(flows)
#
#     with open(file, "w") as f:
#         print(src_n, dst_n, flow_n, file=f)
#         print(" ".join(src), file=f)
#         print(" ".join(dst), file=f)
#         for flow in flows:
#             print(" ".join(map(str,flow)), file=f)

def write_traffic_flows_to_ns3txt(file, flows):
    flow_n = len(flows)

    with open(file, "w") as f:
        print(flow_n, file=f)
        for flow in flows:
            src = flow[0]
            dst = flow[1]
            size = flow[2] * MSG_SIZE
            t_start = flow[3] / 1000000 + 2     # us us us

            print("%s\t%s\t3\t100\t%d\t%f" % (src, dst, size, t_start), file=f)

def generate_workload(filename, verify_only = False):
    data = []
    data.append((0,0))
    with open(filename, "r") as f:
        raw = f.readlines()
        for row in raw:
            size, p = list(map(float, row.split()))
            data.append((size, p))

    def once(rnd = random):
        rand = rnd.uniform(0,1)
        last = 0
        for i in range(len(data)-1):
            if rand > data[i][1] and rand < data[i+1][1]:
                size_diff = data[i+1][0] - data[i][0]
                p_diff = data[i+1][1] - data[i][1]
                return int(data[i][0] + (rand-data[i][1])/p_diff*size_diff)
        return -1

    def verify(rnd = random):
        import numpy as np
        d = []
        for i in range(10000):
            r = once(rnd)
            print(r)
            d.append(int(r))

        x = np.sort(d)
        y = 1. * np.arange(len(d)) / (len(d) - 1)

        import matplotlib.pyplot as plt

        plt.axes(xscale="log")
        plt.plot([row[0] for row in data], [row[1] for row in data], label="raw")
        plt.plot(x, y, label="regenerate")
        plt.legend()
        plt.title(filename)
        plt.savefig(filename+".png")
        # plt.show()

    if verify_only:
        return verify

    ### return a function
    return once

def cal_load(lam, interval, mean_size, capacity):
    return lam  * mean_size / interval / capacity

def workload_mean_size(filename):
    data = []
    data.append((0,0))
    with open(filename, "r") as f:
        raw = f.readlines()
        for row in raw:
            size, p = list(map(float, row.split()))
            data.append((size, p))
    mean_size = 0
    for i in range(1, len(data)):
        size, p = data[i][0], data[i][1] - data[i-1][1]
        mean_size += size*p
    return mean_size * MSG_SIZE


############################################################
#                deprecated generator
############################################################
class DeprecatedGenerator:
    def __init__(self):
        import pandas as pd

    def gen_traffic_matrix(profile):
        src = ["s"+str(i) for i in range(profile["n_tx"])]
        dst = ["d"+str(i) for i in range(profile["n_rx"])]
        df = pd.DataFrame(columns=src, index=dst, data=0)
        rs = profile['random']

        flow_cnt = 0
        while flow_cnt < profile["n_flow"]:
            src_i = dst_i = -1
            while(True):        # allocate source
                rnd = profile["dist_tx"](rs)
                src_i = int(rnd)
                if (src_i < profile["n_tx"] and src_i >= 0):
                    break

            while (True):  # allocate destination
                rnd = profile["dist_rx"](rs)
                dst_i = int(rnd)
                if (dst_i < profile["n_rx"] and dst_i >= 0):
                    break
            src_idx, dst_idx = src[src_i], dst[dst_i]
            if df.at[dst_idx, src_idx] <= 0:
                size = int(profile["dist_size"](rs))
                df.at[dst_idx, src_idx] =  size# if (src, dst) have allocated a flow with higher priority (letter size), skip
                flow_cnt += 1
        return df

    def gen_traffic_flows_with_CBS(profile, t):
        src = ["s"+str(i) for i in range(profile["n_tx"])]
        dst = ["d"+str(i) for i in range(profile["n_rx"])]
        rs = profile['random']
        flow_list = []

        for i in range(profile["n_tx"]):
            src_i = i
            n_flow = 1
            if 'poisson_tx' in profile:
                poisson = profile['poisson_tx']
                n_flow = poisson(rs)*2
            size = int(profile["dist_size"](rs))
            # CBS copy flows
            while(n_flow > 0):
                n_flow -= 1
                while (True):  # allocate destination
                    rnd = profile["dist_rx"](rs)
                    dst_i = int(rnd)
                    if (dst_i < profile["n_rx"] and dst_i >= 0):
                        break
                src_idx, dst_idx = src[src_i], dst[dst_i]
                flow_list.append((src_idx, dst_idx, size, t))
        return src, dst, flow_list

    def gen_traffic_flows_with_poisson(profile, t):
        src = ["s"+str(i) for i in range(profile["n_tx"])]
        dst = ["d"+str(i) for i in range(profile["n_rx"])]
        rs = profile['random']
        flow_list = []

        for i in range(profile["n_flow"]):
            size = i
            src_i = dst_i = -1
            while (True):  # allocate source
                rnd = profile["dist_tx"](rs)
                src_i = int(rnd)
                if (src_i < profile["n_tx"] and src_i >= 0):
                    break

            n_flow = 1
            if 'poisson_tx' in profile:
                poisson = profile['poisson_tx']
                n_flow = poisson(rs)

            while(n_flow > 0):
                n_flow -= 1
                while (True):  # allocate destination
                    rnd = profile["dist_rx"](rs)
                    dst_i = int(rnd)
                    if (dst_i < profile["n_rx"] and dst_i >= 0):
                        break
                src_idx, dst_idx = src[src_i], dst[dst_i]
                size = int(profile["dist_size"](rs))
                flow_list.append((src_idx, dst_idx, size, t))
        return src, dst, flow_list

    def gen_traffic_flows(profile, t=0):
        src = ["s"+str(i) for i in range(profile["n_tx"])]
        dst = ["d"+str(i) for i in range(profile["n_rx"])]
        rs = profile['random']
        flow_list = []

        for i in range(profile["n_flow"]):
            size = i
            src_i = dst_i = -1
            while (True):  # allocate source
                rnd = profile["dist_tx"](rs)
                src_i = int(rnd)
                if (src_i < profile["n_tx"] and src_i >= 0):
                    break

            while (True):  # allocate destination
                rnd = profile["dist_rx"](rs)
                dst_i = int(rnd)
                if (dst_i < profile["n_rx"] and dst_i >= 0):
                    break
            src_idx, dst_idx = src[src_i], dst[dst_i]
            size = int(profile["dist_size"](rs))
            flow_list.append((src_idx, dst_idx, size, t))
        return src, dst, flow_list

    def gen_traffic_triple(profile):
        src = ["s"+str(i) for i in range(profile["n_tx"])]
        dst = ["d"+str(i) for i in range(profile["n_rx"])]
        df = pd.DataFrame(columns=src, index=dst, data=0)
        rs = profile['random']

        def gen_outcast(n_outcast):
            src_i = -1
            while(True):        # allocate source
                rnd = profile["dist_tx"](rs)
                src_i = int(rnd)
                if (src_i < profile["n_tx"] and src_i >= 0):
                    break

            while(n_outcast > 0):

                dst_i = -1
                while (True):  # allocate destination
                    rnd = profile["dist_rx"](rs)
                    dst_i = int(rnd)
                    if (dst_i < profile["n_rx"] and dst_i >= 0):
                        break

                src_idx, dst_idx = src[src_i], dst[dst_i]

                if df.at[dst_idx, src_idx] <= 0:
                    size = int(profile["dist_size"](rs))
                    df.at[dst_idx, src_idx] =  size 
                    n_outcast -= 1
        
        def gen_incast(n_incast):
            dst_i = -1
            while(True):        # allocate source
                rnd = profile["dist_rx"](rs)
                dst_i = int(rnd)
                if (dst_i < profile["n_rx"] and dst_i >= 0):
                    break

            while(n_incast > 0):
                src_i = -1
                while (True):  # allocate destination
                    rnd = profile["dist_tx"](rs)
                    src_i = int(rnd)
                    if (src_i < profile["n_tx"] and src_i >= 0):
                        break

                src_idx, dst_idx = src[src_i], dst[dst_i]

                if df.at[dst_idx, src_idx] <= 0:
                    size = int(profile["dist_size"](rs))
                    df.at[dst_idx, src_idx] =  size 
                    n_incast -= 1

        flow_cnt = profile["n_flow"]
        inoutcast_degree = 10


        while flow_cnt > 0 :
            fr = rs.randint(0, 3)
            n = min(flow_cnt, inoutcast_degree)

            if (fr == 0):
                gen_outcast(n)
            elif (fr == 1):
                gen_incast(n)
            elif (fr == 2):
                n = 1
                gen_outcast(n)

            flow_cnt -= n

        return df

    def gen_traffic_all_to_all(profile, t=0):
        s_start = SRC_START
        src = [str(s_start + i) for i in range(profile["n_tx"])]
        d_start = DST_START
        dst = [str(d_start + i) for i in range(profile["n_rx"])]

        rs = profile['random']
        flow_list = []

        for src_idx in src:
            for dst_idx in dst:
                size = int(profile["dist_size"](rs)) + 1

                flow_list.append((src_idx, dst_idx, size, t))

        return src, dst, flow_list

    def gen(profile, filename):
        df = gen_traffic_triple(profile)
        write_traffic_matrix_to_csv(filename, df)

    def gen_txt(profile, filename):
        src, dst, flow_list = gen_traffic_all_to_all(profile, 2000000000)
        write_traffic_flows_to_ns3txt(filename, src, dst, flow_list)


class All2allGenerator:
    src_start = 1
    dst_start = 1
    time_end = 0
    nflow_end = 0
    n_src = 0
    n_dst = 0
    poisson = None
    size_gen = None
    size_mean = None
    load = None
    capacity = None

    def __init__(self):
        self.random = random.RandomState(0)

    def set_random_seed(self, seed):
        self.random = random.RandomState(seed)

    def set_n_node(self, n):
        self.n_src = n
        self.n_dst = n

    def set_poisson(self, poisson):
        self.poisson = poisson

    def set_size_gen(self, func):
        self.size_gen = func

    def set_size_mean(self, mean):
        self.size_mean = mean

    def set_load(self, load):
        self.load = load

    def set_port_capacity(self, cap):
        self.capacity = cap

    def set_time_end(self, end):
        self.time_end = end

    def set_nflow_end(self, end):
        self.nflow_end = end

    def get_interval(self):
        assert self.poisson and self.size_mean and self.capacity and self.load
        size_once = self.poisson * self.size_mean        # mean size generate once
        cap_time_unit = self.capacity * self.load
        interval = size_once / cap_time_unit
        interval_ceil = math.ceil(interval)
        if interval/interval_ceil < 0.9:
            print("Real generated load with an error > 10%%, raw %f, real %f" % (interval, interval_ceil))

        real_load = self.load * interval/interval_ceil
        print("load --- raw  %f, real %f" % (self.load, real_load))
        return interval_ceil

    def generate(self):
        assert bool(self.nflow_end or self.time_end)

        flow_list = []
        clock = 0
        interval = self.get_interval()

        while True:
            flow_once = self.do_gen()
            [l.append(clock) for l in flow_once]

            flow_list.extend(flow_once)
            clock += interval
            count = len(flow_list)
    
            if self.nflow_end and count >= self.nflow_end:
                break
            if self.time_end and clock >= self.time_end:
                break
            
        return flow_list

    def do_gen(self):
        flows = []
        for src_i in range(self.n_src):
            n = self.random.poisson(lam=self.poisson)
            while n > 0:
                dst_i = int(self.random.uniform(low=0, high=self.n_dst))
                if src_i != dst_i:
                    size = int(self.size_gen(self.random))
                    flow = [self.src_start + src_i, self.dst_start + dst_i, size]
                    flows.append(flow)
                    n -= 1
        return flows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', dest='seed', action='store', default=0, help="Specify the random seed.")
    parser.add_argument('--workload', dest='workload', action='store', default=None, help="Specify the traffic workload input.")
    parser.add_argument('--output_dir', dest='output_dir', action='store', default=None, help="Specify the output directory.")
    parser.add_argument('--nodes', dest='nodes', action='store', default=None, help="Specify the node numbers.")
    parser.add_argument('--load', dest='load', action='store', default=None, help="Specify the traffic load.")
    parser.add_argument('--poisson', dest='poisson', action='store', default=None, help="Specify the poisson.")
    args = parser.parse_args()

    workload_file = args.workload
    mean = workload_mean_size(workload_file)
    work = generate_workload(workload_file)
    print("workload %s, mean %f" % (workload_file, mean))

    nodes = int(args.nodes)
    load = float(args.load)
    poisson = float(args.poisson)
    output_base = args.output_dir
    output_file = os.path.join(output_base, os.path.basename(workload_file).split('.')[0] + "-%dx%d-%dp.csv" % (nodes, nodes, int(load * 100)))
    print("output", output_file)

    g = All2allGenerator()
    g.set_n_node(nodes)
    g.set_poisson(poisson)
    g.set_size_mean(mean)
    g.set_load(load)
    g.set_port_capacity(12500)
    g.set_size_gen(work)
    g.set_time_end(500)

    flows = g.generate()
    write_traffic_flows_to_ns3txt(output_file, flows)
    # print(g.get_interval())
    # print(flows)


