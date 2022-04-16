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
            size = flow[2]
            t_start = flow[3]     # us us us

            print("%d\t%d\t3\t100\t%d\t%f" % (src, dst, size, t_start), file=f)

def generate_workload(filename, verify_only = False):
    data = []
    line = 0
    with open(filename, "r") as f:
        raw = f.readlines()
        for row in raw:
            if line == 0:
                flowCnt = list(map(int, row.split()))
            else:
                src, dst, three, port, size, time = list(map(float, row.split()))
                data.append((dst, src, size, time))
            line += 1
    return data

files = ["facebook-cachefollower-64x64-lam1-80p-dg2.csv",
"facebook-cachefollower-64x64-lam1-80p-dg4.csv",
"google-all-rpc-80p-64x64-lam1-80p-dg2.csv",
"google-all-rpc-80p-64x64-lam1-80p-dg4.csv"]

for file in files:
    flows = generate_workload(file)
    write_traffic_flows_to_ns3txt(file, flows)



