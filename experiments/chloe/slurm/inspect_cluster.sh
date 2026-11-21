#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/chloe/slurm
{
  date -Is
  sinfo -o '%P %a %l %D %G'
  scontrol show partition MGPU-TC2
  sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS,DefaultQOS
  sacctmgr -nP show qos normal format=Name,MaxWall,MaxTRESPJ,MaxTRESPU,MaxJobsPU,MaxSubmitPU
  scontrol show node
} > outputs/chloe/slurm/cluster-inspection.txt
cat outputs/chloe/slurm/cluster-inspection.txt
