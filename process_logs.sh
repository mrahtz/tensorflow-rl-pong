#!/bin/bash

for log in *.log; do
    grep 'Reward total was' "$log" \
        | awk '{print $4, $10}' \
        | sed 's/\; / /g' \
        | awk '{print NR " " $0}' \
        > "${log%.log}.dat"
done
