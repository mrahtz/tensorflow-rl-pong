#!/bin/bash

for log in *.log; do
    grep 'Reward total' "$log" \
        | awk '{print $4, $11}' \
        | sed 's/\;//g' \
        | awk '{print NR " " $0}' \
        > "${log%.log}.dat"
done
