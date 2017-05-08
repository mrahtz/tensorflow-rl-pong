#!/usr/bin/env gnuplot

set term png
set output 'logs.png'
set xlabel "Episode number"
set ylabel "Discounted reward average"

files = system("ls -lt1 *.log | sed 's/.log/.dat/'")
labels = system("ls -lt1 *.log | sed 's/.log//'")

plot for [i=1:words(files)] word(files, i) using 1:3 with lines title word(labels, i) noenhanced
