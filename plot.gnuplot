set term png
set output 'plots.png'
set xlabel "Episode number"
set ylabel "Running reward average"

files = system("ls -lt1 *.log | tr 'log' 'dat' | head -n 5")
labels = system("ls -lt1 *.log | tr 'log' 'dat' | head -n 5 | sed 's/\.dat//g'")

plot for [i=1:words(files)] word(files, i) using 1:3 with lines title word(labels, i) noenhanced
