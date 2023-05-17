
for w0 in -2 -1 0 1 2
do
for w1 in -2 -1 0 1 2
do
for w2 in -2 -1 0 1 2
do
for w3 in -2 -1 0 1 2
do
 	./build/bin/mjpc --task="Quadruped Flat" --steps=100 --horizon=0.35 --w0=$w0 --w1=$w1 --w2=$w2 --w3=$w3
done
done
done
done

