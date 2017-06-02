import sys

f = open(sys.argv[1])
out = open('ambiguous.txt', 'w')

fixed_active = f.readline().split()
fixed_passive = f.readline().split()
scanning_active = f.readline().split()
scanning_passive = f.readline().split()

fixed = fixed_active + fixed_passive
scanning = scanning_active + scanning_passive

for fr in fixed_active:
    out.write('restraint (')
    out.write("{}) (".format(fr))
    out.write(" or ".join(scanning))
    out.write(") 0.0 2.0\n")

for sr in scanning_active:
    out.write('restraint (')
    out.write(" or ".join(fixed))
    out.write(") ({}) 0.0 2.0\n".format(sr))

out.close()
f.close()
