import re, sys

data = sys.stdin.readlines()

for line in data:
    match = re.match("<s> (.+) => (.+) -\d+.\d+", line)
    obs = match.group(1)
    seq = match.group(2)
    seq_list = seq.split()[1:]
    tags = []
    for i, pos in enumerate(seq_list):
        match = re.match(".+_(.+)", pos)
        tags.append(match.group(1))

    pairs = list(zip(obs.split(), tags))
    
    for pair in pairs:
        print(f"{pair[0]}/{pair[1]} ", end="")
    print("")