def qag_split_pairs(text: str):
    pairs = []
    for pair in text.split('|'):
        pair_seg = pair.split('answer:')
        pair_seg2 = []
        for pr in pair_seg:
            pr = pr.strip()
            if pr.endswith(','):
                pr = pr[0:-1]
            if pr.startswith('question:'):
                pr = pr[9:]
            if pr.startswith('answer:'):
                pr = pr[7:]
            pr = pr.strip()
            pair_seg2.append(pr)
        pairs.append(pair_seg2)
    return pairs
