def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def computeavgjac(sets):
    num_sets = len(sets)
    total_similarity = 0
    for i in range(num_sets):
        for j in range(i + 1, num_sets):
            similarity = jaccard_similarity(sets[i], sets[j])
            total_similarity += similarity
    if num_sets == 1:
        average_similarity = total_similarity
    else:
        average_similarity = total_similarity / (num_sets * (num_sets - 1) / 2)
    return  average_similarity