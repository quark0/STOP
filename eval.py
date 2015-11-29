import sys

def getRankedLists(fileName):
    scoreLists = {}
    for line in open(fileName):
        pair = line.split()
        q = pair[0]
        doc = pair[1]
        score = float(pair[2])
        if q not in scoreLists:
            scoreLists[q] = [[],[]]
        scoreLists[q][0].append(doc)
        scoreLists[q][1].append(score)
    rankedLists = {}
    for q in scoreLists:
        scores, docs = zip(*sorted(zip(scoreLists[q][1],scoreLists[q][0]), reverse=True))
        rankedLists[q] = docs
    return rankedLists

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('Usage: %s ground_truth prediction' % sys.argv[0])
    predicted = getRankedLists(sys.argv[2])
    relavenceLists = {}
    for line in open(sys.argv[1]):
        pair = line.split()
        q = pair[0]
        if int(pair[2])== 1:
            if q not in relavenceLists:
                relavenceLists[q] = []
            relavenceLists[q].append(pair[1])

    for top in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        aps = []
        for q in relavenceLists:
            lhat = predicted[q]
            ltruth = relavenceLists[q]
            correct = 0.0
            sum_precision = 0.0
            for pos in range(min(top, len(lhat))):
                if lhat[pos] in ltruth:
                    correct = correct + 1
                    sum_precision = sum_precision + correct/(pos+1)
            ap = sum_precision/len(ltruth)
            aps.append(ap)
            #print "Q:", q, '\tAP:', ap
        print "MAP@%2d = %f" % (top, sum(aps)/len(aps))
