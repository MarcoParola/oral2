import numpy as np

# [REM] ? -> it doesn't work with duplicates
def spearman_footrule_distance(s,t):
    """
    Computes the Spearman footrule distance between two full lists of ranks:

        F(s,t) = (2/|S|^2)*sum( |s(i) - t(i)| )

    the normalized sum over all elements in a set of the absolute difference
    between the rank according to two different lists s and t.  As defined,
    0 <= F(s,t) <= 1.

    s,t should be array-like (lists are OK).

    If s,t are *not* full, this function should not be used.
    """
    # check that the lists are both full
    assert len(s) == len(t)
    return (2.0/len(s)**2)*np.sum(np.abs(np.asarray(s) - np.asarray(t)))


def kendall_tau_distance(s,t):
    """
    Computes the Kendall tau distance between two full lists of ranks,
    which counts all discordant pairs (where s(i) < s(j) but t(i) > t(j),
    or vice versa) and divides by:

            k*(k-1)/2

    This is a slow version of the distance; a faster version can be
    implemented using a version of merge sort (TODO).

    s,t should be array-like (lists are OK).

    If s,t are *not* full, this function should not be used.
    """
    num_concordant = 0
    num_discordant = 0

    for i in range(len(s)):
        for j in range(i + 1, len(t)):
            if s[i] == s[j] and t[i] == t[j]:
                continue

            if (s[i] < s[j] and t[i] < t[j]) or (s[i] > s[j] and t[i] > t[j]):
                num_concordant += 1
            else:
                num_discordant += 1

    tau = (num_concordant - num_discordant) / (num_concordant + num_discordant)

    normalized_tau = (1 - tau) / 2

    return normalized_tau

def jaccard_mod(s,t,l):
    for i in range(0, len(t)):
        if t[i] in s and t[i]!= l+1:
            found += 1
    # percentage of ranked images not found
    return 1-(found/l)

def spearman_footrule_distance_mod(s,t):
    # describe the waste from the ranked position to reach the correct rank
    # describe the distance to bring the element in the correct position in the rank
    mox = 0 
    reversed = list(s)
    reversed.reverse()
    for i in range(0, len(s)):
      mox += abs(s[i] - reversed[i])

    difference = 0
    for i in range(0, len(t)):
        difference += abs(s[i] - t[i])

    return difference/mox

if __name__ == '__main__':
    a = [1,2,3,4,5,6,6,6]
    a1 = [1,2,3,4,5,6,6,6]
    b = [6,6,6,5,4,3,2,1]
    c = [1,2,3,4,5]
    c1 = [1,2,3,4,5]
    d = [5,4,3,2,1]
    e = [1,2,3,4,5,6,6]
    f= [5,2,6,4,6,3,5]
    print(spearman_footrule_distance_mod(e,f))
