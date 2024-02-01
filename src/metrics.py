import numpy as np

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
    assert len(s) == len(t)
    return (2.0/len(s)**2)*np.sum(np.abs(np.asarray(s) - np.asarray(t)))

def kendall_tau_distance(s,t):
    """
    Computes the Kendall tau distance between two full lists of ranks,
    which counts all discordant pairs (where s(i) < s(j) but t(i) > t(j),
    or vice versa) and divides by:

            k*(k-1)/2

    s,t should be array-like (lists are OK).

    If s,t are *not* full, this function should not be used.
    """
    numDiscordant = 0
    for i in range(0, len(s)):
        for j in range(i+1, len(t)):
            if (s[i] < s[j] and t[i] > t[j]) or (s[i] > s[j] and t[i] < t[j]):
                numDiscordant += 1
    return 2.0 * numDiscordant / (len(s) * (len(s)-1))

def jaccard_mod(s,t):
    found=0
    for i in range(0, len(s)):
        if t[i] in s:
            found += 1
    return 1-(found/len(s))

if __name__ == '__main__':
    c = [1,2,3,4,5]
    c1 = [1,2,3,4,5]

    print(jaccard_mod(c,d))