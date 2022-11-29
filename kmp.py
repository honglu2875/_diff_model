from functools import lru_cache


@lru_cache(maxsize=None)
def build_nxt(pattern: tuple) -> tuple:
    # The function is being cached. Use tuple to avoid the cache being tampered out of scope.
    nxt = [0]
    current = 1
    match_idx = 0
    
    while current < len(pattern):
        if pattern[match_idx] == pattern[current]:
            current += 1
            match_idx += 1
            nxt.append(match_idx)
        elif match_idx != 0:
            match_idx = nxt[match_idx - 1]
        else:
            nxt.append(0)
            current += 1
            
    return tuple(nxt)


def kmp(seq, pattern, first_appearance=False):
    nxt = build_nxt(tuple(pattern))
    current = 0
    match_idx = 0
    
    matched = []
    
    while current < len(seq):
        if seq[current] == pattern[match_idx]:
            current += 1
            match_idx += 1
        elif match_idx != 0:
            match_idx = nxt[match_idx - 1]
        else:
            current += 1
            
        if match_idx == len(pattern):
            matched.append(current - len(pattern))
            if first_appearance:
                return matched
            match_idx = nxt[match_idx - 1]
         
    return matched
    


if __name__=='__main__':
    p1 = "aa"
    t1 = "aaaaaaaa"
    assert(kmp(t1, p1) == [0, 1, 2, 3, 4, 5, 6])

    p2 = "abc"
    t2 = "abdabeabfabc"

    assert(kmp(t2, p2) == [9])

    p3 = "aab"
    t3 = "aaabaacbaab"

    assert(kmp(t3, p3) == [1, 8])

    p4 = "11"
    t4 = "111"
    assert(kmp(t4, p4) == [0, 1])
    
    t5 = list("ABC ABCDAB ABCDABCDABDE")
    p5 = list("ABCDABD")
    assert (kmp(t5, p5) == [15])
    
    t6 = list("ABC ABCDAB ABCDABCDABDE")
    p6 = list("ABCDABD")
    assert (kmp(t6, p6, first_appearance=True) == [15])
