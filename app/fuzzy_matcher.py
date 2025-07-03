def edit_distance(s1, s2):
    """
    Compute the Levenshtein edit distance between two strings s1 and s2.
    The function is case-insensitive. Used to find the closest matching movie 
    titles based on user input.

    Returns the minimum number of operations (insert, delete, substitute)
    required to transform s1 into s2.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j  

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1].lower() == s2[j - 1].lower(): 
                dp[i][j] = dp[i - 1][j - 1]
            else:
                insert = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                replace = dp[i - 1][j - 1] + 1
                dp[i][j] = min(insert, delete, replace)

    return dp[m][n]


def best_match_by_edit_distance(input_title, all_titles, threshold=3):
    """
    Find the closest match to input_title from a list of titles using edit distance.

    Parameters:
    - input_title: str, the user's input title to match against
    - all_titles: List[str], candidate titles to compare against
    - threshold: int, maximum edit distance to accept a match

    Returns:
    - The best matching title (str) if within threshold, otherwise None.
    """
    distances = [(title, edit_distance(input_title.lower(), title.lower())) for title in all_titles]
    distances.sort(key=lambda x: x[1])
    best_title, best_score = distances[0]
    return best_title if best_score <= threshold else None