def flatten_list(nested_list):
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def flatten_list_comprehension(nested_list):
    return [item for sublist in nested_list for item in sublist]



def char_count(s):
    count = {}
    for char in s:
        if char.islower() or char == ' ':
            if char in count:
                count[char] += 1
            else:
                count[char] = 1
    return count

def char_count_comprehension(s):
    return {char: s.count(char) for char in set(s) if char.islower() or char == ' '}

