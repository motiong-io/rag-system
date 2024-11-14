import hashlib

def md5hash(input_string: str) -> str:
    md5_object = hashlib.md5()
    md5_object.update(input_string.encode('utf-8'))
    return md5_object.hexdigest()



if __name__ == "__main__":
    print(md5hash("https://en.wikipedia.org/wiki/Python_(programming_language)"))  