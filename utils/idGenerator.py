from nanoid import generate

def generateId():
    return generate(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", size=16)