import pickle

result = { "lion": "yellow", "kitty": "red" }
num = 200
st = "src/results" + str(num) + ".p"
pickle.dump(result,open(st,"wb"))
read = pickle.load(open(st, "rb"))
print(read)
