import pandas
import numpy

from factor_analyzer import FactorAnalyzer

import matplotlib.pyplot as pyplot



dataset  = pandas.read_csv("bfi.csv")


dataset.drop(['gender','age','education'], axis=1, inplace=True)
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

dataset.dropna(inplace=True)

print(dataset)


machine = FactorAnalyzer(n_factors=25, rotation=None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()
print(ev)

pyplot.scatter(range(1,dataset.shape[1]+1), ev)
pyplot.savefig("plot.png")
pyplot.close()


machine = FactorAnalyzer(n_factors=6, rotation='varimax')
machine.fit(dataset)
output = machine.loadings_
numpy.set_printoptions(suppress=True)
print(output)



machine = FactorAnalyzer(n_factors=5, rotation='varimax')
machine.fit(dataset)
loadings = machine.loadings_

print("\nfactor loadings:\n")
print(loadings)
print(machine.get_factor_variance())

dataset = dataset.values

result = numpy.dot(dataset, loadings)

print(result)
print(result.shape)











