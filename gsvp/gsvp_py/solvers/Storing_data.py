
from gsvp_py.solvers.data_generator import generator1
def generators():
    data1 = ['Ovarian','breast_cancer']  
    for i in range(len(data1)):
        generator1(data1[i])


