
from gsvp_py.solvers.data_generator import generator0,generator1, generator2, generator3, generator4
def generators():
    data0 = ['Alpaca','Camel','Camelid']
    data1 = ['Ovarian','Prostate_GE','SMK_CAN_187','GLI_85', 'Leukemia', 
            'diabetes','heart_cleveland','breast_cancer', 'Influenza_I', 'Influenza_II']
    data2 = ['Influenza_III']
    data3 = ['Influenza_IV', 'Influenza_V']
    data4 = ['Influenza_VI']
    
    for i in range(len(data0)):
        generator0(data0[i])
        

    for i in range(len(data1)):
        generator1(data1[i])

    generator2(data2[0])

    for i in range(len(data3)):
        generator3(data3[i])

    generator4(data4[0])

