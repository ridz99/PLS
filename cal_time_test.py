'''
This script is used to calculate the total time for the test dataset to find FA
'''


import os

sum_t = 0

file_folder = '.'

drs = [dr for dr in os.listdir('TEST')]
print(drs)

os.chdir('TEST')

print(os.getcwd())

for dr in drs:
    #os.chdir(dr)
    #print(os.getcwd())
    tmp = [sub for sub in os.listdir(dr)]
    os.chdir(dr)
    print(os.getcwd())
    for sub in tmp:
        #os.chdir(sub)
        #print(os.getcwd())
        files = [f for f in os.listdir(sub) if (f.endswith(".TXT") and f!="SA1.TXT" and f!="SA2.TXT")]
        print(files)
        os.chdir(sub)
        print(os.getcwd())
        for f in files:
            fp = open(f, "r")
            data = fp.readline()
            data = data.split()
            sum_t += float(data[1])
        os.chdir('../')
        print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

print(sum_t)
