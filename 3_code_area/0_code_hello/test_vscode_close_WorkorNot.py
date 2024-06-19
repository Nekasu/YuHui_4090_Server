from time import sleep


path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_vscode_close_WorkorNot.txt'

with open(file=path,mode='w') as f:
    for i in range(0,1000):
        f.write(str(i ))
        f.write('\n')
        sleep(1)
        
