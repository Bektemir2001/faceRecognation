# =============================================================================
# # Dataset түзүү баардык мүмкүн болгон комбинацияларды алабыз
# 0001.txt + 0001.txt --> 0001-0001.txt
# 0001.txt + 0002.txt --> 0001-0002.txt
# ...................................................................
# 0001.txt + 1148.txt --> 0001-1148.txt
# 0001.txt + 1149.txt --> 0001-1149.txt
# 0002.txt + 0001.txt --> 0001-0001.txt
# 0002.txt + 0002.txt --> 0002-0002.txt
# ...................................................................
# 0002.txt + 1148.txt --> 0002-1148.txt
# 0002.txt + 1149.txt --> 0002-1149.txt
# ...................................................................
# 1149.txt + 1148.txt --> 1149-1148.txt
# 1149.txt + 1149.txt --> 1149-1149.txt
# Баардыгы 1149*1149=1 320 201 файл болот.
# Меткаларды төмөнкүдөй аныктайбыз.
# Файлдын атындагы (XXXX-XXXX.txt) дефиске чейинки сан менен
# дефистен кийинки сан барабар болсо 1, ар турдүү болсо 0 деп эсептейбиз
# Баардыгы 1149*1149=1 320 201 файл болот(убакыт 0:33:25.597975 болду).

# Features папкасынан файлдарды окуп, аларды бириктирип Dataset папкасына жазабыз

from datetime import datetime
import os
start_time = datetime.now()
dirname = "/home/bektemir/Desktop/my_projects/faceRecognation/features"
RangeFile = os.listdir(dirname)
i = 0
for file1 in RangeFile:
    f1 = open('/home/bektemir/Desktop/my_projects/faceRecognation/features/'+file1,'r')
    f1file = f1.read()
    ftemp = file1[0:4]
    for file2 in RangeFile:
        ftarget = ftemp+ '-' + file2
        f2 = open('/home/bektemir/Desktop/my_projects/faceRecognation/features/'+file2,'r')
        f2file = f2.read()
        f  = open('/home/bektemir/Desktop/my_projects/faceRecognation/dataset/'+ftarget,'w')
        f2.close()
        f.write(f1file+f2file)
        f.close()
    f1.close()
    print("iteration ", i)
    i += 1
print(datetime.now() - start_time) #0:33:25.597975
