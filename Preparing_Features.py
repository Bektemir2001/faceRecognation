# Ар бир сүрөт үчүн анын өзгөчөлүктөрүн(Features) өзүнчө файл кылып жасабыз
# version-01.txt файлынан ар бирин алып өзүнчө файлга xxxx.txt деген ат менен жасабыз
# Ал файлдар бизге Dataset түзүү үчүн керек
# 0001.txt туура келген сүрот Clean_0001_001_20050913115022_Range.png
# ...................................................................
# 0498.txt туура келген сүрот Clean_0498_101_20050913155318_Range.png
# ...................................................................
# 1149.txt туура келген сүрот Clean_1149_118_20050912204835_Range.png

f = open('/home/bektemir/Desktop/my_projects/faceRecognation/version-01.txt','r')
L = f.readlines()
f.close()
for line_string in L:
    if line_string[0]=="C":
        j = 0
        f = open('/home/bektemir/Desktop/my_projects/faceRecognation/features/'+line_string[6:10]+'.txt','w')
        continue
    if line_string[0]=="2":
        continue
    j +=1
    f.write(line_string)
    if j == 9:
        f.close()
        j  = 0
