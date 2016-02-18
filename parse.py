

infile  = open('submit.csv','r')
outfile = open('test.txt','w') 

head = infile.readline()
outfile.write(head)

while True:
    y = infile.readline()
    x = y.rstrip()
    x = x.strip('\n')
    x = x.replace('\n', '')
    if not x: break
 
    cols = x.split(',')

    num = cols[0].split('_')[0]
    num = int(num)

    if (num >= 501):
      #print num

      z = y.replace('systolic' , 'Systole')
      z = z.replace('diastolic', 'Diastole')

      print z

      outfile.write(z)

