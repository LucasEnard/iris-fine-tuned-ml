with open("misc/dataset/test.csv",'r') as test:
    with open("misc/dataset/train.csv",'r') as train:
        with open("misc/dataset/dataset.sql",'w') as dataset:
            dataset.writelines("INSERT INTO iris.Review (Label,Text) VALUES \n")
            for line in test.readlines():
                line = line.replace("'","''")
                line = list(line)
                line[0] = "'"
                line[2] = "'"
                line[4] = "'"
                line[-2] = "'"
                line = ''.join(line)
                dataset.writelines("(" + line[:-1] + "),\n")
            lines = train.readlines()
            for line in lines[:-1]:
                line = line.replace("'","''")
                line = list(line)
                line[0] = "'"
                line[2] = "'"
                line[4] = "'"
                line[-2] = "'"
                line = ''.join(line)
                dataset.writelines("(" + line[:-1] + "),\n")
            lastline = lines[-1]
            lastline = lastline.replace("'","''")
            lastline = list(lastline)
            lastline[0] = "'"
            lastline[2] = "'"
            lastline[4] = "'"
            lastline[-2] = "'"
            lastline = ''.join(lastline)
            dataset.writelines("(" + lastline[:-1] + ")\n")
            dataset.writelines("go")
